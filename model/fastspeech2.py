import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from simalign import SentenceAligner

from transformer import Encoder, Decoder, PostNet, ProsodyExtractor, ProsodyPredictor
from .modules import VarianceAdaptor
from utils.tools import get_mask_from_lengths

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


class FastSpeech2Pros(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config, pretrain=False):
        super(FastSpeech2Pros, self).__init__()

        self.model_config = model_config
        self.preprocess_config = preprocess_config
        self.encoder = Encoder(model_config)
        self.prosody_predictor = ProsodyPredictor(model_config)
        self.prosody_extractor = ProsodyExtractor(model_config)
        self.h_sd_downsize = nn.Linear(model_config["prosody_predictor"]["sd_dim_in"],
                                        model_config["prosody_extractor"]["dim_out"])
        self.h_sd_downsize2 = nn.Linear(model_config["prosody_predictor"]["sd_dim_in"],
                                        model_config["prosody_extractor"]["dim_out"])
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()

        self.beta = nn.Parameter(torch.tensor(-1.0))

        # Only need this if I'm getting speaker embeddings path from json
        with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json"), "r") as f:
            self.speakers_json = json.load(f)

        # Freeze the weights of the prosody_predictor and prosody_extractor during full training
        if not pretrain:
            for param in self.prosody_predictor.parameters():
                param.requires_grad = False

            for param in self.prosody_extractor.parameters():
                param.requires_grad = False

    def forward(self, langs, texts, text_lens, max_text_len, mels=None, mel_lens=None, max_mel_len=None,
                speaker_embs=None, alignments=None, p_targets=None, e_targets=None, 
                d_targets=None, d_src=None, p_control=1.0, e_control=1.0, d_control=1.0,):

        pretraining = False
        if d_src is None:
            d_src = d_targets
            pretraining = True
        batch_size, src_seq_length = d_src.size(0), d_src.size(1)
        tgt_masks = get_mask_from_lengths(text_lens, max_text_len)
        # print("Tgt masks shape: ", tgt_masks.shape)
        
        # Mel Mask changes in reverse direction
        # print("Max mel len: ", max_mel_len)
        # print("Mel lens: ", mel_lens)
        mel_masks = (get_mask_from_lengths(mel_lens, max_mel_len) if mel_lens is not None else None)
        # print("Mel masks shape: ", mel_masks.shape)
        
        # This should be tgt translations not src texts
        output = self.encoder(texts, tgt_masks)  # torch.Size([Batch, seq_len, 256])

        speaker_embs = speaker_embs.unsqueeze(1).expand(-1, output.size()[1], -1)
        
        # h_sd = output + speaker_embs  # torch.Size([Batch, tgt_seq_len, 256])
        h_sd = torch.cat((output, speaker_embs), dim=-1)
        # print("h_sd shape: ", h_sd.shape, torch.isnan(h_sd).any())

        h_si = output

        e_tgt = self.prosody_predictor(h_sd, h_si)
        # print("e_tgt[0] (log_pi) shape: ", e_tgt[0].shape)  # torch.Size([Batch, tgt_seq_len, N_Components])
        # print("e_tgt[1] (mu) shape: ", e_tgt[1].shape)
                
        mels = mels.unsqueeze(1) # mels shape is [batch_size, 1, melspec W, melspec H]
        enhanced_mels = self.prosody_extractor.add_lang_emb(mels, langs)
        
        e_src = self.prosody_extractor(enhanced_mels)   # e is [batch_size, melspec H, melspec W, 128]
                
        # Split phone pros embeddings by phone duration
        # [batch_size (list), phoneme_sequence_length (list), melspec H (tensor), melspec W (tensor), 128 (tensor)]        
        e_k_src = self.prosody_extractor.split_phones(e_src, d_src)  
        
        # For calculating Lpp loss:
        agg_extracted_prosody = torch.zeros(batch_size, src_seq_length, 256).to(device)
        for b in range(batch_size):
            for i in range(len(e_k_src[b])):
                if e_k_src[b][i].shape[0] == 0 or e_k_src[b][i].shape[1] == 0:
                    agg_extracted_prosody[b,i,:] = torch.zeros(256)
                else:
                    agg_extracted_prosody[b,i,:] = torch.mean(e_k_src[b][i], dim=(0, 1))

        agg_extracted_prosody = agg_extracted_prosody.detach()

        h_sd = self.h_sd_downsize(h_sd) # 512 to 256
        if pretraining:
            h_sd = torch.cat((h_sd, agg_extracted_prosody), dim=-1) # torch.Size([Batch, tgt_seq_len, 512]
        else: # Full training
            # TODO: Allow for new predicted_prosodies_tgt shape
            tgt_samp = self.prosody_predictor.sample2(e_tgt) # torch.Size([2, 88, 256])
            # print("tgt_samp shape: ", tgt_samp.shape, torch.isnan(tgt_samp).any())  

            # print("alignments shape: ", alignments.shape)  # TODO: unpad alignments for realigner otherwise everything mapped to 0.
            adjusted_e_tgt = self.prosody_predictor.prosody_realigner(alignments, tgt_samp, e_k_src, self.beta)
            # print("alignments nan", torch.isnan(alignments).any())

            # Concat
            h_sd = torch.cat((h_sd, adjusted_e_tgt), dim=-1)  # torch.Size([Batch, tgt_seq_len, 512]
            # h_sd = h_sd + adjusted_e_tgt
            # print("Output shape after prosody: ", output.shape, torch.isnan(output).any())  # torch.Size([Batch, tgt_seq_len, 256])

        # Now double check that durations and pitch etc are same as seq_length
        # print("Input mel_masks shape: ", mel_masks.shape)
        # print("Mel masks", mel_masks)

        h_sd = self.h_sd_downsize2(h_sd) # 512 to 256
        (output, p_predictions, e_predictions, log_d_predictions, d_rounded, mel_lens, mel_masks,) = \
            self.variance_adaptor(h_sd, tgt_masks, mel_masks, max_mel_len, p_targets, e_targets, d_targets, p_control,
                                  e_control, d_control, )

        # print("d_rounded shape: ", d_rounded.shape)
        # print("mel_lens shape: ", mel_lens)
        # print("predicted mel_masks shape: ", mel_masks.shape)

        output, mel_masks = self.decoder(output, mel_masks)
        # print("output mel_masks shape: ", mel_masks.shape)

        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        # For calculating Lpp loss:
        # y_e_tgt = prosody_extractor.prosody_realigner(alignments, e_k_src)

        return (output, postnet_output, p_predictions, e_predictions, log_d_predictions, d_rounded, tgt_masks,
                mel_masks, text_lens, mel_lens, agg_extracted_prosody, e_tgt)


class Discriminator(nn.Module):
    def __init__(self, preprocess_config):
        super(Discriminator, self).__init__()

        # Define the layers for your Discriminator
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Adding adaptive pooling layer
        self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 5))

        self.fc_layers = nn.Sequential(
            nn.Linear(512*5*5, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 1),
            nn.Sigmoid()  # Output will be between 0 and 1
        )

    def forward(self, mels):
        mels = mels.unsqueeze(1).permute(0, 1, 3, 2) # [batch_size, n_mel_channels, mel_length]
        
        # Forward pass through the network
        out = self.conv_layers(mels)
        out = self.adaptive_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc_layers(out)
        return out
