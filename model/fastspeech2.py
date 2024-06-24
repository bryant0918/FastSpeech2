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

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2Pros, self).__init__()

        self.model_config = model_config
        self.preprocess_config = preprocess_config
        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()

        # Only need this if I'm getting speaker embeddings path from json
        with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json"), "r") as f:
            self.speakers_json = json.load(f)

    def forward(self, texts, src_lens, max_src_len, mels=None, mel_lens=None, max_mel_len=None,
                speaker_embs=None, alignments=None, p_targets=None, e_targets=None, d_targets=None, 
                d_src=None, p_control=1.0, e_control=1.0, d_control=1.0,):

        pretraining = False
        if d_src is None:
            d_src = d_targets
            pretraining = True

        # Get masks
        batch_size, phone_seq_length = texts.size(0), texts.size(1)


        tgt_masks = get_mask_from_lengths(src_lens, max_src_len)
        # print("Tgt masks shape: ", tgt_masks.shape)
        
        # Mel Mask changes in reverse direction
        # print("Max mel len: ", max_mel_len)
        # print("Mel lens: ", mel_lens)
        mel_masks = (get_mask_from_lengths(mel_lens, max_mel_len) if mel_lens is not None else None)
        # print("Mel masks shape: ", mel_masks.shape)
        
        # This should be tgt translations not src texts
        output = self.encoder(texts, tgt_masks)  # torch.Size([Batch, seq_len, 256])

        speaker_embs = speaker_embs.unsqueeze(1).expand(-1, output.size()[1], -1)
        h_sd = output + speaker_embs  # torch.Size([Batch, tgt_seq_len, 256])
        # print("h_sd shape: ", h_sd.shape, torch.isnan(h_sd).any())

        h_si = output
        prosody_predictor = ProsodyPredictor(config=self.model_config).to(device)

        e_tgt = prosody_predictor(h_sd, h_si)  # TODO: prev_e here doesn't make sense.
        # print("e_tgt[0] (log_pi) shape: ", e_tgt[0].shape)  # torch.Size([Batch, tgt_seq_len, N_Components])
        # print("e_tgt[1] (mu) shape: ", e_tgt[1].shape)
        
        # prosody extractor
        prosody_extractor = ProsodyExtractor(config=self.model_config).to(device)
        
        mels = mels.unsqueeze(1) # mels shape is [batch_size, 1, melspec W, melspec H]
        e_src = prosody_extractor(mels)   # e is [batch_size, melspec H, melspec W, 128]
        print("e_src shape: ", e_src.shape, torch.isnan(e_src).any())
        
        # Split phone pros embeddings by phone duration
        # [batch_size (list), phoneme_sequence_length (list), melspec H (tensor), melspec W (tensor), 128 (tensor)]        
        e_k_src = prosody_extractor.split_phones(e_src, d_src)  
        print("e_k_src shape: ", len(e_k_src), len(e_k_src[0]), e_k_src[0][0].shape)  # 2 58 torch.Size([80, 12, 256])
        # print("d_src[0][0]", d_src[0][0], torch.isnan(d_src).any())

        agg_extracted_prosody = torch.zeros(batch_size, phone_seq_length, 256).to(device)
        for b in range(batch_size):
            for i in range(len(e_k_src[b])):
                agg_extracted_prosody[b,i,:] = torch.mean(e_k_src[b][i], dim=(0, 1))

        print("aggregated_prosody shape: ", agg_extracted_prosody.shape)

        # TODO: Allow for new predicted_prosodies_tgt shape
        tgt_samp = prosody_predictor.sample2(e_tgt) # torch.Size([2, 88, 256])
        print("tgt_samp shape: ", tgt_samp.shape, torch.isnan(tgt_samp).any())  
        
        if not pretraining:
            # print("alignments shape: ", alignments.shape)  # TODO: unpad alignments for realigner otherwise everything mapped to 0.
            adjusted_e_tgt = prosody_predictor.prosody_realigner(alignments, tgt_samp, e_k_src)
            # print("alignments nan", torch.isnan(alignments).any())
            # print("adjusted_e_tgt nan: ", torch.isnan(adjusted_e_tgt).any())

            # Concat
            h_sd = h_sd + adjusted_e_tgt
            # print("Output shape after prosody: ", output.shape, torch.isnan(output).any())  # torch.Size([Batch, tgt_seq_len, 256])

        # Now double check that durations and pitch etc are same as seq_length
        # print("Input mel_masks shape: ", mel_masks.shape)
        # print("Mel masks", mel_masks)

        (output, p_predictions, e_predictions, log_d_predictions, d_rounded, mel_lens, mel_masks,) = \
            self.variance_adaptor(h_sd, tgt_masks, mel_masks, max_mel_len, p_targets, e_targets, d_targets, p_control,
                                  e_control, d_control, )

        # print("d_rounded shape: ", d_rounded.shape)
        # print("mel_lens shape: ", mel_lens)
        # print("predicted mel_masks shape: ", mel_masks.shape)
        # Remap p_predictions, e_predictions, log_d_predictions to tgt size

        output, mel_masks = self.decoder(output, mel_masks)
        # print("output mel_masks shape: ", mel_masks.shape)

        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        # For calculating Lpp loss:
        # y_e_tgt = prosody_extractor.prosody_realigner(alignments, e_k_src)

        return (output, postnet_output, p_predictions, e_predictions, log_d_predictions, d_rounded, tgt_masks,
                mel_masks, src_lens, mel_lens, agg_extracted_prosody, e_tgt)


class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config

        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            with open(
                    os.path.join(
                        preprocess_config["path"]["preprocessed_path"], "speakers.json"
                    ),
                    "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(
                n_speaker,
                model_config["transformer"]["encoder_hidden"],
            )

    def forward(
            self,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels=None,
            mel_lens=None,
            max_mel_len=None,
            p_targets=None,
            e_targets=None,
            d_targets=None,
            p_control=1.0,
            e_control=1.0,
            d_control=1.0,
    ):

        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        output = self.encoder(texts, src_masks)

        # This speaker embedding is just an embedding of the speaker ID (Not allowing for zero-shot voice cloning)
        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(-1, max_src_len, -1)

        (output, p_predictions, e_predictions, log_d_predictions, d_rounded, mel_lens, mel_masks,) = \
            self.variance_adaptor(output, src_masks, mel_masks, max_mel_len, p_targets, e_targets, d_targets, p_control,
                                  e_control, d_control, )

        output, mel_masks = self.decoder(output, mel_masks)

        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )
