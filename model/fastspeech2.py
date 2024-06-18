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
                speaker_embs=None,
                alignments=None, p_targets=None, e_targets=None, d_targets=None, prev_e=None, p_control=1.0,
                e_control=1.0, d_control=1.0,):
        # batch = (ids, raw_texts, raw_translations, speakers, texts, src_lens, max_text_lens, mels, mel_lens,
        #          max_mel_lens, translations, translation_lens, max(translation_lens), speaker_embeddings, 
        #          alignments, pitches, energies, durations)

        # Get masks
        batch_size = texts.size(0)

        tgt_masks = get_mask_from_lengths(src_lens, max_src_len)
        
        mel_masks = (get_mask_from_lengths(mel_lens, max_mel_len) if mel_lens is not None else None)
        
        # This should be tgt translations not src texts
        output = self.encoder(texts, tgt_masks)  # torch.Size([Batch, seq_len, 256])

        speaker_embs = speaker_embs.unsqueeze(1).expand(-1, output.size()[1], -1)
        h_sd = output + speaker_embs  # torch.Size([Batch, seq_len, 256])
        print("h_sd shape: ", h_sd.shape)

        h_si = output
        prosody_predictor = ProsodyPredictor(256, 256, 4, 8).to(device)

        # Test this loop
        prev_prosody = torch.zeros(batch_size, 1, 256).to(device)
        prosodies = []
        for i in range(output.size()[1]):
            h_sd_t = h_sd[:, i, :]
            h_si_t = h_si[:, i, :]
            print("h_sd_t shape: ", h_sd_t.shape)
            out = prosody_predictor(h_sd_t, h_si_t, prev_prosody)
            prosodies.append(out)
            prev_prosody = out
        predicted_prosodies_tgt = torch.stack(prosodies, dim=1)
        print("predicted_prosodies_tgt shape: ", predicted_prosodies_tgt.shape)

        # e_tgt = prosody_predictor(h_sd, h_si, prev_e)  # TODO: prev_e here doesn't make sense.
        # print("e_tgt shape: ", e_tgt[0].shape)
        
        # prosody extractor
        prosody_extractor = ProsodyExtractor(1, 256, 8).to(device)
        
        mels = mels.unsqueeze(1) # mels shape is [batch_size, 1, melspec W, melspec H]
        e_src = prosody_extractor(mels)   # e is [batch_size, melspec H, melspec W, 128]
        
        # Split phone pros embeddings by phone duration
        # [batch_size (list), phoneme_sequence_length (list), melspec H (tensor), melspec W (tensor), 128 (tensor)]
        e_k_src = prosody_extractor.split_phones(e_src, d_targets)

        # TODO: Allow for new predicted_prosodies_tgt shape
        tgt_samp = prosody_predictor.sample2(predicted_prosodies_tgt)
        
        print("alignments shape: ", alignments.shape)  # TODO: unpad alignments for realigner otherwise everything mapped to 0.
        adjusted_e_tgt = prosody_predictor.prosody_realigner(alignments, tgt_samp, e_k_src)

        # Concat
        output = h_sd + adjusted_e_tgt
        print("Output shape after prosody: ", output.shape)

        # Now double check that durations and pitch etc are same as seq_length

        (output, p_predictions, e_predictions, log_d_predictions, d_rounded, mel_lens, mel_masks,) = \
            self.variance_adaptor(output, tgt_masks, mel_masks, max_mel_len, p_targets, e_targets, d_targets, p_control,
                                  e_control, d_control, )

        print("d_rounded shape: ", d_rounded.shape)
        print("mel_lens shape: ", mel_lens.shape)
        print("mel_masks shape: ", mel_masks.shape)
        # Remap p_predictions, e_predictions, log_d_predictions to tgt size


        output, mel_masks = self.decoder(output, mel_masks)

        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        # For calculating Lpp loss:
        # y_e_tgt = prosody_extractor.prosody_realigner(alignments, e_k_src)

        return (output, postnet_output, p_predictions, e_predictions, log_d_predictions, d_rounded, tgt_masks,
                mel_masks, src_lens, mel_lens, adjusted_e_tgt, )


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
