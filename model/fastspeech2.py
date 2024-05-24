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

    def forward(self, speakers, texts, src_lens, max_src_len, speaker_embs, mels=None, mel_lens=None, max_mel_len=None,
                durations=None, alignments=None, p_targets=None, e_targets=None, d_targets=None, p_control=1.0,
                e_control=1.0, d_control=1.0,):

        # Get masks
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (get_mask_from_lengths(mel_lens, max_mel_len) if mel_lens is not None else None)

        output = self.encoder(texts, src_masks)
        print("output of encoder shape: ", output.shape)

        output = output + speaker_embs

        # TODO: prosody
        # prosody extractor
        prosody_extractor = ProsodyExtractor(1, 128, 8).to(device)
        e = prosody_extractor(mels)   # e is [batch_size, melspec H, melspec W, 128]
        
        # Split phone embeddings by phone
        # [batch_size (list), phoneme_sequence_length (list), melspec H (tensor), melspec W (tensor), 128 (tensor)]
        split_phones = prosody_extractor.split_phones(e, durations)

        # TODO: Get alignments

        # Switch phone embeddings order to match target language through alignment model


        # Get predicted prosody of target language without reference
        # prosody_embedding = prosody_predictor(self.speaker_emb)

        # Maximize posterior of Predicted prosody of target language with reference

        # Concat and pass to Variance Adaptor

        (output, p_predictions, e_predictions, log_d_predictions, d_rounded, mel_lens, mel_masks,) = \
            self.variance_adaptor(output, src_masks, mel_masks, max_mel_len, p_targets, e_targets, d_targets, p_control,
                                  e_control, d_control, )

        output, mel_masks = self.decoder(output, mel_masks)

        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (output, postnet_output, p_predictions, e_predictions, log_d_predictions, d_rounded, src_masks,
                mel_masks, src_lens, mel_lens,)


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

        # TODO: prosody
        # prosody_embedding = prosody_predictor(self.speaker_emb)

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
