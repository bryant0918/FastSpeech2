import os
import json

import torch
import torch.nn as nn

from transformer import ProsodyExtractor, ProsodyPredictor
from utils.tools import get_mask_from_lengths
from text import _vocab_size

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


class ProsLearner(nn.Module):
    """ Prosody Learner """

    def __init__(self, preprocess_config, model_config):
        super(ProsLearner, self).__init__()

        self.model_config = model_config
        self.preprocess_config = preprocess_config

        self.prosody_extractor = ProsodyExtractor(config=model_config)
        self.prosody_predictor = ProsodyPredictor(config=model_config)

        # Only need this if I'm getting speaker embeddings path from json
        with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json"), "r") as f:
            self.speakers_json = json.load(f)

        self.text_embedding = nn.Embedding(_vocab_size, model_config["text_embedding_dim"])


    def forward(self, texts, src_lens, max_src_len, mels=None, mel_lens=None, max_mel_len=None,
                speaker_embs=None, alignments=None, p_targets=None, e_targets=None, d_targets=None):

        tgt_masks = get_mask_from_lengths(src_lens, max_src_len)
        # print("Tgt masks shape: ", tgt_masks.shape)
        
        # This should be tgt translations not src texts
        # output = self.encoder(texts, tgt_masks)  # torch.Size([Batch, seq_len, 256])
        print("texts shape: ", texts.shape)
        print("tgt_masks shape: ", tgt_masks.shape)
        print("speaker_embs shape: ", speaker_embs.shape)

        text_embs = self.text_embedding(texts)

        speaker_embs = speaker_embs.unsqueeze(1).expand(-1, text_embs.size()[1], -1)

        # TODO: Somehow combine speaekr_embs with texts to get h_sd
        # Concat speaker embs to each phoneme embedding
        h_sd = torch.cat((text_embs, speaker_embs), dim=-1)  # Shape: [Batch, seq_len, text_embed_dim + speaker_embed_dim]

        print("h_sd shape: ", h_sd.shape, torch.isnan(h_sd).any())  # torch.Size([Batch, tgt_seq_len, 512])

        e_tgt = self.prosody_predictor(h_sd, text_embs)
        print("e_tgt shape: ", e_tgt[0].shape, e_tgt[1].shape)
                
        mels = mels.unsqueeze(1) # mels shape is [batch_size, 1, melspec W, melspec H]
        e_src = self.prosody_extractor(mels)   # e is [batch_size, melspec H, melspec W, 256]
        print("e_src shape: ", e_src.shape, torch.isnan(e_src).any())
        
        
        # Split phone pros embeddings by phone duration
        # [batch_size (list), phoneme_sequence_length (list), melspec H (tensor), melspec W (tensor), 128 (tensor)]        
        e_k_src = self.prosody_extractor.split_phones(e_src, d_targets)
        # print("e_k_src shape: ", len(e_k_src), len(e_k_src[0]), e_k_src[0][0].shape, torch.isnan(e_k_src[0][0]).any())
        # print("d_src[0][0]", d_src[0][0], torch.isnan(d_src).any())

        # TODO: Allow for new predicted_prosodies_tgt shape
        tgt_samp = self.prosody_predictor.sample2(e_tgt)
        # print("tgt_samp shape: ", tgt_samp.shape, torch.isnan(tgt_samp).any())  
        
        # TODO: Return predicted and extracted prosodies
        
        return None


