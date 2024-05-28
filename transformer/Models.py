import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import transformer.Constants as Constants
from .Layers import FFTBlock
from text.symbols import symbols


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)


class Encoder(nn.Module):
    """ Encoder """

    def __init__(self, config):
        super(Encoder, self).__init__()

        n_position = config["max_seq_len"] + 1
        n_src_vocab = len(symbols) + 1
        d_word_vec = config["transformer"]["encoder_hidden"]
        n_layers = config["transformer"]["encoder_layer"]
        n_head = config["transformer"]["encoder_head"]
        d_k = d_v = (
            config["transformer"]["encoder_hidden"]
            // config["transformer"]["encoder_head"]
        )
        d_model = config["transformer"]["encoder_hidden"]
        d_inner = config["transformer"]["conv_filter_size"]
        kernel_size = config["transformer"]["conv_kernel_size"]
        dropout = config["transformer"]["encoder_dropout"]

        self.max_seq_len = config["max_seq_len"]
        self.d_model = d_model

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=Constants.PAD
        )
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, src_seq, mask, return_attns=False):


        enc_slf_attn_list = []
        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]

        # -- Prepare masks
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        # -- Forward
        if not self.training and src_seq.shape[1] > self.max_seq_len:
            enc_output = self.src_word_emb(src_seq) + get_sinusoid_encoding_table(
                src_seq.shape[1], self.d_model
            )[: src_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                src_seq.device
            )

        else:
            enc_output = self.src_word_emb(src_seq) + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)

        for enc_layer in self.layer_stack:

            enc_output, enc_slf_attn = enc_layer(
                enc_output, mask=mask, slf_attn_mask=slf_attn_mask
            )

            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        return enc_output


class Decoder(nn.Module):
    """ Decoder """

    def __init__(self, config):
        super(Decoder, self).__init__()

        n_position = config["max_seq_len"] + 1
        d_word_vec = config["transformer"]["decoder_hidden"]
        n_layers = config["transformer"]["decoder_layer"]
        n_head = config["transformer"]["decoder_head"]
        d_k = d_v = (
            config["transformer"]["decoder_hidden"]
            // config["transformer"]["decoder_head"]
        )
        d_model = config["transformer"]["decoder_hidden"]
        d_inner = config["transformer"]["conv_filter_size"]
        kernel_size = config["transformer"]["conv_kernel_size"]
        dropout = config["transformer"]["decoder_dropout"]

        self.max_seq_len = config["max_seq_len"]
        self.d_model = d_model

        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, enc_seq, mask, return_attns=False):

        dec_slf_attn_list = []
        batch_size, max_len = enc_seq.shape[0], enc_seq.shape[1]

        # -- Forward
        if not self.training and enc_seq.shape[1] > self.max_seq_len:
            # -- Prepare masks
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = enc_seq + get_sinusoid_encoding_table(
                enc_seq.shape[1], self.d_model
            )[: enc_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                enc_seq.device
            )
        else:
            max_len = min(max_len, self.max_seq_len)

            # -- Prepare masks
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = enc_seq[:, :max_len, :] + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)
            mask = mask[:, :max_len]
            slf_attn_mask = slf_attn_mask[:, :, :max_len]

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output, mask=mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]

        return dec_output, mask


class ProsodyExtractor(nn.Module):
    def __init__(self, dim_in=1, dim_out=128, hidden_dim=8):
        super(ProsodyExtractor, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=dim_in, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Flatten(start_dim=2),
        )
        # Bi-GRU layer
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=dim_out//2, num_layers=1, bidirectional=True, batch_first=True)

    def forward(self, x):
        """"
        Returns
        -------
        prosody_embedding e
        """
        # Apply network
        x = self.cnn(x)

        x = x.permute(0, 2, 1)  # Permute to [batch_size, height*width, 8] (N,L,H_in)
        print("x.size:", x.size())

        # Apply Bi-GRU layer
        x, _ = self.gru(x)

        # TODO: Don't hardcode 80 here, use n_mel_channels from preprocess_config
        return x.view(x.size()[0], 80, -1, x.size()[-1])   # [batch_size, melspec H, melspec W, 128]

    def split_phones(self, x, durations):
        """
        Split the phone embeddings by phone
        Input:
            x: Torch.tensor - [batch_size, melspec H, melspec W, 128]
            durations: List - [batch_size, duration_sequence_length]

        Output:
            List of phone embeddings - [batch_size,
        """

        # Figure out how to split a batch
        # phone_emb_chunks = []
        # start_frame = 0
        # for i in range(len(duration)):
        #     phone_emb_chunks.append(e[:, :, start_frame:start_frame + duration[i]])
        #     start_frame += duration[i]
        #
        # return phone_emb_chunks

        zeros = torch.zeros((durations.shape[0], 1), dtype=torch.int)

        concated = torch.cat([zeros, durations], dim=1)
        cumulative_durations = torch.cumsum(concated, dim=1)

        start_frames = cumulative_durations[:, :-1]
        end_frames = cumulative_durations[:, 1:]

        batch_phone_emb_chunks = []
        # Iterate over each sample in the batch
        for b in range(x.shape[0]):
            sample_phone_emb_chunks = []
            # Iterate over each phoneme
            for i in range(start_frames.shape[1]):
                start = start_frames[b, i].item()
                end = end_frames[b, i].item()
                # Extract the chunk using slicing
                chunk = x[b, :, start:end, :]
                sample_phone_emb_chunks.append(chunk)
            batch_phone_emb_chunks.append(sample_phone_emb_chunks)

        return batch_phone_emb_chunks


class ProsodyPredictor(nn.Module):
    def __init__(self, dim_in, dim_out, n_components, hidden_dim):
        super(ProsodyPredictor, self).__init__()
        self.num_sigma_channels = dim_out * n_components

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.n_components = n_components

        # GRU layer (SD)
        self.gru = nn.GRU(input_size=8, hidden_size=512, num_layers=1, bidirectional=False, batch_first=True)
        # Linear layer to output nonlinear transformation parameters
        self.normal_linear = nn.Linear(512, self.dim_out * self.n_components * 5)

        self.conv1 = nn.Conv1d(in_channels=dim_in, out_channels=8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.layernorm = nn.LayerNorm(8)
        self.dropout = nn.Dropout(0.1)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, padding=1)

        # Bi-GRU layer (SI)
        self.BiGru = nn.GRU(input_size=256, hidden_size=32, num_layers=1, bidirectional=True, batch_first=True)
        # Linear layer for just means and log variances
        self.normal_linear2 = nn.Linear(64, dim_out * n_components + self.num_sigma_channels)

        self.linear2 = nn.Linear(dim_out * n_components, dim_out * n_components)
        self.linear3 = nn.Linear(dim_out * n_components, dim_out * n_components)

    def forward(self, h_sd, h_si, prev_e=None, eps=1e-6):
        # First predict the Speaker Independent means and log-variances
        h_si, _ = self.BiGru(h_si)   # Should we concat h_si and h_n?
        print("h_si.size:", h_si.size())
        h_si = self.normal_linear2(h_si)

        # Separate mus and log-variances
        mu = h_si[..., :self.dim_out * self.n_components]
        v = h_si[..., self.dim_out * self.n_components:]

        # Run Speaker Dependent features through the base network
        print("h_sd.size:", h_sd.size())
        # Permute the input tensor batch, sentence_length, embedding_dim = 20, 19, 10
        h_sd = h_sd.permute(0, 2, 1)  # Changes shape to (Batch_size, 256, Sequence_length) for conv layer
        h_sd = self.relu(self.conv1(h_sd))
        h_sd = h_sd.permute(0, 2, 1)  # Changes shape back to (Batch_size, sequence_length, channels) for layernorm
        h_sd = self.dropout(self.layernorm(h_sd))
        h_sd = h_sd.permute(0, 2, 1)  # Changes shape back to (Batch_size, channels, sequence_length) for conv layer
        h_sd = self.relu(self.conv2(h_sd))
        h_sd = h_sd.permute(0, 2, 1)  # Changes shape back to (Batch_size, sequence_length, channels) for layernorm
        h_sd = self.dropout(self.layernorm(h_sd))

        if prev_e:
            h_sd, _ = self.gru(h_sd, prev_e)
        else:
            h_sd, _ = self.gru(h_sd)
        h_sd = self.normal_linear(h_sd)

        # Separate Transformation Parameters
        alphas = h_sd[..., :self.dim_out * self.n_components]
        a = h_sd[..., self.dim_out * self.n_components:self.dim_out * self.n_components * 2]
        b = h_sd[..., self.dim_out * self.n_components * 2:self.dim_out * self.n_components * 3]
        c = h_sd[..., self.dim_out * self.n_components * 3:self.dim_out * self.n_components * 4]
        d = h_sd[..., self.dim_out * self.n_components * 4:]

        # Perform non-Linear Transformation
        mu = self.linear2(torch.tanh(torch.multiply(a, mu) + b))
        v = self.linear3(torch.tanh(torch.multiply(c, v) + d))

        # This puts batch_size in the last dimension
        # mu = mu.reshape(-1, self.n_components, self.dim_out)
        # sigma = v.reshape(-1, self.n_components, self.dim_out)

        # Add Noise (Don't know if necessary, can set eps=0)
        sigma = torch.exp(v + eps)

        log_pi = torch.log_softmax(alphas, dim=-1)

        return log_pi, mu, sigma   # each is [batch_size, text_sequence_length, n_components]

    def phone_loss(self, x, y):
        log_pi, mu, sigma = self.forward(x)
        z_score = (y.unsqueeze(1) - mu) / sigma
        normal_loglik = (
                -0.5 * torch.einsum("bij,bij->bi", z_score, z_score)
                - torch.sum(torch.log(sigma), dim=-1)
        )
        loglik = torch.logsumexp(log_pi + normal_loglik, dim=-1)
        return -loglik    # Sum over all phones for total loss (L_pp)

    def sample(self, x):
        log_pi, mu, sigma = self.forward(x)
        cum_pi = torch.cumsum(torch.exp(log_pi), dim=-1)
        rvs = torch.rand(len(x), 1).to(x)
        rand_pi = torch.searchsorted(cum_pi, rvs)
        rand_normal = torch.randn_like(mu) * sigma + mu
        samples = torch.take_along_dim(rand_normal, indices=rand_pi.unsqueeze(-1), dim=1).squeeze(dim=1)
        return samples


class BaseProsodyPredictor(nn.Module):
    def __init__(self, dim_in, dim_out, n_components, hidden_dim):
        super(BaseProsodyPredictor, self).__init__()
        self.num_sigma_channels = dim_out * n_components
        num_weights_channels = n_components

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.n_components = n_components

        self.gru = nn.GRU(input_size=8, hidden_size=512, num_layers=1, bidirectional=False, batch_first=True)
        # Linear layer to output speaker independent means and log-variances
        self.normal_linear = nn.Linear(512, dim_out*n_components+self.num_sigma_channels + num_weights_channels)

        self.conv1 = nn.Conv1d(in_channels=dim_in, out_channels=8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.layernorm = nn.LayerNorm(8)
        self.dropout = nn.Dropout(0.1)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, padding=1)

    def forward(self, x, prev_e=None, eps=1e-6):
        print("x.size:", x.size())
        # Permute the input tensor batch, sentence_length, embedding_dim = 20, 19, 10
        x = x.permute(0, 2, 1)  # Changes shape to (Batch_size, 256, Sequence_length) for conv layer
        x = self.relu(self.conv1(x))
        x = x.permute(0, 2, 1)  # Changes shape back to (Batch_size, sequence_length, channels) for layernorm
        x = self.dropout(self.layernorm(x))
        x = x.permute(0, 2, 1)  # Changes shape back to (Batch_size, channels, sequence_length) for conv layer
        x = self.relu(self.conv2(x))
        x = x.permute(0, 2, 1)  # Changes shape back to (Batch_size, sequence_length, channels) for layernorm
        x = self.dropout(self.layernorm(x))

        if prev_e:
            x, _ = self.gru(x, prev_e)
        else:
            x, _ = self.gru(x)

        x = self.normal_linear(x)

        mu = x[..., :self.dim_out * self.n_components]
        sigma = x[..., self.dim_out * self.n_components:self.dim_out*self.n_components+self.num_sigma_channels]
        pi = x[..., self.dim_out*self.n_components+self.num_sigma_channels:]

        # This puts batch_size in the last dimension
        # mu = mu.reshape(-1, self.n_components, self.dim_out)
        # sigma = sigma.reshape(-1, self.n_components, self.dim_out)

        # Add Noise (Don't know if necessary, can set eps=0)
        sigma = torch.exp(sigma + eps)

        log_pi = torch.log_softmax(pi, dim=-1)

        return log_pi, mu, sigma

    def phone_loss(self, e, y):
        log_pi, mu, sigma = e
        z_score = (y.unsqueeze(1) - mu) / sigma
        normal_loglik = (
                -0.5 * torch.einsum("bij,bij->bi", z_score, z_score)
                - torch.sum(torch.log(sigma), dim=-1)
        )
        loglik = torch.logsumexp(log_pi + normal_loglik, dim=-1)
        return -loglik    # Sum over all phones for total loss (L_pp)

    def sample(self, x):
        log_pi, mu, sigma = self.forward(x)
        cum_pi = torch.cumsum(torch.exp(log_pi), dim=-1)
        rvs = torch.rand(len(x), 1).to(x)
        rand_pi = torch.searchsorted(cum_pi, rvs)
        rand_normal = torch.randn_like(mu) * sigma + mu
        samples = torch.take_along_dim(rand_normal, indices=rand_pi.unsqueeze(-1), dim=1).squeeze(dim=1)
        return samples


if __name__ == "__main__":
    model = ProsodyExtractor()
    print(model)

    # Example batch data
    batch_size = 3
    num_mel_bins = 80
    max_frames = 158

    # Generate random batch of mel spectrograms (replace with actual data)
    batch_mel_spectrogram_embs = torch.rand(batch_size, num_mel_bins, max_frames, 128)

    # Example durations, starts, and ends for each spectrogram in the batch
    durations = torch.tensor([
        [7, 5, 4, 12, 2, 4, 6, 4, 7, 8, 8, 6, 5, 7, 5, 6, 9, 4, 12, 13, 6, 5, 13],
        [6, 6, 5, 10, 3, 5, 5, 6, 6, 9, 7, 7, 4, 8, 6, 5, 8, 5, 11, 12, 5, 6, 12],
        [8, 4, 6, 11, 4, 3, 7, 5, 8, 7, 9, 5, 6, 6, 7, 4, 9, 6, 10, 14, 7, 4, 11]])

    print(durations)

    split = model.split_phones(batch_mel_spectrogram_embs, durations)

    print("Split phone embeddings: ", type(split), len(split), len(split[0]), len(split[1]), len(split[2]),
          type(split[0]), type(split[0][0]), split[0][0].size(), split[0][1].size(), type(split[0][0][0]))

    # split0 = torch.tensor(split[0])

    # Split is [batch_size, phoneme_sequence_length, melspec H, melspec W, 128]

    # print("Split phone embeddings: ", split0.size())

    pass