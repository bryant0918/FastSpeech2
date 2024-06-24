import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import transformer.Constants as Constants
from .Layers import FFTBlock
from text.symbols import symbols

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


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
            enc_output, enc_slf_attn = enc_layer(enc_output, mask=mask, slf_attn_mask=slf_attn_mask)
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
    def __init__(self, config):   # TODO: intialize with config
        super(ProsodyExtractor, self).__init__()

        if config:
            dim_in = config["prosody_extractor"]["dim_in"]
            dim_out = config["prosody_extractor"]["dim_out"]
            hidden_dim = config["prosody_extractor"]["hidden_dim"]

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
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=dim_out // 2, num_layers=1, bidirectional=True,
                          batch_first=True)

    def forward(self, x):
        """"
        Input:
            x: mels
        Returns
        -------
        prosody_embedding e
        """
        # Apply network
        x = self.cnn(x)

        x = x.permute(0, 2, 1)  # Permute to [batch_size, height*width, 8] (N,L,H_in)

        # Apply Bi-GRU layer
        x, _ = self.gru(x)  # [batch_size, melspec H * melspec W, 128]

        # x.view(x.size()[0], seq_length, num_directions, hidden_size).

        # TODO: Don't hardcode 80 here, use n_mel_channels from preprocess_config
        return x.view(x.size()[0], 80, -1, x.size()[-1])  # [batch_size, melspec H, melspec W, 128]

    def split_phones(self, x, durations, device='cuda'):
        """
        Split the phone embeddings by phone
        Input:
            x: e, Torch.tensor - [batch_size, melspec H, melspec W, 128]
            durations: List - [batch_size, duration_sequence_length]

        Output:
            List of phone embedding chunks - [batch_size, phoneme_sequence_length, melspec H, melspec W, 128]
        """

        zeros = torch.zeros((durations.shape[0], 1), dtype=torch.int).to(device)
        # print("zeros.size:", zeros.size())
        # print("durations.size:", durations.size())
        # print(durations)

        concated = torch.cat([zeros, durations], dim=1)
        cumulative_durations = torch.cumsum(concated, dim=1)
        # print("Cumulative durations", cumulative_durations)

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
    def __init__(self, config):   # TODO: intialize with config
        super(ProsodyPredictor, self).__init__()

        sd_dim_in = config["prosody_predictor"]["sd_dim_in"]
        si_dim_in = config["prosody_predictor"]["si_dim_in"]
        dim_out = config["prosody_predictor"]["dim_out"]
        conv_hidden_dim = config["prosody_predictor"]["conv_hidden_dim"]            
        n_components = config["prosody_predictor"]["n_components"]
        gru_hidden_dim = config["prosody_predictor"]["gru_hidden_dim"]
        bigru_hidden_dim = config["prosody_predictor"]["bigru_hidden_dim"]
        
        self.num_sigma_channels = dim_out * n_components

        self.sd_dim_in = sd_dim_in
        self.dim_out = dim_out
        self.n_components = n_components

        # GRU layer (SD)
        self.gru = nn.GRU(input_size=conv_hidden_dim, hidden_size=gru_hidden_dim, num_layers=1, bidirectional=False, batch_first=True)
        # Linear layer to output nonlinear transformation parameters
        self.normal_linear = nn.Linear(gru_hidden_dim, self.dim_out * self.n_components * 4 + self.n_components)

        # CNN Block (SD)
        self.conv1 = nn.Conv1d(in_channels=sd_dim_in, out_channels=conv_hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.layernorm = nn.LayerNorm(conv_hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.conv2 = nn.Conv1d(in_channels=conv_hidden_dim, out_channels=conv_hidden_dim, kernel_size=3, padding=1)

        # Bi-GRU layer (SI)
        self.BiGru = nn.GRU(input_size=si_dim_in, hidden_size=bigru_hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        # Linear layer for just means and log variances
        self.normal_linear2 = nn.Linear(bigru_hidden_dim*2, dim_out * n_components + self.num_sigma_channels)

        self.linear2 = nn.Linear(dim_out * n_components, dim_out * n_components)
        self.linear3 = nn.Linear(dim_out * n_components, dim_out * n_components)

    def forward(self, h_sd, h_si, eps=1e-6):
        # First predict the Speaker Independent means and log-variances
        h_si, _ = self.BiGru(h_si)  # Should we concat h_si and h_n?
        h_si = self.normal_linear2(h_si)

        # Separate mus and log-variances
        mu = h_si[..., :self.dim_out * self.n_components]
        v = h_si[..., self.dim_out * self.n_components:]

        # Run Speaker Dependent features through the base network
        # Permute the input tensor batch, sentence_length, embedding_dim = 20, 19, 10
        
        h_sd = h_sd.permute(0, 2, 1)  # Changes shape to (Batch_size, 256, Sequence_length) for conv layer
        h_sd = self.relu(self.conv1(h_sd))
        h_sd = h_sd.permute(0, 2, 1)  # Changes shape back to (Batch_size, sequence_length, channels) for layernorm
        h_sd = self.dropout(self.layernorm(h_sd))
        h_sd = h_sd.permute(0, 2, 1)  # Changes shape back to (Batch_size, channels, sequence_length) for conv layer
        h_sd = self.relu(self.conv2(h_sd))
        h_sd = h_sd.permute(0, 2, 1)  # Changes shape back to (Batch_size, sequence_length, channels) for layernorm
        h_sd = self.dropout(self.layernorm(h_sd))

        h_sd, _ = self.gru(h_sd)

        h_sd = self.normal_linear(h_sd)

        # Separate Transformation Parameters
        alphas = h_sd[..., :self.n_components]
        a = h_sd[..., self.n_components:self.dim_out * self.n_components + self.n_components]
        b = h_sd[..., self.dim_out * self.n_components + self.n_components:
                      2 * self.dim_out * self.n_components + self.n_components]
        c = h_sd[..., 2 * self.dim_out * self.n_components + self.n_components:
                      3 * self.dim_out * self.n_components + self.n_components]
        d = h_sd[..., 3 * self.dim_out * self.n_components + self.n_components:]

        # Perform non-Linear Transformation
        mu = self.linear2(torch.tanh(torch.multiply(a, mu) + b))
        v = self.linear3(torch.tanh(torch.multiply(c, v) + d))

        # Add Noise (Don't know if necessary, can set eps=0)
        sigma = torch.exp(v + eps)

        log_pi = torch.log_softmax(alphas, dim=-1) # [batch_size, text_sequence_length, n_components]

        # Separate
        mu = mu.reshape(mu.size()[0], -1, self.n_components, self.dim_out)
        sigma = sigma.reshape(sigma.size()[0], -1, self.n_components, self.dim_out)

        return log_pi, mu, sigma  # mu,sigma is [batch_size, text_sequence_length, n_components, dim_out]

    def phone_loss(self, x, y):
        """
        Calculate the negative log-likelihood of the phone sequence given the prosody features
        Input:
            x: (h_sd, h_si, prev_e)
            y: prosody embeddings e_k from prosody extractor
        Output:
            -loglik: Negative log-likelihood of the phone sequence given the prosody features
        """
        log_pi, mu, sigma = self.forward(x)
        z_score = (y.unsqueeze(1) - mu) / sigma
        normal_loglik = (
                -0.5 * torch.einsum("bij,bij->bi", z_score, z_score)
                - torch.sum(torch.log(sigma), dim=-1)
        )
        loglik = torch.logsumexp(log_pi + normal_loglik, dim=-1)
        return -loglik  # Sum over all phones for total loss (L_pp)

    def sample(self, h_sd, h_si, prev_e=None):
        log_pi, mu, sigma = self.forward(h_sd, h_si, prev_e)
        cum_pi = torch.cumsum(torch.exp(log_pi), dim=-1)
        rvs = torch.rand(len(h_sd), 1).to(h_sd) # Not working

        rand_pi = torch.searchsorted(cum_pi, rvs)
        rand_normal = torch.randn_like(mu) * sigma + mu
        samples = torch.take_along_dim(rand_normal, indices=rand_pi.unsqueeze(-1), dim=1).squeeze(dim=1)
        return samples

    def sample2(self, e):
        """
        Take a sample from e (log_pi, mu, sigma)
        Input:
            e: (log_pi, mu, sigma)
                log_pi: [batch_size, text_sequence_length, n_components]
                mu: [batch_size, text_sequence_length, n_components, dim_out]
                sigma: [batch_size, text_sequence_length, n_components, dim_out]
        Output:
            sample: [batch_size, text_sequence_length, dim_out]
        """

        log_pi, mu, sigma = e

        # Convert log_pi to probabilities
        pi = torch.exp(log_pi)

        # # Sample a component index for each item in the batch and sequence
        batch_size, seq_len, n_components, dim_out = mu.size()

        components = torch.multinomial(pi.view(-1, n_components), 1)
        components = components.view(batch_size, seq_len)

        # # Gather the corresponding means and variances
        selected_mu = torch.gather(mu, 2, components.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, dim_out)).squeeze(2)
        selected_sigma = torch.gather(sigma, 2, components.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, dim_out)).squeeze(2)

        # # Sample from the selected Gaussian components
        normal_samples = torch.randn_like(selected_mu)
        sample = selected_mu + selected_sigma * normal_samples

        return sample

    def prosody_realigner(self, phone_alignments, tgt_samp, e_k_src):
        """
        Realigns and shifts tgt_samp distribution by e_k_src
        """
        # print("phone_alignments", phone_alignments.shape)
        beta = 0.1
        batch_size = len(phone_alignments)
        seq_length = tgt_samp.shape[1]
        # print("tgt samp shape", tgt_samp.shape)
        sum_e = torch.zeros(batch_size, seq_length, tgt_samp.shape[2], device=device)
        new_e = torch.zeros(batch_size, seq_length, tgt_samp.shape[2], device=device)
        counts = torch.zeros(batch_size, seq_length, device=device)  # Tensor to keep count of how many times each index is updated

        # print("Sum_e", sum_e.shape)
        # print("e_k_src", type(e_k_src), len(e_k_src), len(e_k_src[0]), len(e_k_src[1]))

        # Works for when target sentence is longer

        for b in range(batch_size):
            for j in range(len(phone_alignments[b])):
                if j >= tgt_samp[b].shape[0]:
                        print("CONTINUING IN PROSODY REALIGNER")
                        continue
                
                # Skip padded entries in second dimension
                if j != 0 and torch.all(phone_alignments[b][j] == 0):
                    continue

                for idx, i in enumerate(phone_alignments[b][j]):

                    # Skip padded entries in third dimension
                    if idx != 0 and i == 0:
                        continue

                    # Reshape B to be broadcastable to A's shape
                    try:
                        B_broadcasted = tgt_samp[b][j].unsqueeze(0).unsqueeze(0)
                    except:
                        print("tgt_samp[b].shape", tgt_samp[b].shape, b,j, len(phone_alignments[b]))
                        print(phone_alignments[b][j])
                        raise ValueError("Error")
                    # Compute the weighted combination
                    if e_k_src[b][i].numel() == 0:
                        result = B_broadcasted
                    else:
                        result = (1 - beta) * B_broadcasted + beta * e_k_src[b][i]
                    new_mean = result.mean(dim=(0, 1))  # [256]

                    # Accumulate the new_mean for each index
                    # sum_results[b][:, i] += new_mean
                    try:
                        sum_e[b, j] += new_mean
                    except:
                        print("\nsum_e[b].shape", sum_e[b].shape, b,j,i, len(phone_alignments[b]))
                        print(len(phone_alignments[b][j]), phone_alignments[b][j])
                        raise IndexError("Error")
                    counts[b, j] += 1

                    if torch.isnan(new_mean).any():
                        print('NAN', result)
                        print(B_broadcasted) 
                        print(e_k_src[b][i])
                        raise ValueError("Nan in new_mean")
            
        mask = counts > 0
        new_e[mask] = sum_e[mask] / counts[mask].unsqueeze(-1)
            
        return new_e


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
        self.normal_linear = nn.Linear(512, dim_out * n_components + self.num_sigma_channels + num_weights_channels)

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
        sigma = x[..., self.dim_out * self.n_components:self.dim_out * self.n_components + self.num_sigma_channels]
        pi = x[..., self.dim_out * self.n_components + self.num_sigma_channels:]

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
        return -loglik  # Sum over all phones for total loss (L_pp)

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
