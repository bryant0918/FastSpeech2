import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# import transformer.Constants as Constants
# from .Layers import FFTBlock
# from text.symbols import symbols


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
    def __init__(self, dim_in, dim_out, n_components, hidden_dim):
        super(ProsodyExtractor, self).__init__()
        num_sigma_channels = dim_out * n_components
        num_weights_channels = n_components
        # Bi-GRU layer
        self.gru = nn.GRU(input_size=8 * 8, hidden_size=32, num_layers=1, bidirectional=True, batch_first=True)
        # Linear layer to output parameters of Gaussians
        self.normal_linear = nn.Linear(64, dim_out*n_components+num_sigma_channels)

        self.pi_network = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Linear(8, n_components), # Do Gru here?
        )
        self.normal_network = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )

    def forward(self, x, eps=1e-6):
        """"
        Returns
        -------
        log_pi: (bsz, n_components)
        mu: (bsz, n_components, dim_out)
        sigma: (bsz, n_components, dim_out)
        """
        # Apply first for weights (w_i)
        log_pi = torch.log_softmax(self.pi_network(x), dim=-1)

        # Apply first normal network
        normal_params = self.normal_network(x)

        # Reshape for GRU layer (batch_size, sequence_length, feature_size)
        normal_params = normal_params.permute(0, 2, 3, 1).contiguous()
        batch_size, height, width, channels = normal_params.size()
        normal_params = normal_params.view(batch_size, height, width * channels)
        # Apply Bi-GRU layer
        normal_params, hn = self.gru(normal_params)
        # Apply linear projection to get Gaussian parameters
        normal_params = self.normal_linear(normal_params)
        mu = normal_params[..., :self.dim_out * self.n_components]
        sigma = normal_params[..., self.dim_out * self.n_components:]

        # Add Noise (Don't know if necessary, can set eps=0)
        sigma = torch.exp(sigma + eps)

        mu = mu.reshape(-1, self.n_components, self.dim_out)
        sigma = sigma.reshape(-1, self.n_components, self.dim_out)

        return log_pi, mu, sigma


class ProsodyPredictor(nn.Module):
    def __init__(self, dim_in, dim_out, n_components, hidden_dim):
        super(ProsodyPredictor, self).__init__()
        self.num_sigma_channels = dim_out * n_components
        num_weights_channels = n_components

        self.base_network = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.LayerNorm(8),
            nn.Dropout(0.1),
            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.LayerNorm(8),
            nn.Dropout(0.1),
        )
        # GRU layer
        self.gru = nn.GRU(input_size=8 * 8, hidden_size=32, num_layers=1, bidirectional=False, batch_first=True)
        # Linear layer to output parameters for non-Linear Transform
        self.linear = nn.Linear(64, dim_out * n_components * 5)

        # Bi-GRU layer
        self.BiGru = nn.GRU(input_size=8 * 8, hidden_size=32, num_layers=1, bidirectional=True, batch_first=True)
        # Linear layer for just means and log variances
        self.normal_linear = nn.Linear(64, dim_out * n_components + self.num_sigma_channels)

        self.linear2 = nn.Linear(dim_out * n_components * 2, dim_out * n_components)
        self.linear3 = nn.Linear(dim_out * n_components * 2, dim_out * n_components)

    def forward(self, h_sd, h_si, prev_e, eps=1e-6):
        # First predict the Speaker Independent means and log-variances
        h_si, _ = self.BiGru(h_si)   # Should we concat h_si and h_n?
        h_si = self.normal_linear(h_si)

        # Separate mus and log-variances
        mu = h_si[..., :self.dim_out * self.n_components]
        v = h_si[..., self.dim_out * self.n_components:]

        # Run Speaker Dependent features through the base network
        x = self.base_network(h_sd)
        out = torch.concatenate(x, prev_e)

        out = self.gru(out) # Should this output be out, hn?

        out = self.linear(out)

        alphas = out[..., :self.dim_out * self.n_components]
        a = out[..., self.dim_out * self.n_components:self.dim_out * self.n_components * 2]
        b = out[..., self.dim_out * self.n_components * 2:self.dim_out * self.n_components * 3]
        c = out[..., self.dim_out * self.n_components * 3:self.dim_out * self.n_components * 4]
        d = out[..., self.dim_out * self.n_components * 4:]

        A = torch.diag(a)
        C = torch.diag(c)

        # Perform non-Linear Transformation
        mu = self.linear2(torch.tanh(A @ mu + b))
        v = self.linear3(torch.tanh(C @ v + d))

        mu = mu.reshape(-1, self.n_components, self.dim_out)
        sigma = sigma.reshape(-1, self.n_components, self.dim_out)

        # Add Noise (Don't know if necessary, can set eps=0)
        sigma = torch.exp(sigma + eps)

        log_pi = torch.log_softmax(alphas, dim=-1)

        return log_pi, mu, sigma

    def phone_loss(self, x, y):
        log_pi, mu, sigma = self.forward(x)
        z_score = (y.unsqueeze(1) - mu) / sigma
        normal_loglik = (
                -0.5 * torch.einsum("bij,bij->bi", z_score, z_score)
                - torch.sum(torch.log(sigma), dim=-1)
        )
        loglik = torch.logsumexp(log_pi + normal_loglik, dim=-1)
        return -loglik    # Sum over all phones for total loss (L_pp)


class BaseProsodyPredictor(nn.Module):
    def __init__(self,dim_in, dim_out, n_components, hidden_dim):
        super(BaseProsodyPredictor, self).__init__()
        self.num_sigma_channels = dim_out * n_components
        num_weights_channels = n_components
        # First Conv2d layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        # Second Conv2d layer
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        # Bi-GRU layer
        self.gru = nn.GRU(input_size=8 * 8, hidden_size=32, num_layers=1, bidirectional=False, batch_first=True)
        # Linear layer to output speaker independent means and log-variances
        self.normal_linear = nn.Linear(64, dim_out*n_components+self.num_sigma_channels + num_weights_channels)

        self.base_network = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.LayerNorm(8),
            nn.Dropout(0.1),
            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.LayerNorm(8),
            nn.Dropout(0.1),
        )

    def forward(self, x, prev_e, eps=1e-6):
        x = self.base_network(x)

        out = torch.concatenate(x, prev_e)

        out = self.gru(out) # Should this output be out, hn?

        out = self.Linear(out)

        mu = out[..., :self.dim_out * self.n_components]
        sigma = out[..., self.dim_out * self.n_components:self.dim_out*self.n_components+self.num_sigma_channels]
        pi = out[..., self.dim_out*self.n_components+self.num_sigma_channels:]

        mu = mu.reshape(-1, self.n_components, self.dim_out)
        sigma = sigma.reshape(-1, self.n_components, self.dim_out)

        # Add Noise (Don't know if necessary, can set eps=0)
        sigma = torch.exp(sigma + eps)

        log_pi = torch.log_softmax(pi, dim=-1)

        return log_pi, mu, sigma


        # # Apply first Conv2d, BatchNorm, and ReLU
        # x = F.relu(self.bn1(self.conv1(x)))
        # # Apply second Conv2d, BatchNorm, and ReLU
        # x = F.relu(self.bn2(self.conv2(x)))
        # # Reshape for GRU layer (batch_size, sequence_length, feature_size)
        # x = x.permute(0, 2, 3, 1).contiguous()
        # batch_size, height, width, channels = x.size()
        # x = x.view(batch_size, height, width * channels)
        # # Apply Bi-GRU layer
        # x, _ = self.gru(x)
        # # Apply linear projection to get Gaussian parameters
        # x = self.linear(x)

        # return out

    def phone_loss(self, x, y):
        log_pi, mu, sigma = self.forward(x)
        z_score = (y.unsqueeze(1) - mu) / sigma
        normal_loglik = (
                -0.5 * torch.einsum("bij,bij->bi", z_score, z_score)
                - torch.sum(torch.log(sigma), dim=-1)
        )
        loglik = torch.logsumexp(log_pi + normal_loglik, dim=-1)
        return -loglik    # Sum over all phones for total loss (L_pp)


if __name__ == "__main__":

    # Create the model
    model = ProsodyExtractor(1,1,4,8)
    # Print the model summary
    print(model)
    # Create the model
    model = BaseProsodyPredictor(1,1,4,8)
    # Print the model summary
    print(model)
    # Create the model
    model = ProsodyPredictor(1, 1, 4, 8)
    # Print the model summary
    print(model)
