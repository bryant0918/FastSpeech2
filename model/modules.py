import os
import json
import copy
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from utils.tools import get_mask_from_lengths, pad


class VarianceAdaptor(nn.Module):
    """Variance Adaptor"""

    def __init__(self, preprocess_config, model_config):
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = VariancePredictor(model_config)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(model_config)
        self.energy_predictor = VariancePredictor(model_config)

        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        assert self.pitch_feature_level in ["phoneme_level", "frame_level"]
        assert self.energy_feature_level in ["phoneme_level", "frame_level"]

        pitch_quantization = model_config["variance_embedding"]["pitch_quantization"]
        energy_quantization = model_config["variance_embedding"]["energy_quantization"]
        n_bins = model_config["variance_embedding"]["n_bins"]
        assert pitch_quantization in ["linear", "log"]
        assert energy_quantization in ["linear", "log"]
        with open(
            os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
        ) as f:
            stats = json.load(f)
            pitch_min, pitch_max = stats["pitch"][:2]
            energy_min, energy_max = stats["energy"][:2]

        if pitch_quantization == "log":
            self.pitch_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(pitch_min), np.log(pitch_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.pitch_bins = nn.Parameter(
                torch.linspace(pitch_min, pitch_max, n_bins - 1),
                requires_grad=False,
            )
        if energy_quantization == "log":
            self.energy_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(energy_min), np.log(energy_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.energy_bins = nn.Parameter(
                torch.linspace(energy_min, energy_max, n_bins - 1),
                requires_grad=False,
            )

        self.pitch_embedding = nn.Embedding(
            n_bins, model_config["transformer"]["encoder_hidden"]
        )
        self.energy_embedding = nn.Embedding(
            n_bins, model_config["transformer"]["encoder_hidden"]
        )

        self.pitch_downsize = nn.Linear(
            model_config["transformer"]["encoder_hidden"]*2,
            model_config["transformer"]["encoder_hidden"],
        )
        self.energy_downsize = nn.Linear(
            model_config["transformer"]["encoder_hidden"]*2,
            model_config["transformer"]["encoder_hidden"],
        )
        

    def get_pitch_embedding(self, x, target, mask, control):
        prediction = self.pitch_predictor(x, mask)
        if target is not None:
            embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
        else:
            prediction = prediction * control
            embedding = self.pitch_embedding(
                torch.bucketize(prediction, self.pitch_bins)
            )
        return prediction, embedding

    def get_energy_embedding(self, x, target, mask, control):
        prediction = self.energy_predictor(x, mask)
        if target is not None:
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:
            prediction = prediction * control
            embedding = self.energy_embedding(
                torch.bucketize(prediction, self.energy_bins)
            )
        return prediction, embedding

    def forward(self, x, src_mask, mel_mask=None, max_len=None, pitch_target=None, 
                energy_target=None, duration_target=None, p_control=1.0, 
                e_control=1.0, d_control=1.0):
        
        log_duration_prediction = self.duration_predictor(x, src_mask)
        
        if self.pitch_feature_level == "phoneme_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                x, pitch_target, src_mask, p_control
            )
            x = self.pitch_downsize(torch.cat((x, pitch_embedding), dim=-1))
            # x = x + pitch_embedding
        
        if self.energy_feature_level == "phoneme_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, src_mask, e_control
            )
            x = self.energy_downsize(torch.cat((x, energy_embedding), dim=-1))
            # x = x + energy_embedding

        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len)
            duration_rounded = duration_target
        else:
            duration_rounded = torch.clamp(
                (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
                min=0,
            )
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            mel_mask = get_mask_from_lengths(mel_len)

        if self.pitch_feature_level == "frame_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                x, pitch_target, mel_mask, p_control
            )
            x = x + pitch_embedding
        if self.energy_feature_level == "frame_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, mel_mask, p_control
            )
            x = x + energy_embedding

        return (
            x,
            pitch_prediction,
            energy_prediction,
            log_duration_prediction,
            duration_rounded,
            mel_len,
            mel_mask,
        )

class LengthRegulator(nn.Module):
    """Length Regulator"""

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        device = x.device
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(device)

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len

class VariancePredictor(nn.Module):
    """Duration, Pitch and Energy Predictor"""

    def __init__(self, model_config):
        super(VariancePredictor, self).__init__()

        self.input_size = model_config["transformer"]["encoder_hidden"]
        self.filter_size = model_config["variance_predictor"]["filter_size"]
        self.kernel = model_config["variance_predictor"]["kernel_size"]
        self.conv_output_size = model_config["variance_predictor"]["filter_size"]
        self.dropout = model_config["variance_predictor"]["dropout"]

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=1,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out

class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        w_init="linear",
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x

EPS = 1e-10

class VQLayer(nn.Module):
    '''
        VQ-layer modified from
            https://github.com/iamyuanchung/VQ-APC/blob/283d338/vqapc_model.py
    '''
    def __init__(self, input_size, codebook_size, code_dim, gumbel_temperature):
        '''
            Defines a VQ layer that follows an RNN layer.
            input_size: an int indicating the pre-quantized input feature size,
                usually the hidden size of RNN.
            codebook_size: an int indicating the number of codes.
            code_dim: an int indicating the size of each code. If not the last layer,
                then must equal to the RNN hidden size.
            gumbel_temperature: a float indicating the temperature for gumbel-softmax.
        '''
        super(VQLayer, self).__init__()
        # Directly map to logits without any transformation.
        self.codebook_size = codebook_size
        self.vq_logits = nn.Linear(input_size, codebook_size)
        self.gumbel_temperature = gumbel_temperature
        self.codebook_CxE = nn.Linear(codebook_size, code_dim, bias=False)
        self.token_usg = np.zeros(codebook_size)

    def forward(self, inputs_BxLxI, testing, lens=None):
        logits_BxLxC = self.vq_logits(inputs_BxLxI)
        if testing:
            # During inference, just take the max index.
            shape = logits_BxLxC.size()
            _, ind = logits_BxLxC.max(dim=-1)
            onehot_BxLxC = torch.zeros_like(logits_BxLxC).view(-1, shape[-1])
            onehot_BxLxC.scatter_(1, ind.view(-1, 1), 1)
            onehot_BxLxC = onehot_BxLxC.view(*shape)
        else:
            onehot_BxLxC = F.gumbel_softmax(logits_BxLxC, tau=self.gumbel_temperature, 
                                          hard=True, eps=EPS, dim=-1)
            self.token_usg += onehot_BxLxC.detach().cpu()\
                        .reshape(-1,self.codebook_size).sum(dim=0).numpy()
        codes_BxLxE = self.codebook_CxE(onehot_BxLxC)

        return logits_BxLxC, codes_BxLxE

    def report_ppx(self):
        ''' Computes perplexity of distribution over codebook '''
        acc_usg = self.token_usg/sum(self.token_usg)
        return 2**sum(-acc_usg * np.log2(acc_usg+EPS))

    def report_usg(self):
        ''' Computes usage each entry in codebook '''
        acc_usg = self.token_usg/sum(self.token_usg)
        # Reset
        self.token_usg = np.zeros(self.codebook_size)
        return acc_usg

class MaskConvBlock(nn.Module):
    """ Masked Convolution Blocks as described in NPC paper """
    def __init__(self, input_size, hidden_size, kernel_size, mask_size):
        super(MaskConvBlock, self).__init__()
        assert kernel_size-mask_size>0,"Mask > kernel somewhere in the model"
        # CNN for computing feature (ToDo: other activation?)
        self.act = nn.Tanh()
        self.pad_size = (kernel_size-1)//2
        self.conv = nn.Conv1d(in_channels=input_size,
                               out_channels=hidden_size,
                               kernel_size=kernel_size,
                               padding=self.pad_size 
                               )
        # Fixed mask for NPC
        mask_head = (kernel_size-mask_size)//2
        mask_tail = mask_head + mask_size
        conv_mask = torch.ones_like(self.conv.weight)
        conv_mask[:,:,mask_head:mask_tail] = 0
        self.register_buffer('conv_mask', conv_mask)

    def forward(self, feat):
        feat = nn.functional.conv1d(feat, 
                                    self.conv_mask*self.conv.weight,
                                    bias=self.conv.bias,
                                    padding=self.pad_size
                                   )
        feat = feat.permute(0,2,1) # BxCxT -> BxTxC
        feat = self.act(feat)
        return feat

class ConvBlock(nn.Module):
    """ Convolution Blocks as described in NPC paper """
    def __init__(self, input_size, hidden_size, residual, dropout, 
                 batch_norm, activate):
        super(ConvBlock, self).__init__()
        self.residual = residual
        if activate == 'relu':
            self.act = nn.ReLU()
        elif activate == 'tanh':
            self.act = nn.Tanh()
        else:
            raise NotImplementedError
        self.conv = nn.Conv1d(input_size,
                              hidden_size,
                              kernel_size=3,
                              stride=1,
                              padding=1
                             )
        self.linear = nn.Conv1d(hidden_size,
                              hidden_size,
                              kernel_size=1,
                              stride=1,
                              padding=0
                             )
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn1 = nn.BatchNorm1d(hidden_size)
            self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, feat):
        res = feat
        out = self.conv(feat)
        if self.batch_norm:
            out = self.bn1(out)
        out = self.act(out)
        out = self.linear(out)
        if self.batch_norm:
            out = self.bn2(out)
        out = self.dropout(out)
        if self.residual:
            out = out + res
        return self.act(out)

class NPC(nn.Module):
    """ NPC model with stacked ConvBlocks & Masked ConvBlocks """
    def __init__(self, config):
        super(NPC, self).__init__()

        kernel_size = config['npc']['kernel_size']
        mask_size = config['npc']['mask_size']
        n_blocks = config['npc']['n_blocks']
        hidden_size = config['npc']['hidden_size']
        input_size = config['npc']['input_size']
        dropout = config['npc']['dropout']
        batch_norm = config['npc']['batch_norm']
        activate = config['npc']['activate']
        residual = config['npc']['residual']
        disable_cross_layer = config['npc']['disable_cross_layer']
        vq = config['npc']['vq']
        dim_bottleneck = config['npc']['dim_bottleneck']

        # Setup
        assert kernel_size%2==1,'Kernel size can only be odd numbers'
        assert mask_size%2==1,'Mask size can only be odd numbers'
        assert n_blocks>=1,'At least 1 block needed'
        self.code_dim = hidden_size
        self.n_blocks = n_blocks
        self.input_mask_size = mask_size
        self.kernel_size = kernel_size
        self.disable_cross_layer = disable_cross_layer
        self.apply_vq = vq is not None
        self.apply_ae = dim_bottleneck is not None
        if self.apply_ae:
            assert not self.apply_vq
            self.dim_bottleneck = dim_bottleneck

        # Build blocks
        self.blocks, self.masked_convs = [], []
        cur_mask_size = mask_size
        for i in range(n_blocks):
            h_dim = input_size if i==0 else hidden_size
            res = False if i==0 else residual
            # ConvBlock
            self.blocks.append(ConvBlock(h_dim, hidden_size, res,
                                         dropout, batch_norm, activate))
            # Masked ConvBlock on each or last layer
            cur_mask_size = cur_mask_size + 2 
            if self.disable_cross_layer and (i!=(n_blocks-1)):
                self.masked_convs.append(None)
            else:
                self.masked_convs.append(MaskConvBlock(hidden_size, 
                                                       hidden_size, 
                                                       kernel_size, 
                                                       cur_mask_size))
        self.blocks = nn.ModuleList(self.blocks)
        self.masked_convs = nn.ModuleList(self.masked_convs)

        # Creates N-group VQ
        if self.apply_vq:
            self.vq_layers = []
            vq_config = copy.deepcopy(vq)
            codebook_size = vq_config.pop('codebook_size')
            self.vq_code_dims = vq_config.pop('code_dim')
            assert len(self.vq_code_dims)==len(codebook_size)
            assert sum(self.vq_code_dims)==hidden_size
            for cs,cd in zip(codebook_size,self.vq_code_dims):
                self.vq_layers.append(VQLayer(input_size=cd,
                                              code_dim=cd,
                                              codebook_size=cs,
                                              **vq_config))
            self.vq_layers = nn.ModuleList(self.vq_layers)

        # Back to spectrogram
        if self.apply_ae:
            self.ae_bottleneck = nn.Linear(hidden_size, 
                                           self.dim_bottleneck,bias=False)
            self.postnet = nn.Linear(self.dim_bottleneck, input_size)
        else:
            self.postnet = nn.Linear(hidden_size, input_size)
    
    def create_msg(self):
        msg_list = []
        msg_list.append('Model spec.| Method = NPC\t| # of Blocks = {}\t'\
                        .format(self.n_blocks))
        msg_list.append('           | Desired input mask size = {}'\
                        .format(self.input_mask_size))
        msg_list.append('           | Receptive field size = {}'\
                        .format(self.kernel_size+2*self.n_blocks))
        return msg_list

    def report_ppx(self):
        ''' Returns perplexity of VQ distribution '''
        if self.apply_vq:
            # ToDo: support more than 2 groups
            rt = [vq_layer.report_ppx() for vq_layer in self.vq_layers]+[None]
            return rt[0],rt[1]
        else:
            return None, None

    def report_usg(self):
        ''' Returns usage of VQ codebook '''
        if self.apply_vq:
            # ToDo: support more than 2 groups
            rt = [vq_layer.report_usg() for vq_layer in self.vq_layers]+[None]
            return rt[0],rt[1]
        else:
            return None, None

    def get_unmasked_feat(self, sp_seq, n_layer):
        ''' Returns unmasked features from n-th layer ConvBlock '''
        unmasked_feat = sp_seq.permute(0,2,1) # BxTxC -> BxCxT
        for i in range(self.n_blocks):
            unmasked_feat = self.blocks[i](unmasked_feat)
            if i == n_layer:
                unmasked_feat = unmasked_feat.permute(0,2,1)
                break
        return unmasked_feat

    def forward(self, sp_seq, testing=False):
        # BxTxC -> BxCxT (reversed in Masked ConvBlock)
        unmasked_feat = sp_seq.permute(0,2,1)
        # Forward through each layer
        for i in range(self.n_blocks):
            unmasked_feat = self.blocks[i](unmasked_feat)
            if self.disable_cross_layer:
                # Last layer masked feature only
                if i==(self.n_blocks-1):
                    feat = self.masked_convs[i](unmasked_feat)
            else:
                # Masked feature aggregation
                masked_feat = self.masked_convs[i](unmasked_feat)
                if i == 0:
                    feat = masked_feat
                else:
                    feat = feat + masked_feat
        # Apply bottleneck and predict spectrogram
        if self.apply_vq:
            q_feat = []
            offet = 0
            for vq_layer,cd in zip(self.vq_layers,self.vq_code_dims):
                _, q_f = vq_layer(feat[:,:,offet:offet+cd], testing)
                q_feat.append(q_f)
                offet += cd
            q_feat = torch.cat(q_feat,dim=-1)
            pred = self.postnet(q_feat)
        elif self.apply_ae:
            feat = self.ae_bottleneck(feat)
            pred  = self.postnet(feat) 
        else:
            pred = self.postnet(feat)
        return pred, feat

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Use is_grad_enabled to determine whether we are in training mode
    if not torch.is_grad_enabled():
        # In prediction mode, use mean and variance obtained by moving average
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 3, 4)
        if len(X.shape) == 2:
            # When using a fully connected layer, calculate the mean and
            # variance on the feature dimension
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        elif len(X.shape) == 3:
            # When using a one-dimensional convolutional layer over the (N,L) slice,
            # it is commonly referred to as Temporal Batch Normalization, 
            # calculate the mean and variance on the channel dimension (axis=1). 
            # Here we need to maintain the shape of X, so that the broadcasting
            # operation can be carried out later.
            mean = X.mean(dim=(0, 2), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2), keepdim=True)
        else:
            # When using a two-dimensional convolutional layer, calculate the
            # mean and variance on the channel dimension (axis=1). Here we
            # need to maintain the shape of X, so that the broadcasting
            # operation can be carried out later
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # In training mode, the current mean and variance are used
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean = (1.0 - momentum) * moving_mean + momentum * mean
        moving_var = (1.0 - momentum) * moving_var + momentum * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y, moving_mean.data, moving_var.data

class BatchNorm(nn.Module):
    """ Custom BatchNorm function since nn.BatchNorm2d won't work with DDP"""
    # num_features: the number of outputs for a fully connected layer or the
    # number of output channels for a convolutional layer. num_dims: 2 for a
    # fully connected layer and 4 for a convolutional layer
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        elif num_dims == 3:
            shape = (1, num_features, 1)
        else:
            shape = (1, num_features, 1, 1)
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # The variables that are not model parameters are initialized to 0 and
        # 1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # If X is not on the main memory, copy moving_mean and moving_var to
        # the device where X is located
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # Save the updated moving_mean and moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.1)
        return Y
    