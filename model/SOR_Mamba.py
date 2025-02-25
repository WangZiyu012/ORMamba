import torch
import torch.nn as nn
from layers.SORMamba_EncDec import Encoder, EncoderLayer
from layers.Embed import DataEmbedding_inverted


import concurrent.futures
import threading

import torch.nn as nn
import torch.nn.functional as F
import torch

from layers.SelfAttention_Family import FullAttention, AttentionLayer

import torch
import torch.nn as nn
import math

from mamba_ssm import Mamba_rc as Mamba


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                        Mamba(
                            d_model=configs.d_model,  # Model dimension d_model
                            d_state=configs.d_state,  # SSM state expansion factor
                            d_conv=2,  # Local convolution width
                            expand=1,  # Block expansion factor)
                        ),
                        Mamba(
                            d_model=configs.d_model,  # Model dimension d_model
                            d_state=configs.d_state,  # SSM state expansion factor
                            d_conv=2,  # Local convolution width
                            expand=1,  # Block expansion factor)
                        ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.head_type = configs.head_type
        if self.head_type == "pretrain":
            self.head = nn.Linear(configs.d_model, configs.seq_len, bias=True)
            self.lin_proj = nn.Linear(configs.d_model, configs.d_model, bias=True)
        else:
            self.head = nn.Linear(configs.d_model, configs.pred_len, bias=True)
            

    def compute_correlation_matrix(self, tensor):
        B, L, N = tensor.shape
        mean = tensor.mean(dim=1, keepdim=True)  # Mean over the L dimension
        centered_tensor = tensor - mean  # Centered tensor of shape (B, L, N)
        covariance = centered_tensor.transpose(1, 2) @ centered_tensor / L  # Covariance matrix of shape (B, N, N)
        std_dev = torch.sqrt(torch.diagonal(covariance, dim1=1, dim2=2)).unsqueeze(2)+1e-5  # Standard deviations of shape (B, N, 1)
        correlation_matrix = covariance / (std_dev @ std_dev.transpose(1, 2))  # Correlation matrix of shape (B, N, N)
        return correlation_matrix

    def pretrain(self, x_enc, x_mark_enc,train):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape # B L N
        corr_gt = self.compute_correlation_matrix(x_enc)
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        enc_out, attns,loss_reg = self.encoder(enc_out, attn_mask=None,train=train)
        enc_out = enc_out[:,:N,:]
        enc_out = self.lin_proj(enc_out)
        
        corr_pred = self.compute_correlation_matrix(enc_out.permute(0,2,1))
        loss = torch.mean((corr_gt-corr_pred)**2)

        return loss,loss_reg
        
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec,train):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape # B L N
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        enc_out, attns,loss_reg = self.encoder(enc_out, attn_mask=None,train=train)
        dec_out = self.head(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates

        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out,loss_reg


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,train):
        dec_out,loss_reg = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec,train)
        return dec_out[:, -self.pred_len:, :],loss_reg  # [B, L, D]