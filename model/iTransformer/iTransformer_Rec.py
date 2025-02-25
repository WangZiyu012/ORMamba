import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np

from mamba_ssm import Mamba
class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.mask_p = configs.mask_p
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        #self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        
        self.head_type = configs.head_type
        if self.head_type == "pretrain":
            self.projector = nn.Linear(configs.d_model, configs.seq_len, bias=True)
            self.lin_proj = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        else:
            self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
            

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N 
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out
    
    def compute_correlation_matrix(self, tensor):
        B, L, N = tensor.shape
    
        # Step 1: Center the data
        mean = tensor.mean(dim=1, keepdim=True)  # Mean over the L dimension
        centered_tensor = tensor - mean  # Centered tensor of shape (B, L, N)
        
        # Step 2: Compute covariance
        covariance = centered_tensor.transpose(1, 2) @ centered_tensor / L  # Covariance matrix of shape (B, N, N)
        
        # Step 3: Compute standard deviations
        std_dev = torch.sqrt(torch.diagonal(covariance, dim1=1, dim2=2)).unsqueeze(2)+1e-5  # Standard deviations of shape (B, N, 1)
        
        # Step 4: Form correlation matrix
        correlation_matrix = covariance / (std_dev @ std_dev.transpose(1, 2))  # Correlation matrix of shape (B, N, N)
        
        return correlation_matrix
    
    def random_mask(self, tensor, mask_percent=0.5) :
        batch_size, dim, num_channels = tensor.shape
        mask = torch.rand(batch_size, dim) >= mask_percent # Randomly set 50% of the values to True (unmasked)
        mask = mask.unsqueeze(-1).expand(-1, -1, num_channels).int()  # Shape becomes [batch_size, dim, num_channels]
        mask = mask.to(tensor.device)
        masked_tensor = tensor * mask  # Values at masked positions will be set to 0
        return masked_tensor, mask
    
    def pretrain(self, x_enc, x_mark_enc,train,mi=False):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        x_enc, mask = self.random_mask(x_enc,mask_percent=self.mask_p)
        _, _, N = x_enc.shape # B L N
        corr_gt = self.compute_correlation_matrix(x_enc)
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        #enc_out = self.head(enc_out).permute(0, 2, 1)[:, :, :N] 
        enc_out = enc_out[:,:N,:]
        pred = self.lin_proj(enc_out).permute(0,2,1) # [32,]
        
        loss = torch.mean(((pred-x_enc)**2)*mask)

        return loss,0
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,train,mi=False):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :],0  # [B, L, D]