import concurrent.futures
import threading

import torch.nn as nn
import torch.nn.functional as F
import torch
#from mamba_ssm import Mamba
from mamba_ssm import Mamba

from layers.SelfAttention_Family import FullAttention, AttentionLayer
class EncoderLayer(nn.Module):
    def __init__(self, attention, attention_r, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.attention_r = attention_r
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.man = Mamba(
            d_model=11,  # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=2,  # Local convolution width
            expand=1,  # Block expansion factor)
        )
        self.man2 = Mamba(
            d_model=11,  # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=2,  # Local convolution width
            expand=1,  # Block expansion factor)
        )
        self.a = AttentionLayer(
                        FullAttention(False, 2, attention_dropout=0.1,
                                      output_attention=True), 11,1)
    
    def perm(self,C):
        idx = torch.randperm(C)
        idx_inv = torch.argsort(idx)
        return idx, idx_inv
    
    def forward(self, x, attn_mask=None, tau=None, delta=None,train=True):
        x1 = self.attention(x)
        x2 = self.attention(x.flip(dims=[1])).flip(dims=[1])
        x = x + (x1+x2)
        
        attn = 1
        if train:
            loss_reg = torch.sqrt(torch.sum((x1 - x2) ** 2)/(x.shape[0]*x.shape[1]))
        else:
            loss_reg = 0
        
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn, loss_reg


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None,train=True):
        # x [B, L, D]
        attns = []
        loss_reg = 0
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn, loss = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta,train=train)
                
                x = conv_layer(x)
                attns.append(attn)
            x, attn,loss = self.attn_layers[-1](x, tau=tau, delta=None,train=train)
            loss_reg += loss
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn, loss = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta,train=train)
                loss_reg += loss
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)
        return x, attns,loss_reg

