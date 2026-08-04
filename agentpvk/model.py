"""
Autoregressive Transformer for SMILES generation.
Causal attention, predicts next token from previous tokens.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=256, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class SMILESGenerator(nn.Module):
    """Autoregressive SMILES generator with causal transformer."""

    def __init__(self, vocab_size, d_model=256, n_head=8, n_layer=4,
                 d_ff=1024, dropout=0.1, max_len=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len

        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = SinusoidalPE(d_model, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=d_ff,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)

    def forward(self, token_ids, mask=None):
        x = self.embed(token_ids)
        x = self.pos_enc(x)
        L = x.size(1)
        causal_mask = torch.triu(torch.full((L, L), float('-inf'), device=x.device), diagonal=1)
        x = self.encoder(x, mask=causal_mask, src_key_padding_mask=mask)
        return self.head(self.ln(x))
