"""Hybrid LSTM -> Transformer encoder for ISL gesture sequence classification.

Architecture (as required by PROJECT_CONTEXT.md):

    input (B, T, D) landmark sequence
      -> per-frame Linear projection to d_model
      -> 2-layer LSTM (short-term local motion between adjacent frames)
      -> sinusoidal positional encoding
      -> 4-layer Transformer Encoder, 8 heads, padding mask
         (long-range dependencies across the full gesture sequence)
      -> masked mean-pool over time
      -> dropout -> Linear classification head (softmax over vocabulary)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (batch-first)."""

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class HybridLSTMTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int = 256,
        lstm_layers: int = 2,
        transformer_layers: int = 4,
        nhead: int = 8,
        ff_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.d_model = d_model

        self.proj = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False,
        )

        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers
        )
        self.encoder_norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """x: (B, T, D) float32; mask: (B, T) bool, True = valid frame."""
        if x.dim() != 3:
            raise ValueError(f"Expected (B, T, D) input, got {tuple(x.shape)}")

        if mask is None:
            mask = torch.ones(x.size(0), x.size(1), dtype=torch.bool, device=x.device)
        pad_mask = ~mask  # TransformerEncoder: True where padded

        h = self.proj(x)
        h = self.input_norm(h)
        h, _ = self.lstm(h)
        h = self.pos_enc(h)
        h = self.transformer(h, src_key_padding_mask=pad_mask)
        h = self.encoder_norm(h)

        # Masked mean-pool over time (guard against all-padded rows).
        m = mask.unsqueeze(-1).to(h.dtype)              # (B, T, 1)
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return self.head(pooled)


if __name__ == "__main__":
    # Quick shape self-test.
    model = HybridLSTMTransformer(input_dim=1632, num_classes=50)
    x = torch.randn(4, 96, 1632)
    mask = torch.ones(4, 96, dtype=torch.bool)
    mask[0, 60:] = False  # simulate padding
    out = model(x, mask)
    assert out.shape == (4, 50), out.shape
    print("HybridLSTMTransformer OK:", tuple(out.shape))
