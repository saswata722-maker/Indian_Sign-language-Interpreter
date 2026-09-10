"""
Sequence Model Architecture
Hybrid architecture combining Bi-LSTM (local motion) with Transformer Encoder (global context).

Based on AI4Bharat INCLUDE's transformer approach with LSTM front-end for temporal modeling.
"""

import torch
import torch.nn as nn


class HybridSignModel(nn.Module):
    """
    Hybrid Bi-LSTM + Transformer Encoder for sign language recognition.

    Architecture:
    1. Input projection: Linear(input_dim → d_model)
    2. Local motion extractor: Bidirectional LSTM
    3. Global context extractor: Transformer Encoder
    4. Classification: Mean pooling → LayerNorm → Dropout → Linear

    Args:
        input_dim: Feature dimension per frame (234 for pose+hands+flags).
        num_classes: Number of sign classes.
        d_model: Transformer embedding dimension.
        nhead: Number of attention heads.
        num_transformer_layers: Number of transformer encoder layers.
        dropout: Dropout probability.
    """

    def __init__(self, input_dim=234, num_classes=8, d_model=128, nhead=4,
                 num_transformer_layers=2, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)

        # Local motion extractor (Bi-LSTM)
        self.lstm = nn.LSTM(
            d_model, d_model // 2, num_layers=1,
            batch_first=True, bidirectional=True
        )

        # Global context extractor (Transformer Encoder)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers
        )

        # Temporal attention pooling (replaces mean pooling): lets the model
        # focus on the discriminative "signing" frames instead of weighting
        # all frames (incl. idle ones) equally.
        self.attn_pool = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(0.3),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (Batch, Sequence_Len, input_dim).

        Returns:
            logits: Tensor of shape (Batch, num_classes).
        """
        x = self.input_proj(x)                    # (B, T, d_model)
        lstm_out, _ = self.lstm(x)                # (B, T, d_model)
        trans_out = self.transformer(lstm_out)    # (B, T, d_model)

        # Attention pooling over time
        attn_scores = self.attn_pool(trans_out)   # (B, T, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (B, T, 1)
        pooled = (attn_weights * trans_out).sum(dim=1)    # (B, d_model)

        return self.classifier(pooled)            # (B, num_classes)
