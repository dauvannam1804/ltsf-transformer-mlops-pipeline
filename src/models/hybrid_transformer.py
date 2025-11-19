import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.dlinear import DLinear

class HybridDLinearTransformer(nn.Module):
    """Hybrid model: DLinear decomposition + Transformer for seasonal"""
    def __init__(self, seq_len, pred_len=7, moving_avg=5, d_model=64, n_heads=4, num_layers=1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        # 1) Decomposition (reuse DLinear)
        self.decomp = DLinear(seq_len, pred_len, moving_avg)

        # 2) Trend branch: Linear
        self.trend_linear = nn.Linear(seq_len, pred_len)

        # 3) Seasonal branch: Transformer
        self.embedding = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=128,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.seasonal_head = nn.Linear(d_model * seq_len, pred_len)

    def forward(self, x):
        # Decompose
        trend, seasonal = self.decomp.decompose(x)

        # Trend → Linear
        trend_pred = self.trend_linear(trend)

        # Seasonal → Transformer
        seasonal_seq = seasonal.unsqueeze(-1)          # [B, L, 1]
        seasonal_emb = self.embedding(seasonal_seq)    # [B, L, d_model]
        seasonal_encoded = self.transformer(seasonal_emb)
        seasonal_flat = seasonal_encoded.reshape(x.size(0), -1)
        seasonal_pred = self.seasonal_head(seasonal_flat)

        # Combine
        return trend_pred + seasonal_pred
