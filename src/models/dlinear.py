import torch
import torch.nn as nn

class DLinear(nn.Module):
    """Decomposition Linear for univariate time series - handles trend and seasonality"""
    def __init__(self, seq_len, pred_len=7, moving_avg=5):
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.moving_avg = min(moving_avg, seq_len - 1)

        # Linear layers for trend and seasonal components
        self.linear_trend = nn.Linear(self.seq_len, self.pred_len)
        self.linear_seasonal = nn.Linear(self.seq_len, self.pred_len)

        # Create moving average kernel for trend extraction
        self.avg_kernel: torch.Tensor
        self.register_buffer(
            'avg_kernel',
            torch.ones(1, 1, self.moving_avg) / self.moving_avg
        )

    def decompose(self, x):
        """Decompose series into trend and seasonal components"""
        batch_size, seq_len = x.shape
        x_reshaped = x.unsqueeze(1)  # [B, 1, L]

        # Apply moving average for trend
        padding = self.moving_avg // 2
        x_padded = torch.nn.functional.pad(
            x_reshaped, (padding, padding), mode='replicate'
        )
        trend = torch.nn.functional.conv1d(x_padded, self.avg_kernel, padding=0)
        trend = trend.squeeze(1)

        # Adjust length if needed
        if trend.shape[1] != seq_len:
            trend = torch.nn.functional.interpolate(
                trend.unsqueeze(1),
                size=seq_len,
                mode='linear',
                align_corners=False,
            ).squeeze(1)

        # Seasonal = x - trend
        seasonal = x - trend

        return trend, seasonal

    def forward(self, x):
        # x: [batch_size, seq_len]
        trend, seasonal = self.decompose(x)

        # Predict each component
        trend_pred = self.linear_trend(trend)
        seasonal_pred = self.linear_seasonal(seasonal)

        # Final prediction = trend + seasonal
        return trend_pred + seasonal_pred