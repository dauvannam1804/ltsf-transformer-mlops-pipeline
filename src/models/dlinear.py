import torch
import torch.nn as nn

class DLinear(nn.Module):
    def __init__(self, seq_len, pred_len=7, moving_avg=5):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.moving_avg = min(moving_avg, seq_len - 1)
        
        self.linear_trend = nn.Linear(seq_len, pred_len)
        self.linear_seasonal = nn.Linear(seq_len, pred_len)
        
        self.register_buffer('avg_kernel', 
                           torch.ones(1, 1, self.moving_avg) / self.moving_avg)
    
    def decompose(self, x):
        batch_size, seq_len = x.shape
        x_reshaped = x.unsqueeze(1)
        
        padding = self.moving_avg // 2
        x_padded = torch.nn.functional.pad(x_reshaped, (padding, padding), mode='replicate')
        trend = torch.nn.functional.conv1d(x_padded, self.avg_kernel)
        trend = trend.squeeze(1)
        
        if trend.shape[1] != seq_len:
            trend = torch.nn.functional.interpolate(
                trend.unsqueeze(1), size=seq_len,
                mode='linear', align_corners=False
            ).squeeze(1)
        
        seasonal = x - trend
        return trend, seasonal
    
    def forward(self, x):
        trend, seasonal = self.decompose(x)
        return self.linear_trend(trend) + self.linear_seasonal(seasonal)
