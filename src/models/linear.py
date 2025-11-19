import torch.nn as nn

class Linear(nn.Module):
    """Simple Linear model for univariate time series forecasting"""
    def __init__(self, seq_len, pred_len=7):
        super(Linear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [batch_size, seq_len] - historical prices
        # Output: [batch_size, pred_len] - 7-day future predictions
        return self.linear(x)
