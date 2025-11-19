import torch.nn as nn

class NLinear(nn.Module):
    """Normalized Linear for univariate time series - handles distribution shift"""
    def __init__(self, seq_len, pred_len=7):
        super(NLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Linear projection from seq_len → pred_len
        self.linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [batch_size, seq_len]

        # 1) Normalize by subtracting the last value
        last_value = x[:, -1].unsqueeze(1)          # [B, 1]
        x_normalized = x - last_value               # [B, seq_len]

        # 2) Linear projection in normalized space
        pred_normalized = self.linear(x_normalized) # [B, pred_len]

        # 3) Add the last value back to shift prediction
        pred = pred_normalized + last_value         # broadcast to [B, pred_len]

        return pred
