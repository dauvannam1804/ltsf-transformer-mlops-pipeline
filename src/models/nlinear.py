import torch.nn as nn

class NLinear(nn.Module):
    def __init__(self, seq_len, pred_len=7):
        super().__init__()
        self.linear = nn.Linear(seq_len, pred_len)
    
    def forward(self, x):
        last_value = x[:, -1:]
        x_norm = x - last_value
        pred = self.linear(x_norm)
        return pred + last_value
