import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, seq_len, pred_len=7):
        super().__init__()
        self.linear = nn.Linear(seq_len, pred_len)
    
    def forward(self, x):
        return self.linear(x)
