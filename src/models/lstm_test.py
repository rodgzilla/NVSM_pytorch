import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
lstm = nn.LSTM(
    3, # 3D input
    7, # 7D output
    3  # number of stacked lstm layers
)
inputs = torch.randn(
    12, # sequences of length 12
    4,  # 4 sequences (batch size)
    3   # 3D sequence items
)
h0 = torch.randn(
    3, # lstm layers (3) * num directions (1)
    4, # batch size
    7  # output dim
)
c0 = torch.randn(
    3, # lstm layers (3) * num directions (1)
    4, # batch size
    7  # output dim
)
out, (hn, cn) = lstm(inputs, (h0, c0))
