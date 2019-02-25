import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
lstm = nn.LSTM(
    3, # 3D input
    7, # 7D output
    1  # number of stacked lstm layers
)
inputs = torch.randn(
    12, # sequences of length 12
    50,  # 50 sequences (batch size)
    3   # 3D sequence items
)
target = torch.randint(7, (50,))
hn = torch.randn(
    1, # lstm layers (1) * num directions (1)
    50, # batch size
    7  # output dim
)
cn = torch.randn(
    1, # lstm layers (1) * num directions (1)
    50, # batch size
    7  # output dim
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(lstm.parameters())

for i in range(5000):
    optimizer.zero_grad()
    hn, cn = hn.detach(), cn.detach()
    out, (hn, cn) = lstm(inputs, (hn, cn))
    pred = out[-1]
    loss = criterion(pred, target)
    loss.backward()
    optimizer.step()
    if i % 50 == 0:
        print(i, loss)
