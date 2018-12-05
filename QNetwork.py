import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        return self.l2(torch.relu(self.l1(x)))