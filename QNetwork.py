import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, n_input=4, num_hidden=128, n_output=2):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(n_input, num_hidden)
        self.l2 = nn.Linear(num_hidden, n_output)

    def forward(self, x):
        return self.l2(torch.relu(self.l1(x)))