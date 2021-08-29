import random
from pprint import pprint
from dataset import ModelNet40
import numpy as np
import torch
import torch.nn as nn

class Gaussian(nn.Module):
    def __init__(self):
        super(Gaussian, self).__init__()
        self.work = nn.Parameter(torch.Tensor([0]))
    def forward(self, x):
        return self.work * x

class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.sigma = nn.Parameter(torch.Tensor([1.0]))
        self.gaussian = nn.ModuleList([Gaussian(),Gaussian(),Gaussian(),Gaussian()])
        self.lin = nn.Linear(20,40)
    def forward(self, x):
        x = self.sigma * x
        x = self.gaussian(x)
        return self.lin(x)

net = TestNet()
net.train()

for p in net.parameters():
    print(p.data)

for i in range(len(net.gaussian)):
    # value = nn.Parameter(torch.Tensor([max(net.gaussian[i].work.item(), 1e-4)]))
    # net.gaussian[i].work = value
    net.gaussian[i].work = nn.Parameter(torch.clamp(net.gaussian[i].work, min=1e-4))

for p in net.parameters():
    print(p.data)