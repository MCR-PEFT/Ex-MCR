import os
import torch
from torch import nn
import torch.nn.functional as F

from tqdm import tqdm

import copy

def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    elif activation.lower() == 'sigmoid':
        return nn.Sigmoid()
    else:
        return nn.ReLU(inplace=True)
def get_clones(module: nn.Module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class MLP(nn.Module):
    def __init__(
        self, channel=512, res_expansion=1.0, bias=True, activation='relu'):
        super().__init__()
        self.act = get_activation(activation)
        self.net1 = nn.Sequential(
            nn.Linear(channel, int(channel * res_expansion), bias=bias),
            nn.BatchNorm1d(int(channel * res_expansion)),
            self.act
        )
        self.net2 = nn.Sequential(
            nn.Linear(int(channel * res_expansion), channel, bias=bias),
            nn.BatchNorm1d(channel)
        )

    def forward(self, x):
        return self.net2(self.net1(x))

class Projector(torch.nn.Module):
    def __init__(self, init_mode):
        super(Projector, self).__init__()
 
        self.mlp1 = MLP(res_expansion=2)
        self.mlp2 = MLP(res_expansion=2)

        self.init_weights(init_mode)

    def forward(self, embs):
        embs = self.mlp1(embs)
        embs = self.mlp2(embs)
        return F.normalize(embs, dim=-1)

    def init_weights(self, mode):
        # initialize transformer
        if mode == 'eye':
            for m in self.parameters():
                if m.dim() > 1:
                    nn.init.eye_(m)
        elif mode == 'xav':
            for m in self.parameters():
                if m.dim() > 1:
                    nn.init.xavier_uniform_(m)
    
    def get_device(self):
        return next(self.parameters()).device

class Ex_MCR_Head(torch.nn.Module):
    def __init__(self):
        super(Ex_MCR_Head, self).__init__()
        self.Head1 = nn.Linear(512, 512, bias=True)
        self.Head2 = Projector('xav')
        
        for m in self.Head1.parameters():
            if m.dim() > 1:
                nn.init.eye_(m)

    def get_device(self):
        return next(self.parameters()).device
