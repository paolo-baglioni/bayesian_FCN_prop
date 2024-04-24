import json
import torch
import os
import sys
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.transforms import ToTensor
import torchvision.datasets as datasets

class Erf(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.erf(x)

class NeuralNetwork(nn.Module):
    def __init__(self,N0, N1, bias_flag):
        super(NeuralNetwork, self).__init__()
        self.N0 = N0
        self.N1 = N1
        self.first_layer = nn.Linear(N0, N1, bias=bias_flag)
        nn.init.normal_(self.first_layer.weight, std = 1)
        self.activ = Erf()
        self.second_layer = nn.Linear(N1, 1, bias=bias_flag)
        nn.init.normal_(self.second_layer.weight, std = 1)
    def forward(self, x):
        x = self.first_layer( torch.div(x , self.N0**0.5 ))
        x = self.activ(x)
        x = self.second_layer( torch.div(x , self.N1**0.5 ))
        return x
        
class LangevinMC(torch.optim.Optimizer) :
    def __init__(self, params, epsilon=0.001, temperature= 0.001):
        super(LangevinMC, self).__init__(params, defaults={'epsilon': epsilon, 'temperature': temperature })

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['epsilon']
            t = group['temperature']
            for p in group['params']:
                d_p = torch.randn_like(p) * (2*lr*t)**0.5
                d_p.add_(p.grad, alpha=-lr)
                p.add_(d_p)