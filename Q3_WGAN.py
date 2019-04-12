import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import numpy as np
import matplotlib.pyplot as plt
import samplers
import argparse

class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.convs = nn.Sequential()
        layer1 = nn.Conv2d(3, 128, 5, padding=2, stride=2)
        layer2 = nn.Conv2d(128, 256, 5, padding=2, stride=2)
        layer3 = nn.Conv2d(256, 512, 5, padding=2, stride=2)
        nl = nn.LeakyReLU(negative_slope=0.2)
        self.convs.append(layer1)
        self.convs.append(nl)
        self.convs.append(layer2)
        self.convs.append(nl)
        self.convs.append(layer3)
        self.convs.append(nl)
        
        self.linear1 = nn.Linear(4*4*512, 100)
        self.linear2 = nn.Linear(100, 1)
        return
        
    
    def forward(self, x)
        out = self.convs(x).view(32, 4*4*512)
        out = F.sigmoid(self.linear2(F.leaky_relu(self.linear1(out), negative_slope=0.2)))
        return out
    

class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.linear1 = nn.Linear(100, 4*4*512)
        
        self.deconvs = nn.Sequential()
        layer1 = nn.ConvTranspose2d(512, 256, 5, padding=2, stride=2)
        layer2 = nn.ConvTranspose2d(256, 128, 5, padding=2, stride=2)
        layer3 = nn.ConvTranspose2d(128, 3, 5, padding=2, stride=2)
        nl = nn.LeakyReLU(negative_slope=0.2)
        self.deconvs.append(layer1)
        self.deconvs.append(nl)
        self.deconvs.append(layer2)
        self.deconvs.append(nl)
        self.deconvs.append(layer3)
        self.deconvs.append(nn.tanh())
        return
    
    
    def forward(self, z)
        out = F.leaky_relu(self.linear1(z), negative_slope=0.2).view(32, 512, 4, 4)
        out = self.deconvs(out)
        return out