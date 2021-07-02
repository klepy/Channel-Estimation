# -*- coding: utf-8 -*-
"""
Created on Thu May  6 23:40:14 2021

@author: klepy
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import utils

class DnCNN(nn.Module):
    def __init__(self,num_of_layers,features):
        channels = 2
        # num_of_layers = 4
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        # features = 4
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        # layers.append(nn.Tanh())

        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
            # layers.append(nn.Tanh())
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        
        # Multi Unit Residual Learning
        self.dncnn = nn.Sequential(*layers)

        

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y-out



        