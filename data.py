# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 20:16:27 2021

@author: Ji Ho Choi
"""
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import utils
import torch
from torch.utils.data import Dataset
# import torchsummary



#%%
class IRSDataset(Dataset):  
    def __init__(self, data, labels):
        super().__init__()
        x = data
        y = labels
        self.x = torch.from_numpy(x).cuda()
        self.y = torch.from_numpy(y).cuda()
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


















