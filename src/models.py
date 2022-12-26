import os
import sys
import time
import math
import torch
import pickle
import scipy.io as sio
import seaborn as sns
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable

sys.path.append('./')
import src.Deep_PCE as dPC

class DNN(nn.Module):
    def __init__(self, dim, num_c, hiden_neurons,ActFun='Relu'):
        super(DNN, self).__init__()
        
        if ActFun == 'Relu' or ActFun == 'relu' or ActFun == 'ReLu':
            self.actfun = nn.ReLU()
        elif ActFun == 'GELU' or ActFun == 'gelu':
            self.actfun = nn.GELU()
        hiden_neurons.insert(0, dim)
        hiden_neurons.append(num_c)
        layer_num = len(hiden_neurons)
        layers = []
        for i in range(layer_num-1):
            layers.append(nn.Linear(hiden_neurons[i], hiden_neurons[i+1]))
            layers.append(self.actfun)
        del layers[-1]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        c = self.layers(x)
        return c