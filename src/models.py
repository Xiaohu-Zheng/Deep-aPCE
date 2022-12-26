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


class PCNN(nn.Module):
    def __init__(self, dim, num_c, hiden_neurons, c, ActFun='Relu'):
        super(PCNN, self).__init__()
        self.c = nn.Parameter(c)
        
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

    def forward(self, x, phi_x):
        c = self.layers(x)
        y_pce = torch.sum(phi_x * self.c, dim=1).view(-1, 1)
        return c, y_pce


class MCQR_PCNN(nn.Module):
    def __init__(self, dim, num_c_dpce, hiden_neurons, c_pcnn, order_pcnn, order_mat_pcnn, pc_pcnn, p_orders_pcnn, ActFun='Relu'):
        super(MCQR_PCNN, self).__init__()
        self.c_pcnn = nn.Parameter(c_pcnn)
        self.order_pcnn = order_pcnn
        self.order_mat_pcnn = order_mat_pcnn
        self.pc_pcnn = pc_pcnn
        self.p_orders_pcnn = p_orders_pcnn
        self.num_c_dpce = num_c_dpce
        
        if ActFun == 'Relu' or ActFun == 'relu' or ActFun == 'ReLu':
            self.actfun = nn.ReLU()
        elif ActFun == 'GELU' or ActFun == 'gelu':
            self.actfun = nn.GELU()
        hiden_neurons.insert(0, dim+1)
        hiden_neurons.append(self.num_c_dpce)
        layer_num = len(hiden_neurons)
        layers = []
        for i in range(layer_num-1):
            layers.append(nn.Linear(hiden_neurons[i], hiden_neurons[i+1]))
            layers.append(self.actfun)
        del layers[-1]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        c = self.layers(x)
        tau_mean = torch.zeros(x.size(0), x.size(1)).to(x.device)
        tau_mean[:, -1:] = 0.5 + tau_mean[:, -1:]
        tau_std = torch.ones(x.size(0), x.size(1)).to(x.device)
        tau_std[:, -1:] = (1/12) ** 0.5 * tau_std[:, -1:]
        x_tau01 = (x - tau_mean) / tau_std
        phi_x = dPC.orthogonal_basis(x_tau01, self.order_pcnn, self.order_mat_pcnn, 
                                     self.pc_pcnn, self.p_orders_pcnn)
        y_pce = torch.sum(phi_x * self.c_pcnn, dim=1).view(-1, 1)
        return c, y_pce