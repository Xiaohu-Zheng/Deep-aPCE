import math
import torch
from torch import nn
import numpy as np

import src.Deep_PCE as dPC

class CalculatePCELoss(nn.Module):
    """The PCE regression loss.

    Args:
        x: the input
        y: the true value
        c: the coefficients of PCE
    Attributes:
        loss: the PCE loss result.
    """

    def __init__(self, order, order_mat, pc='apc', p_orders=None, sparse_basis=None, norm_factor=None):
        super(CalculatePCELoss, self).__init__()
        self.order_mat = order_mat
        self.order = order
        self.criterion = nn.L1Loss()
        self.pc = pc
        self.p_orders = p_orders
        self.sparse_basis = sparse_basis
        self.norm_factor = norm_factor

    def forward(self, x, y, c):
        y_pred = dPC.deep_pce_fun(x, c, self.order, self.order_mat, self.pc, 
                                  self.p_orders, self.sparse_basis, self.norm_factor)
        loss = self.criterion(y_pred, y)

        return loss


class CoefficientPCELoss(nn.Module):
    def __init__(self, lam_mean=1, lam_var=1):
        super(CoefficientPCELoss, self).__init__()
        self.lam_mean = lam_mean
        self.lam_var = lam_var

    def forward(self, y_pred, c):
        c_mean = c.mean(dim=0)
        c_inter = c_mean[1:]
        loss = self.lam_mean * torch.abs(c_mean[0] - y_pred.mean()) + \
               self.lam_var * torch.abs(torch.sum(c_inter ** 2) - y_pred.var())

        return loss


class CoefficientPCELoss_coeff_batch(nn.Module):
    def __init__(self, order, order_mat, device, pc='apc', p_orders=None):
        super(CoefficientPCELoss_coeff_batch, self).__init__()
        self.order_mat = order_mat
        self.order = order
        self.pc = pc
        self.p_orders = p_orders
        self.device = device

    def forward(self, x_coeff, net, coeff_batch_size=int(1e+06)):
        num_coeff = x_coeff.size(0)
        coeff_batch = math.ceil(num_coeff/coeff_batch_size)
        x_coeff_all = torch.chunk(x_coeff, coeff_batch)
        for j in range(len(x_coeff_all)):
            x_coeff_batch = x_coeff_all[j]
            x_coeff_batch = x_coeff_batch.to(self.device)
            c_batch = net(x_coeff_batch)[0]
            y_batch = dPC.deep_pce_fun(x_coeff_batch, c_batch, self.order, self.order_mat, self.pc, self.p_orders)
            if j==0:
                c_sum = c_batch.sum(dim=0).view(1, c_batch.size(1))
                y_sum = y_batch.sum()
                y_sum_square = (y_batch ** 2).sum()
            else:
                c_sum += c_batch.sum(dim=0).view(1, c_batch.size(1))
                y_sum += y_batch.sum()
                y_sum_square += (y_batch ** 2).sum()
        c_mean = (c_sum / num_coeff).view(-1)
        c_inter = c_mean[1:]
        y_mean = y_sum / num_coeff
        y_var = y_sum_square / num_coeff - y_mean ** 2
        loss = torch.abs(c_mean[0] - y_mean) + torch.abs(torch.sum(c_inter ** 2) - y_var)

        return loss
