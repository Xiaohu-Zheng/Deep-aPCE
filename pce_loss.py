import math
import torch
from torch import nn
import numpy as np

import data_process as dp


class PCELoss(nn.Module):
    """The quantile regression loss.

    Args:
        x: the input
        y: the true value
        c: the coefficients of PCE
    Attributes:
        loss: the PCE loss result.
    """

    def __init__(self, pattern, order=2):
        super(PCELoss, self).__init__()
        self.order = order
        self.pattern = pattern

    def forward(self, x, y, c, device):
        X1 = torch.ones((len(x), 1)).to(device)
        X2 = x
        if self.pattern == 'PCE':
            if self.order == 1:
                X = torch.cat((X1, X2), dim=1)
            if self.order == 2:
                X3 = x ** 2 - 1
                X = torch.cat((X1, X2, X3), dim=1)
            elif self.order == 3:
                X3 = x ** 2 - 1
                X4 = x ** 3 - 3 * x
                X = torch.cat((X1, X2, X3, X4), dim=1)
            elif self.order == 4:
                X3 = x ** 2 - 1
                X4 = x ** 3 - 3 * x
                X5 = 9 * x ** 4 + 45 * x ** 3 - 135 * x ** 2 - 27
                X = torch.cat((X1, X2, X3, X4, X5), dim=1)
        elif self.pattern == 'polynomial':
            if self.order == 1:
                X = torch.cat((X1, X2), dim=1)
            if self.order == 2:
                X3 = x ** 2
                X = torch.cat((X1, X2, X3), dim=1)
            elif self.order == 3:
                X3 = x ** 2
                X4 = x ** 3
                X = torch.cat((X1, X2, X3, X4), dim=1)
            elif self.order == 4:
                X3 = x ** 2
                X4 = x ** 3
                X5 = x ** 4
                X = torch.cat((X1, X2, X3, X4, X5), dim=1)

        loss = torch.abs(torch.sum(X*c, dim=1).view(-1, 1) - y).mean()

        return loss


class CalculatePCELoss(nn.Module):
    """The PCE regression loss.

    Args:
        x: the input
        y: the true value
        c: the coefficients of PCE
    Attributes:
        loss: the PCE loss result.
    """

    def __init__(self, order, order_mat, device):
        super(CalculatePCELoss, self).__init__()
        self.order_mat = order_mat
        self.order = order
        self.device = device
        self.criterion = nn.MSELoss()

    def forward(self, x, y, c, mu):
        y_pred = dp.cal_pce_pred_fun(x, c, mu, self.order_mat, self.order, self.device)
        # zero = torch.zeros_like(y_pred)
        # y_pred = torch.where(y_pred < 0, zero, y_pred)
        loss = torch.abs(y_pred - y).mean()
        # loss = (0.5 * (y_pred - y) ** 2).mean()
        # loss = self.criterion(y_pred, y)

        return loss


class CoefficientPCELoss(nn.Module):
    def __init__(self, order, order_mat, device):
        super(CoefficientPCELoss, self).__init__()
        self.order_mat = order_mat
        self.order = order
        self.device = device

    def forward(self, x_coeff, c, mu):
        c_mean = c.mean(dim=0)
        c_inter = c_mean[1:-1]
        y_pred = dp.cal_pce_pred_fun(x_coeff, c, mu, self.order_mat, self.order, self.device)
        loss = torch.abs(c_mean[0] - y_pred.mean()) + torch.abs(torch.sum(c_inter ** 2) - y_pred.var())

        return loss


# class CoefficientPCELoss_coeff_batch(nn.Module):
#     def __init__(self, order, order_mat, device):
#         super(CoefficientPCELoss_coeff_batch, self).__init__()
#         self.order_mat = order_mat
#         self.order = order
#         self.device = device

#     def forward(self, x_coeff, mu, net, coeff_batch_size=int(1e+03)):

#         num_coeff = x_coeff.size(0)
#         coeff_batch = math.ceil(num_coeff/coeff_batch_size)
#         x_coeff_all = torch.chunk(x_coeff, coeff_batch)
#         for j in range(len(x_coeff_all)):
#             x_coeff_batch = x_coeff_all[j]
#             x_coeff_batch = x_coeff_batch.to(self.device)
#             c_batch = net(x_coeff_batch)
#             y_batch = dp.cal_pce_pred_fun(x_coeff_batch, c_batch, mu, self.order_mat, self.order, self.device)
#             if j==0:
#                 c = c_batch
#                 y_pred = y_batch
#                 print('------------------')
#             else:
#                 c = torch.cat((c, c_batch), dim=0)
#                 y_pred = torch.cat((y_pred, y_batch), dim=0)

#         c_mean = c.mean(dim=0)
#         c_inter = c_mean[1:-1]
#         loss = torch.abs(c_mean[0] - y_pred.mean()) + torch.abs(torch.sum(c_inter ** 2) - y_pred.var())

#         return loss

class CoefficientPCELoss_coeff_batch(nn.Module):
    def __init__(self, order, order_mat, device):
        super(CoefficientPCELoss_coeff_batch, self).__init__()
        self.order_mat = order_mat
        self.order = order
        self.device = device

    def forward(self, c_coeff, y_coeff):
        c_mean = c_coeff.mean(dim=0)
        c_inter = c_mean[1:-1]
        loss = torch.abs(c_mean[0] - y_coeff.mean()) + torch.abs(torch.sum(c_inter ** 2) - y_coeff.var())

        return loss


class CalculatePCELoss_round(nn.Module):
    """The PCE regression loss.

    Args:
        x: the input
        y: the true value
        c: the coefficients of PCE
    Attributes:
        loss: the PCE loss result.
    """

    def __init__(self, order, order_mat, device, round_num=6):
        super(CalculatePCELoss_round, self).__init__()
        self.order_mat = order_mat
        self.order = order
        self.device = device
        self.criterion = nn.MSELoss()
        self.round_num = round_num

    def forward(self, x, y, c, mu):
        y_pred = dp.cal_pce_pred_fun(x, c, mu, self.order_mat, self.order, self.device)
        y_pred = y_pred.cpu()
        y_pred = y_pred.numpy()
        torch.from_numpy(np.round(y_pred, self.round_num))
        y_pred = y_pred.to(self.device)
        loss = torch.abs(y_pred - y).mean()

        return loss


class CoefficientPCELoss_round(nn.Module):
    def __init__(self, order, order_mat, device, round_num=6):
        super(CoefficientPCELoss_round, self).__init__()
        self.order_mat = order_mat
        self.order = order
        self.device = device
        self.round_num = round_num

    def forward(self, x_coeff, c, mu):
        c_mean = c.mean(dim=0)

        c0_mean= c_mean[0]
        c0_mean = c0_mean.numpy()
        c0_mean = torch.from_numpy(np.round(c0_mean, self.round_num))

        c_inter = c_mean[1:-1]
        c_2sum = torch.sum(c_inter ** 2)
        c_2sum = c_2sum.numpy()
        c_2sum = torch.from_numpy(np.round(c_2sum, self.round_num))
        y_pred = dp.cal_pce_pred_fun(x_coeff, c, mu, self.order_mat, self.order, self.device)
        y_pred_mean = y_pred.mean()
        y_pred_mean = y_pred_mean.numpy()
        y_pred_mean = torch.from_numpy(np.round(y_pred_mean, self.round_num))

        y_pred_var = y_pred.var()
        y_pred_var = y_pred_var.numpy()
        y_pred_var = torch.from_numpy(np.round(y_pred_var, self.round_num))
        loss = torch.abs(c0_mean - y_pred_mean) + torch.abs(c_2sum - y_pred_var)

        return loss