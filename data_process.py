import math
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset


class TensorDataset(Dataset):
    # TensorDataset继承Dataset, 重载了__init__, __getitem__, __len__
    # 实现将一组Tensor数据对封装成Tensor数据集
    # 能够通过index得到数据集的数据，能够通过len，得到数据集大小

    def __init__(self, data_tensor, label_tensor):
        self.data_tensor = data_tensor
        self.label_tensor = label_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.label_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


def input_data_augment(x, tau=None):
    if tau is None:
        tau = torch.zeros(x.size(0), 1).fill_(0.5)
    elif type(tau) == float:
        tau = torch.zeros(x.size(0), 1).fill_(tau)

    x_tau = torch.cat((x, tau), 1)

    return x_tau


def prediction_for_PCE_regression(num, x, net, device, pattern, order):
    batch_size_now = x.size(0)
    all_prediction = torch.zeros(num, batch_size_now)
    net = net.to(device)
    x = x.to(device)
    with torch.no_grad():
        for i in range(num):
            c = net(x)
            outputs = pce_pred_fun(c, x, pattern, order, device)
            output_inter = outputs.view(batch_size_now)
            all_prediction[i, :] = output_inter
    std, mean = torch.std_mean(all_prediction, dim=0)

    return mean.view(-1, 1), std.view(-1, 1)


def k_center_moment(x, k):
    '计算x的前k阶中心矩'
    mu = torch.zeros((k, x.size(1)))
    for i in range(k):
        mu[i] = torch.mean(x ** (i + 1), dim=0)

    return mu


def order_mat_fun(n, order):
    '计算aPCE模型的各阶数组合'
    N = order + 1
    L = N ** n
    M = math.factorial(order + n) // (math.factorial(order) * math.factorial(n))
    order_mat = torch.zeros((M, n))
    row = 0
    for j in range(L):
        order_row = torch.zeros(n)
        for i in range(n):
            if i == n-1:
                order_row[i] = j % N
            else: 
                z = math.floor(j / (N ** (n - i - 1)))
                order_row[i] = z % N
        if torch.sum(order_row) <= order:
            order_mat[row] = order_row
            row +=1
    
    return order_mat


def aPCE_poly_basis_fun(x, mu, order, device):
    mu = mu.to(device)
    X1 = torch.ones(x.size()).to(device)
    X2 = x
    if order == 2:
        X3 = x ** 2 - mu[2] * x - 1
        poly_basis = torch.cat((X1, X2, X3), dim=0).view(order+1, x.size(0), -1)
    elif order == 3:
        X3 = x ** 2 - mu[2] * x - 1
        X4 = (1 - mu[2] + mu[2] ** 2) * x ** 3 + \
             (- mu[2] * mu[3] + mu[4] - mu[2]) * x ** 2 + \
             (-mu[2] * mu[4] + mu[2] ** 2 - mu[3] + mu[2] * mu[3]) * x + \
             (mu[2] ** 2 - mu[2] ** 3 + mu[2] * mu[3] - mu[4])
        poly_basis = torch.cat((X1, X2, X3, X4), dim=0).view(order+1, x.size(0), -1)
    elif order == 4:
        X3 = x ** 2 - mu[2] * x - 1
        X4 = (1 - mu[2] + mu[2] ** 2) * x ** 3 + \
             (- mu[2] * mu[3] + mu[4] - mu[2]) * x ** 2 + \
             (-mu[2] * mu[4] + mu[2] ** 2 - mu[3] + mu[2] * mu[3]) * x + \
             (mu[2] ** 2 - mu[2] ** 3 + mu[2] * mu[3] - mu[4])
        X5 = (- mu[2] ** 4 + mu[2] ** 2 * mu[5] - 2 * mu[4] * mu[3] + mu[3] ** 2) * x ** 4 + \
             (mu[2] ** 3 * mu[3] - mu[4] * mu[3] - mu[2] ** 2 * mu[6] + \
             mu[4] ** 2 + mu[3] * mu[5]) * x ** 3 + \
             (- mu[3] ** 2 * mu[2] ** 2 + mu[2] * mu[3] * mu[6] + mu[3] * mu[4] ** 2 - \
             mu[3] ** 2 * mu[5] + mu[2] ** 3 * mu[4] - mu[2] * mu[4] * mu[5]) * x ** 2 + \
             (- mu[4] ** 3 + mu[3] ** 2 * mu[2] ** 2 - mu[2] ** 2 * mu[3] * mu[4] + \
             mu[3] * mu[4] * mu[5] + mu[2] * mu[4] * mu[5] + mu[2] * mu[4] * mu[6] - \
             mu[2] * mu[3] * mu[6] - mu[2] ** 3 * mu[4] + mu[2] ** 3 * mu[5] - \
             mu[2] * mu[5] ** 2) * x + \
            (mu[2] ** 2 * mu[4] * mu[3] + mu[2] ** 3 * mu[6] - mu[2] ** 2 * mu[4] **2 - \
            2 * mu[2] ** 2 * mu[3] * mu[5] + 2 * mu[3] ** 2 * mu[4] - mu[3] ** 3)
        poly_basis = torch.cat((X1, X2, X3, X4, X5), dim=0).view(order+1, x.size(0), -1)

    return poly_basis


def cal_pce_pred_fun(x, c, mu, order_mat, order, device):
    poly_basis = aPCE_poly_basis_fun(x, mu, order, device)
    X_basis = torch.zeros((x.size(0), order_mat.size(0))).to(device)
    for i in range(order_mat.size(0)):
        index = order_mat[i]
        inter_basis = torch.ones((x.size(0),1)).to(device)
        for j in range(x.size(1)):
            inter_basis = inter_basis * poly_basis[index[j].int()][:, j:(j+1)]
        X_basis[:, i:(i+1)] = inter_basis

    y = torch.sum(X_basis * c, dim=1).view(-1, 1)

    return y


def prediction_for_cal_PCE_regression(num, x, net, mu, order_mat, order, device):
    batch_size_now = x.size(0)
    all_prediction = torch.zeros(num, batch_size_now)
    net = net.to(device)
    x = x.to(device)
    with torch.no_grad():
        for i in range(num):
            c = net(x)
            outputs = cal_pce_pred_fun(x, c, mu, order_mat, order, device)
            output_inter = outputs.view(batch_size_now)
            all_prediction[i, :] = output_inter
            if i == 0:
                c_all = c
            else:
                c_all = torch.cat((c_all, c), dim=0)
    std, mean = torch.std_mean(all_prediction, dim=0)
    c_mean = c_all.mean(dim=0)

    return mean.view(-1, 1), std.view(-1, 1), c_mean.view(1, -1)


def probability_fun(x, threshold):
    y = torch.where(x < threshold)[0]
    p = y.size(0) / x.size(0)
    return p


def lognormal_pram(m0, cov):
    sigma = m0 * cov
    v = sigma ** 2
    mu = math.log(m0**2/math.sqrt(m0**2+v))
    std = math.sqrt(math.log(1+v/m0**2))

    return mu, std