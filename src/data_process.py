import math
import torch
import numpy as np
from pyDOE import lhs
from scipy import stats
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


def probability_fun(x, threshold):
    y = torch.where(x <= threshold)[0]
    p = y.size(0) / x.size(0)
    return p

def multi_probability_fun(x, threshold):
    for i in range(threshold.size(1)):
        x_i = x[:, i:(i+1)]
        threshold_i = threshold[0, i]
        y = torch.where(x_i <= threshold_i)[0]
        p_i = y.size(0) / x.size(0)
        if i == 0:
            p = [p_i]
        else:
            p += [p_i]
    return torch.tensor(p)


def lognormal_pram(m0, cov):
    sigma = m0 * cov
    v = sigma ** 2
    mu = math.log(m0**2/math.sqrt(m0**2+v))
    std = math.sqrt(math.log(1+v/m0**2))

    return mu, std


def gumbel_pdf(x, mu=0, beta=1):
    z = (x - mu) / beta
    return torch.exp(-z - torch.exp(-z)) / beta


def lognormal_pdf(x, mu=0, sigma=1):
    z = -(torch.log(x) - mu) ** 2 / (2 * sigma ** 2)
    return 1 / (math.sqrt(2 * math.pi) * x * sigma) * torch.exp(z)


def normal_pdf(x, mu=0, sigma=1):
    z = -(x - mu) ** 2 / (2 * sigma ** 2)
    return 1 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(z)


def rosenbaltt(prob, dis_type, param):
    if dis_type == 'norm':
        X = stats.norm.ppf(prob, param[0], param[1])
    if dis_type == 'lognorm':
        X = stats.lognorm.ppf(prob, param[1], scale=math.exp(param[0]))
    elif dis_type == 'gamma':
        X = stats.gamma.ppf(prob, param[0], param[1])
    elif dis_type == 'exp':
        X = stats.expon.ppf(prob, param)
    elif dis_type == 'uniform':
        xi = stats.uniform.ppf(prob)
        X = xi * (param[1] - param[0]) + param[0]
    elif dis_type == 'gumbel':
        X = stats.gumbel_r.ppf(prob, param[0], param[1])
    elif dis_type == 'beta':
        X = stats.beta.ppf(prob, param[0], param[1])

    return X


def lhd_samping(num, dis_type, param):
    if dis_type == 'truncnorm':
        X_max = param[0] + 3 * param[1]
        X_min = param[0] - 3 * param[1]
        x_max = (X_max - param[0]) / param[1]
        x_min = (X_min - param[0]) / param[1]
        prob_max = stats.norm(loc=0, scale=1).cdf(x_max)
        prob_min = stats.norm(loc=0, scale=1).cdf(x_min)
        prob = lhs(1, samples=num, criterion='center')
        prob = prob * (prob_max - prob_min) + prob_min
        x = stats.norm(loc=0, scale=1).ppf(prob)
        X = stats.norm(loc=param[0], scale=param[1]).ppf(prob)
    else:
        prob = lhs(1, samples=num, criterion='center')
        x = stats.norm(loc=0, scale=1).ppf(prob)
        X = rosenbaltt(prob, dis_type, param)
    # X = np.random.normal(param[0], param[1], size=(num, 1))

    return x, X

def normal_moment(k):
    mu_k = torch.zeros((k+1,1))
    for kth in range(k+1):
        if kth == 0:
            mu_k[kth, 0] = 1
        elif (kth % 2) == 0:
            mu_k[kth, 0] = torch.tensor(list(range(kth))[1::2]).prod().item()
    return mu_k

def uniform_moment(k, lb=-math.sqrt(3), ub=math.sqrt(3)):
    mu_k = torch.zeros((k+1, 1))
    for i in range(k+1):
        if i == 0:
            mu_k[i, 0] = 1
        else:
            mu_k_temp = 0
            for j in range(i + 1):
                mu_k_temp += (lb ** j) * (ub ** (i - j))
            mu_k[i, 0] = 1 / (i+1) * mu_k_temp
    return mu_k

def interval_number_Max(X1, X2):
    temp =(X1[:, 1:2]-X2[:, 0:1]) / (X1[:, 1:2]-X1[:, 0:1] + X2[:, 1:2]-X2[:, 0:1])
    zero_mat = torch.zeros(X1.size(0), 1)
    a_x1_x2 = temp.where(temp>0, zero_mat)
    one_mat = torch.ones(X1.size(0), 1)
    size_prob = a_x1_x2.where(a_x1_x2 < one_mat, one_mat)

    max_X = X1.where(size_prob>0.5, X2)
    return max_X