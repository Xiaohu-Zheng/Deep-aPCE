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
from torch.utils.tensorboard import SummaryWriter

sys.path.append('/mnt/zhengxiaohu/nn_for_pce')
import data_process as dp
from pce_loss import CalculatePCELoss, CoefficientPCELoss

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("Use CPU")
    device = torch.device('cpu')
else:
    print("Use GPU")
# device = torch.device('cpu')

root_path = "/mnt/zhengxiaohu/nn_for_pce/satellite_frequency_example/"

# 目标函数
num = 60
order = 2
dim = 6
lr_c = 0.05
max_epoch = 7000
object_fun = "PCE_{}_order_{}data".format(order, num)

# 计算k阶中心矩
mu_x_k = torch.tensor([[ 0.0], [ 1.0], [ 0.0], [ 3.0], [0.0], [ 15.0], [0.0]])
mu_k = torch.cat((mu_x_k, mu_x_k, mu_x_k, mu_x_k, mu_x_k, mu_x_k), dim=1)


# 定义神经网络
class Net_c(nn.Module):
    def __init__(self, dim, num_c):
        super(Net_c, self).__init__()

        self.fc1 = nn.Linear(dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 128)
        self.fc6 = nn.Linear(128, num_c)
        self.actfun = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.actfun(x)
        x = self.fc2(x)
        x = self.actfun(x)
        x = self.fc3(x)
        x = self.actfun(x)
        x = self.fc4(x)
        x = self.actfun(x)
        x = self.fc5(x)
        x = self.actfun(x)
        c = self.fc6(x)

        return c


# 计算PCE展开系数的个数
order_mat = dp.order_mat_fun(dim, order)
num_c = order_mat.size(0)

# 加载交叉验证数据
output_fold = open(root_path+'data_fold/data_folds{}'.format(num), 'rb')
dataloader = pickle.load(output_fold)

# 交叉训练模型
if not os.path.exists(root_path + 'trained_model_fold'):
        os.makedirs(root_path + 'trained_model_fold')

# 初始化模型
net_c = Net_c(dim, num_c)
net_c = net_c.to(device)

for k, data_fold_k in enumerate(dataloader, 0):
    print("The {}th cross-validation traning:".format(k))
    # 准备测试数据
    x_test, y_test = data_fold_k

    # 模型预测
    net_c.load_state_dict(torch.load(
        root_path + 'trained_model_fold/{}_model_c_{}_{}_fold.pth'.format(object_fun, max_epoch, k+1)))

    x_test = x_test.to(device)
    y_nn, _, c_mean = dp.prediction_for_cal_PCE_regression(1, 
                        x_test, net_c, mu_k, order_mat, order, device)
    if k == 0:
        error = torch.abs(y_nn - y_test)
    else:
        error_temp = torch.abs(y_nn - y_test)
        error = torch.cat((error, error_temp), dim=0)

    print("%.8f"%torch.mean(torch.abs(y_nn - y_test)))
error_average = torch.mean(error)
print("error_average:%.8f"%(error_average))