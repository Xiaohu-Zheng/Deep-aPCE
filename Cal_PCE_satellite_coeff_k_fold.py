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

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("Use CPU")
    device = torch.device('cpu')
else:
    print("Use GPU")
# device = torch.device('cpu')

root_path = "/mnt/zhengxiaohu/nn_for_pce/satellite_frequency_example/"

# 目标函数
train = True
# train = False
num =  60
order = 2
dim = 6
lr_c = 0.05
max_epoch = 7000
object_fun = "PCE_{}_order_{}data".format(order, num)

mean = torch.tensor([2.69, 7.85, 4.43, 6.89e4, 2.0e5, 1.138e5])
std  = torch.tensor([0.00897, 0.02617, 0.01477, 2.29667e3, 2.0e3, 1.89667e3])

if train:
    # 准备无标签数据
    num_coeff = int(1e+05)
    x_coeff = torch.normal(0, 1, size=(num_coeff, dim))


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

if train:
    print("training on ", device)
    x_coeff = x_coeff.to(device)

    for k, data_fold_k in enumerate(dataloader, 0):
        print("The {}th cross-validation traning:".format(k))
        # 准备测试数据
        x_test, y_test = data_fold_k

        if k == 1 or k == 2 or k == 3:
            # -------------------------------------------第k折训练模型-------------------------------------------------------
            # 初始化模型
            net_c = Net_c(dim, num_c)

            # 定义优化器
            net_c = net_c.to(device)
            criterion_c = CalculatePCELoss(order, order_mat, device)
            criterion_coeff = CoefficientPCELoss(order, order_mat, device)
            optimizer_c = optim.Adam(net_c.parameters(), lr=lr_c)
            for epoch in range(max_epoch):
                if epoch % 300 == 299:
                    optimizer_c.param_groups[0]['lr'] = optimizer_c.param_groups[0]['lr'] * 0.7

                train_l_fea_sum, train_l_c_sum, train_acc_sum, batch_count, start = 0.0, 0.0, 0.0, 0, time.time()
                for i, data in enumerate(dataloader, 0):
                    if i != k:
                        # 获得输入
                        x, y = data
                        x, y = x.to(device), y.to(device)

                        # 梯度归零
                        optimizer_c.zero_grad()

                        # net_c forward + backward + optimize
                        c = net_c(x)
                        loss_c = criterion_c(x, y, c, mu_k)
                        if epoch >= 0:
                            c_coeff = net_c(x_coeff)
                            loss_coeff = criterion_coeff(x_coeff, c_coeff, mu_k)
                            loss = loss_c + loss_coeff
                        else:
                            loss = loss_c
                        loss.backward()
                        optimizer_c.step()
                        train_l_c_sum += loss.item()
                        batch_count += 1

                x_test, y_test = x_test.to(device), y_test.to(device)
                c_test = net_c(x_test)
                test_acc = criterion_c(x_test, y_test, c_test, mu_k)

                print("The {}th cross-validation".format(k+1), 'epoch %d, loss_c=%.6f, test_acc=%.6f, lr_fea=%.6f, time %.1f sec'
                    % (epoch + 1, train_l_c_sum / batch_count, test_acc, optimizer_c.param_groups[0]['lr'], time.time() - start))

            print('order={}, dim={}, Trainning over!'.format(order, dim))

            # 保存训练的模型
            torch.save(net_c.state_dict(), root_path +
                    'trained_model_fold/{}_model_c_{}_{}_fold.pth'.format(object_fun, max_epoch, k+1))
            #---------------------------------------------------------------------------------------------------------------------

# # 模型预测
# net_c.load_state_dict(torch.load(
#     root_path + 'trained_model_fold/{}_model_c_{}_{}.pth'.format(object_fun, x_num, max_epoch)))

# pre_batch_size = 800
# num_pre = 600
# num_batch = math.ceil(num_pre/pre_batch_size)

# #----------------------------------------------------------
# data_pred = root_path + 'sat_data/data{}.mat'.format(num_pre)
# data_pred = sio.loadmat(data_pred)
# x_pred = torch.from_numpy(data_pred['x']).float()
# x_pred = (x_pred - mean) / std
# y_grd = torch.from_numpy(data_pred['f']).float()
# x_pred = torch.chunk(x_pred, num_batch)
# #----------------------------------------------------------
 
# for i in range(len(x_pred)):
#     print(i,'/',len(x_pred))
#     x_input = x_pred[i]
#     x_input = x_input.to(device)
#     y_nn, _, c_mean = dp.prediction_for_cal_PCE_regression(1, 
#                            x_input, net_c, mu_k, order_mat, order, device)
#     if i == 0:
#         y_pred = y_nn
#         c_all = c_mean
#     else:
#         y_pred = torch.cat((y_pred, y_nn), dim=0)
#         c_all = torch.cat((c_all, c_mean), dim=0)

# # 计算PCE模型的均值及方差
# c_mean = c_all.mean(dim=0)
# mean_by_c = c_mean[0]
# c_mean_inter = c_mean[1:-1]
# std_by_c = ((c_mean_inter ** 2).sum()) ** 0.5
# print(mean_by_c, std_by_c)
# mean_MC = y_grd.mean()
# std_MC = y_grd.std()
# mean_PCE = y_pred.mean()
# std_PCE = y_pred.std()
# print('PCE prediction mean({} data):'.format(x_num), mean_PCE.item(), '\n',
#       'MC sampling mean:', mean_MC.item(), '\n',
#       'PCE prediction standard deviation({} data):'.format(x_num), std_PCE.item(), '\n',
#       'MC sampling standard deviation:', std_MC.item())

# #------------------------------------------------------------------------------------------
# # skewness
# ske_mc = (((y_grd - mean_MC) / std_MC) ** 3).mean()
# ske_PCE = (((y_pred - mean_PCE) / std_PCE) ** 3).mean()
# print('ske_mc:%.8f'%(ske_mc.item()))
# print('ske_PCE:%.8f'%(ske_PCE.item()))

# # kurtosis
# kur_mc = (((y_grd - mean_MC) / std_MC) ** 4).mean()
# kur_PCE = (((y_pred - mean_PCE) / std_PCE) ** 4).mean()
# print('kur_mc:%.8f'%(kur_mc.item()))
# print('kur_PCE:%.8f'%(kur_PCE.item()))

# R2 = 1 - torch.sum( (y_grd - y_pred) ** 2) / torch.sum( (y_grd - mean_MC)**2 )
# print('R2:%.8f'%(R2))

# e3 = torch.sqrt(torch.sum((y_pred - y_grd)**2) / torch.sum(y_grd**2))
# print('e3:%.8f'%(e3))
# #------------------------------------------------------------------------------------------
