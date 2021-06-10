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
import matplotlib.font_manager as fm
from torch.utils.tensorboard import SummaryWriter

sys.path.append('/mnt/zhengxiaohu/nn_for_pce')
import data_process as dp
from pce_loss import CalculatePCELoss, CoefficientPCELoss

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("Use CPU")
    device = torch.device('cpu')
else:
    print("Use GPU")
# device = torch.device('cpu')

root_path = "/mnt/zhengxiaohu/nn_for_pce/satellite_frequency_example/"

# 目标函数
order = 2
dim = 6
max_epoch = 8000
object_fun = "Cal_PCE_coeff_{}order_satellite128_old_1e5".format(order)

mean = torch.tensor([2.69, 7.85, 4.43, 6.89e4, 2.0e5, 1.138e5])
std  = torch.tensor([0.00897, 0.02617, 0.01477, 2.29667e3, 2.0e3, 1.89667e3])

# 计算k阶中心矩
mu_x_k = torch.tensor([[ 0.0], [ 1.0], [ 0.0], [ 3.0], [0.0], [ 15.0], [0.0]])
mu_k = torch.cat((mu_x_k, mu_x_k, mu_x_k, mu_x_k, mu_x_k, mu_x_k), dim=1)


# 定义神经网络
class Net_c(nn.Module):
    def __init__(self, dim, num_c):
        super(Net_c, self).__init__()

        # self.fc1 = nn.Linear(dim, 64)
        # self.fc2 = nn.Linear(64, 128)
        # self.fc3 = nn.Linear(128, 256)
        # self.fc4 = nn.Linear(256, 128)
        # self.fc5 = nn.Linear(128, 64)
        # self.fc6 = nn.Linear(64, num_c)
        # self.actfun = nn.ReLU()

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


# 初始化模型
order_mat = dp.order_mat_fun(dim, order)
num_c = order_mat.size(0)
net_c = Net_c(dim, num_c)
net_c = net_c.to(device)


# 模型预测
# num_pre = int(10e6)
# pre_batch_size = 500000

# num_pre = int(10e7)
num_pre = 600
pre_batch_size = 500000
num_batch = math.ceil(num_pre/pre_batch_size)

#----------------------------------------------------------
data_pred = root_path + 'sat_data_old/data{}.mat'.format(num_pre)
data_pred = sio.loadmat(data_pred)
x_pred = torch.from_numpy(data_pred['x']).float()
x_plt = x_pred
x_pred = (x_pred - mean) / std
y_grd = torch.from_numpy(data_pred['f']).float()
#----------------------------------------------------------
# # # x_pred = torch.normal(0, 1, size=(num_pre, dim))
# # # output_x = open(root_path+'pred_data/x_pred', 'wb')
# # # pickle.dump(x_pred, output_x)

# pkl_file_x = open(root_path+'pred_data/x_pred', 'rb')
# x_pred = pickle.load(pkl_file_x)
#----------------------------------------------------------

x_pred = torch.chunk(x_pred, num_batch)
x_nums = [30, 40, 50, 60, 70]
j = 0
color = ['olive','blue','orange','green','red']
for x_num in x_nums:
    net_c.load_state_dict(torch.load(
    root_path + 'trained_models/{}_model_c_{}_{}.pth'.format(object_fun, x_num, max_epoch)))
    if x_num ==70:
        net_c.load_state_dict(torch.load(
        root_path + 'trained_models/{}_model_c_{}_{}.pth'.format(object_fun, x_num, 7000)))
    print('--------------------------------------------------------------')
    print(x_num) 
    for i in range(len(x_pred)):
        print(i,'/',len(x_pred))
        x_input = x_pred[i]
        x_input = x_input.to(device)
        y_nn, _, c_mean = dp.prediction_for_cal_PCE_regression(1, 
                            x_input, net_c, mu_k, order_mat, order, device)
        if i == 0:
            y_pred = y_nn
            c_all = c_mean
        else:
            y_pred = torch.cat((y_pred, y_nn), dim=0)
            c_all = torch.cat((c_all, c_mean), dim=0)

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
    # # print('PCE prediction mean({} data):'.format(x_num), mean_PCE.item(), '\n',
    # #     'PCE prediction standard deviation({} data):'.format(x_num), std_PCE.item())
    # print('PCE prediction mean({} data):'.format(x_num), mean_PCE.item(), '\n',
    #     'MC sampling mean:', mean_MC.item(), '\n',
    #     'PCE prediction standard deviation({} data):'.format(x_num), std_PCE.item(), '\n',
    #     'MC sampling standard deviation:', std_MC.item())

    # # 计算平均绝对偏差
    # mean_error = torch.mean(torch.abs(y_pred-y_grd))
    # print("Average absolute error:%.8f"%(mean_error.item()))

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

    # R2 = 1 - ((1 / num_pre) * torch.sum( (y_grd - y_pred) ** 2)) / ((1 / (num_pre-1)) * torch.sum( (y_grd - mean_MC)**2 ))
    # print('R2:%.8f'%(R2))

    # e3 = torch.sqrt(torch.sum((y_pred - y_grd)**2) / torch.sum(y_grd**2))
    # print('e3:%.8f'%(e3))

    Pr_fail = dp.probability_fun(y_pred, 81.0)
    print("Failue probability:%.10f"%(Pr_fail))
    #------------------------------------------------------------------------------------------

    # 画统计直方图
    my_font = fm.FontProperties(fname="/mnt/zhengxiaohu/times/times.ttf")
    # plt.figure(figsize=[6, 4], dpi=360)
    # # sns.distplot(y_grd, bins=50, norm_hist=True, hist = True, kde=False, color="green")
    # sns.distplot(y_pred, bins=50, norm_hist=True, hist = True, kde=True, color="red", 
    #          kde_kws={"color": "red", "lw": 1.5, 'linestyle':'-'},
    #          label='Deep aPCE')
    # # sns.distplot(y_grd, bins=50, norm_hist=True, hist = False, kde=True, color="green",
    # #          kde_kws={"color": "green", "lw": 1.5, 'linestyle':'--'},
    # #          label='MCS')
    # plt.grid(axis='y', alpha=0.1)
    # plt.xlabel('First-order frequency', fontproperties=my_font)
    # plt.ylabel('Density', fontproperties=my_font)
    # plt.xticks(fontproperties=my_font)
    # plt.yticks(fontproperties=my_font)
    # plt.legend(prop=my_font)
    # plt.savefig(
    #     root_path + 'Satellite_figs/Satellite_{}.pdf'.format(x_num), bbox_inches = 'tight', pad_inches=0.02)
    # plt.savefig(
    #     root_path + 'Satellite_figs/Satellite_{}.png'.format(x_num), bbox_inches = 'tight', pad_inches=0.02)

    # 画预测结果误差图
    error, _ = torch.sort(torch.abs(y_pred-y_grd), dim=0)
    print(error[-1])
    x_sort = torch.linspace(1, num_pre, num_pre).view(-1, 1)
    if x_num == x_nums[0]:
        plt.figure(figsize=[6, 4], dpi=360)
    plt.plot(x_sort, error, color=color[j], label='{} labeled training data'.format(x_num))
    j = j + 1
    plt.grid(axis='y', alpha=0.1)
    plt.xlabel('Serial number of input', fontproperties=my_font)
    plt.ylabel('Absolute error', fontproperties=my_font)
    plt.xticks(fontproperties=my_font)
    plt.yticks(fontproperties=my_font)
    plt.legend(prop=my_font)
    if x_num == x_nums[-1]:
        plt.savefig(
            root_path + 'Satellite_figs/Error_{}.pdf'.format(x_num), bbox_inches = 'tight', pad_inches=0.02)
        plt.savefig(
            root_path + 'Satellite_figs/Error_{}.png'.format(x_num), bbox_inches = 'tight', pad_inches=0.02)
