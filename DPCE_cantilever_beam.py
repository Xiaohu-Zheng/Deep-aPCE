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

sys.path.append('/mnt/jfs/zhengxiaohu/Deep_aPCE')
from src.models import DNN
import src.Deep_PCE as dPC
import src.data_process as dp
from src.pce_loss import CalculatePCELoss, CoefficientPCELoss
from cantilever_fun import cantilever_fun

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("Use CPU")
    device = torch.device('cpu')
else:
    print("Use GPU")
# device = torch.device('cpu')
print(os.getpid())

root_path = "/mnt/jfs/zhengxiaohu/Deep_aPCE/cantilever_beam/"
data_path = '/mnt/jfs/zhengxiaohu/Deep_aPCE/cantilever_beam/data/'

# 目标函数
lam = 40 # 3(90, 6000, temp=False) 3(40, 6000, temp=False)
model = 2
# train = True
train = False
# temp = True
temp = False
pc_dpce='apc'
order = 2
dim = 7
lr_c = 0.01
x_num = 70
x_coeff_batch_size = 20000
max_epoch = 7000
object_fun = f"DPCE_{order}order_cantilever_{x_num}data_model_{model}_lam{lam}"

# 参数
m_I = 5.3594e+08
sigma_I= 5.3594e+07

m_L = 3000
sigma_L = 150

m_q = 50.0
sigma_q = m_q * 0.15

m_F1 = 7.0e+04
sigma_F1 = m_F1 * 0.18

m_F2 = 1.0e+05
sigma_F2 = m_F2 * 0.20

m_E = 2.6e+05
sigma_E = m_E * 0.12

m_delta = 30.0
sigma_delta = m_delta * 0.30

mean = torch.tensor([[m_q, m_F1, m_F2, m_E, m_I, m_L, m_delta]])
std = torch.tensor([[sigma_q, sigma_F1, sigma_F2, sigma_E, sigma_I, 
                     sigma_L, sigma_delta]])

if train:
    # 准备有标签数据
    data = data_path + 'samples_{}.mat'.format(x_num)
    data = sio.loadmat(data)
    X_train = torch.from_numpy(data['X']).float()
    x_train = (X_train - mean) / std
    y_train = cantilever_fun(X_train)
    dataset = dp.TensorDataset(x_train, y_train)

    # 加载训练数据
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=50000,
                                            shuffle=True, num_workers=2)

    # 准备无标签数据
    num_coeff = int(1e+06)
    data_coeff = data_path + f'samples_{num_coeff}.mat'
    data_coeff = sio.loadmat(data_coeff)
    X_coeff = torch.from_numpy(data_coeff['X']).float()
    x_coeff = (X_coeff - mean) / std

    # 准备测试数据集
    num_test = 10000
    data_test = data_path + 'samples_{}.mat'.format(num_test)
    data_test = sio.loadmat(data_test)
    X_test = torch.from_numpy(data_test['X']).float()
    x_test = (X_test - mean) / std
    y_test = cantilever_fun(X_test)


# 计算k阶中心矩
mu_k_rooth = root_path + 'mu_ks/mu_k.mat'
mu_k_temp = sio.loadmat(mu_k_rooth)
mu_k = torch.from_numpy(mu_k_temp['mu_k']).float()
p_orders = dPC.all_orders_univariate_basis_coefficients(mu_k, order)
norm_factor_dpce = dPC.norm_factor_basis(order, mu_k, p_orders)

# 初始化模型
order_mat = dPC.order_mat_fun(dim, order)
num_c = order_mat.size(0)

if model == 0:
    # 模型-1：
    hiden_neurons = [64, 128, 64]
if model == 5:
    # 模型-1：
    hiden_neurons = [64, 128, 128]
elif model == 1:
    # 模型-2：
    hiden_neurons = [64, 128, 128, 64]
elif model == 2:
    # 模型-3：
    hiden_neurons = [64, 128, 256, 256, 256]
elif model == 3:
    # 模型-4：
    hiden_neurons = [64, 128, 256, 128, 64]
elif model == 4:
    # 模型-4：
    hiden_neurons = [64, 128, 256, 256, 256, 128]
net_c = DNN(dim, num_c, hiden_neurons)

# 定义优化器
net_c = net_c.to(device)
criterion_pce_loss = CalculatePCELoss(order, order_mat, pc_dpce, p_orders, norm_factor=norm_factor_dpce)
criterion_coeff_deep = CoefficientPCELoss(lam_mean=lam, lam_var=lam)
optimizer_c = optim.Adam(net_c.parameters(), lr=lr_c)

# 训练模型
test_acc_best = 1.0
if train:
    print("training on ", device)
    x_coeff = x_coeff.to(device)

    for epoch in range(max_epoch):  # loop over the dataset multiple times
        if epoch % 300 == 299:
            optimizer_c.param_groups[0]['lr'] = optimizer_c.param_groups[0]['lr'] * 0.7

        train_l_fea_sum, train_l_c_sum, train_acc_sum, batch_count, start = 0.0, 0.0, 0.0, 0, time.time()
        for i, data in enumerate(dataloader, 0):
            # 获得输入
            x, y = data
            x, y = x.to(device), y.to(device)
            
            # 梯度归零
            optimizer_c.zero_grad()

            # net_c forward + backward + optimize
            c_nn = net_c(x)
            loss_y_dpce = criterion_pce_loss(x, y, c_nn)
            
            c_coeff = net_c(x_coeff)
            y_dpce_coeff = dPC.deep_pce_fun(x_coeff, c_coeff, order, order_mat, 
                                            pc_dpce, p_orders, norm_factor=norm_factor_dpce)
            loss_dpce_coeff = criterion_coeff_deep(y_dpce_coeff, c_coeff)
            loss = loss_y_dpce + loss_dpce_coeff

            loss.backward()
            optimizer_c.step()
            train_l_c_sum += loss.item()
            batch_count += 1

        x_test, y_test = x_test.to(device), y_test.to(device)
        c_dpce_test = net_c(x_test)
        test_acc_dpce = criterion_pce_loss(x_test, y_test, c_dpce_test)

#         if test_acc_best > test_acc_dpce:
#             test_acc_best = test_acc_dpce
#             if not os.path.exists(root_path + 'trained_models'):
#                 os.makedirs(root_path + 'trained_models')
#             torch.save(net_c.state_dict(), root_path + f'trained_models/{object_fun}_{max_epoch}_temp.pth')

        print('epoch %d, loss_c=%.6f, test_acc_dpce=%.6f, lr_fea=%.6f, time %.4f sec'
            % (epoch + 1, train_l_c_sum / batch_count, test_acc_dpce, optimizer_c.param_groups[0]['lr'], time.time() - start))

#     print('order={}, dim={}, Trainning over!'.format(order, dim))

#     # 保存训练的模型
#     if not os.path.exists(root_path + 'trained_models'):
#         os.makedirs(root_path + 'trained_models')
#     torch.save(net_c.state_dict(), root_path + f'trained_models/{object_fun}_{max_epoch}.pth')

# 模型预测
if temp == True:
    net_c.load_state_dict(torch.load(root_path + f'trained_models/{object_fun}_{max_epoch}_temp.pth',
        map_location='cuda:0'))
else:
    net_c.load_state_dict(torch.load(root_path + f'trained_models/{object_fun}_{max_epoch}.pth',
    map_location='cuda:0'))

num_pre = int(1e7)
pre_batch_size = 5000000
num_batch = math.ceil(num_pre/pre_batch_size)

#-----------------------------------------------------------------------------------
# data_pred = data_path + 'samples_{}.mat'.format(num_pre)
# data_pred = sio.loadmat(data_pred)
# X_pred = torch.from_numpy(data_pred['X']).float()
# x_pred = (X_pred - mean) / std
# y_grd = cantilever_fun(X_pred)

data_pred = data_path + 'test_1e7.mat'
data_pred = sio.loadmat(data_pred)
x_pred = torch.from_numpy(data_pred['x']).float()
y_grd = torch.from_numpy(data_pred['y']).float()
#-------------------------------------------------------------------------------------

DPCE = dPC.Predict_Deep_PCE_regression(net_c, order, order_mat, pc_dpce, p_orders)
x_pred = x_pred.to(device)
start_pre = time.time()
Y_dpce_pre = DPCE.prediction(x_pred, num_batch, norm_factor_dpce)
print(time.time()-start_pre)

# 计算PCE模型的均值及方差
mean_MC = y_grd.mean()
std_MC = y_grd.std()
mean_dpce = Y_dpce_pre.mean()
std_dpce = Y_dpce_pre.std()
print('DPCE prediction mean({} data):'.format(x_num), mean_dpce.item(), '\n',
      'MC sampling mean:', mean_MC.item(), '\n',
      'DPCE prediction standard deviation({} data):'.format(x_num), std_dpce.item(), '\n',
      'MC sampling standard deviation:', std_MC.item())

mean_error = abs(Y_dpce_pre.mean() - y_grd.mean()) / y_grd.mean() * 100
# mean_error = abs(Y_dpce_pre.mean() - 18.0938) / 18.0938 * 100
print('Mean error:{}'.format(mean_error))

std_error_dpce = abs(Y_dpce_pre.std() - y_grd.std()) / y_grd.std() * 100
# std_error_dpce = abs(Y_dpce_pre.std() - 9.5318) / 9.5318 * 100
print('DPCE Stddard deviation error:{}'.format(std_error_dpce))

#------------------------------------------------------------------------------------------
# skewness
ske_mc = (((y_grd - mean_MC) / std_MC) ** 3).mean()
ske_dpce = (((Y_dpce_pre - mean_dpce) / std_dpce) ** 3).mean()
print('ske_mc:%.8f'%(ske_mc.item()))
print('ske_DPCE:%.8f'%(ske_dpce.item()))

ske_error_dpce = abs(ske_dpce - ske_mc) / ske_mc * 100
# ske_error_dpce = abs(ske_dpce - 0.7508) / 0.7508 * 100
print('DPCE Ske error:{}'.format(ske_error_dpce))

# kurtosis
kur_mc = (((y_grd - mean_MC) / std_MC) ** 4).mean()
kur_dpce = (((Y_dpce_pre - mean_dpce) / std_dpce) ** 4).mean()
print('kur_mc:%.8f'%(kur_mc.item()))
print('kur_DPCE:%.8f'%(kur_dpce.item()))

kur_error_dpce = abs(kur_dpce - kur_mc) / kur_mc * 100
# kur_error_dpce = abs(kur_dpce - 4.2615) / 4.2615 * 100
print('DPCE Kur error:{}'.format(kur_error_dpce))

#----------------------------------------------------------------------------------------------
rmse_dpce = torch.sqrt(((y_grd - Y_dpce_pre.cpu()) ** 2).mean())
print('RMSE_dpce:%.8f'%(rmse_dpce))

mae_dpce = (torch.absolute(y_grd - Y_dpce_pre.cpu())).mean()
print('MAE_dpce:%.8f'%(mae_dpce))

mre_dpce = (torch.absolute((y_grd - Y_dpce_pre.cpu()) / y_grd)).mean()
print('MRE_dpce:%.8f'%(mre_dpce))

R2_dpce = 1 - torch.sum( (y_grd - Y_dpce_pre.cpu()) ** 2) / torch.sum( (y_grd - mean_MC)**2 )
print('R2_dpce:%.8f'%(R2_dpce))

e3_dpce = torch.sqrt(torch.sum((Y_dpce_pre.cpu() - y_grd)**2) / torch.sum(y_grd**2))
print('e3_nn:%.8f'%(e3_dpce))

# Failure probability 185.7
threshold = 0
prob_mcs = dp.probability_fun(y_grd, threshold)
prob_dpce = dp.probability_fun(Y_dpce_pre.cpu(), threshold)
prob_dpce_error = abs(prob_dpce - prob_mcs) / prob_mcs * 100
print('prob_MCS:%.8f'%(prob_mcs))
print('prob_DPCE:%.8f'%(prob_dpce))
print('prob_dpce_error:%.4f'%(prob_dpce_error))

error_DPCE = torch.abs(y_grd - Y_dpce_pre.cpu())
file_name = f'errors/errors_DPCE_2order.mat'
path = root_path + file_name
data = {f"error_DPCE_2order": error_DPCE.numpy()}
sio.savemat(path, data)