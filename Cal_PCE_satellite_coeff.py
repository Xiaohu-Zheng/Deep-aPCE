import os
import sys
import time
import math
import torch
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

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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
order = 2
dim = 6
lr_c = 0.05
x_num = 60
max_epoch = 8002
object_fun = "Cal_PCE_coeff_{}order_satellite128_old_1e5".format(order)

mean = torch.tensor([2.69, 7.85, 4.43, 6.89e4, 2.0e5, 1.138e5])
std  = torch.tensor([0.00897, 0.02617, 0.01477, 2.29667e3, 2.0e3, 1.89667e3])

if train:
    # 准备有标签数据(rho1, rho2, rho3, E1, E2, E3)
    data = root_path + 'sat_data_old/data{}.mat'.format(x_num)
    data = sio.loadmat(data)
    x_train = torch.from_numpy(data['x']).float()
    x_train = (x_train - mean) / std
    y_train = torch.from_numpy(data['f']).float()
    dataset = dp.TensorDataset(x_train, y_train)

    # 加载训练数据
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1000,
                                            shuffle=True, num_workers=2)

    # 准备无标签数据
    num_coeff = int(1e+05)
    x_coeff = torch.normal(0, 1, size=(num_coeff, dim))

    # 准备测试数据
    num_test = 600
    data_test = root_path + 'sat_data_old/data{}.mat'.format(num_test)
    data_test = sio.loadmat(data_test)
    x_test = torch.from_numpy(data_test['x']).float()
    x_test = (x_test - mean) / std
    y_test = torch.from_numpy(data_test['f']).float()


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
        

        # self.fc1 = nn.Linear(dim, 64)
        # self.fc2 = nn.Linear(64, 128)
        # self.fc3 = nn.Linear(128, 256)
        # self.fc4 = nn.Linear(256, 512)
        # self.fc5 = nn.Linear(512, 512)
        # self.fc6 = nn.Linear(512, num_c)

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

# 定义优化器
net_c = net_c.to(device)
criterion_c = CalculatePCELoss(order, order_mat, device)
criterion_coeff = CoefficientPCELoss(order, order_mat, device)
optimizer_c = optim.Adam(net_c.parameters(), lr=lr_c)

# 训练模型
if not os.path.exists(root_path + 'trained_models'):
        os.makedirs(root_path + 'trained_models')
test_acc0 = 0.007
if train:
    print("training on ", device)
    x_coeff = x_coeff.to(device)
    # writer = SummaryWriter(
    #     root_path + 'runs/run_{}{}'.format(object_fun, max_epoch))

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
        if test_acc < test_acc0:
            test_acc0 = test_acc
            torch.save(net_c.state_dict(), root_path +
            'trained_models/{}_model_c_{}_{}.pth'.format(object_fun, x_num, max_epoch))

        # for name, param in net.named_parameters():  # 读入net的参数
        #     writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch + 1)
        # writer.add_scalar('loss', train_l_sum / batch_count, epoch + 1)

        print('epoch %d, loss_c=%.6f, test_acc=%.6f, lr_fea=%.6f, time %.1f sec'
            % (epoch + 1, train_l_c_sum / batch_count, test_acc, optimizer_c.param_groups[0]['lr'], time.time() - start))

    print('order={}, dim={}, Trainning over!'.format(order, dim))

    # 保存最终的训练模型
    torch.save(net_c.state_dict(), root_path +
            'trained_models/{}_model_c_{}_{}_final.pth'.format(object_fun, x_num, max_epoch))

# 模型预测
net_c.load_state_dict(torch.load(
    root_path + 'trained_models/{}_model_c_{}_{}.pth'.format(object_fun, x_num, max_epoch)))

pre_batch_size = 800
num_pre = 600
num_batch = math.ceil(num_pre/pre_batch_size)

#----------------------------------------------------------
data_pred = root_path + 'sat_data_old/data{}.mat'.format(num_pre)
data_pred = sio.loadmat(data_pred)
x_pred = torch.from_numpy(data_pred['x']).float()
x_pred = (x_pred - mean) / std
y_grd = torch.from_numpy(data_pred['f']).float()
x_pred = torch.chunk(x_pred, num_batch)
#----------------------------------------------------------
 
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

# 计算PCE模型的均值及方差
c_mean = c_all.mean(dim=0)
mean_by_c = c_mean[0]
c_mean_inter = c_mean[1:-1]
std_by_c = ((c_mean_inter ** 2).sum()) ** 0.5
print(mean_by_c, std_by_c)
mean_MC = y_grd.mean()
std_MC = y_grd.std()
mean_PCE = y_pred.mean()
std_PCE = y_pred.std()
print('PCE prediction mean({} data):'.format(x_num), mean_PCE.item(), '\n',
      'MC sampling mean:', mean_MC.item(), '\n',
      'PCE prediction standard deviation({} data):'.format(x_num), std_PCE.item(), '\n',
      'MC sampling standard deviation:', std_MC.item())

#------------------------------------------------------------------------------------------
# skewness
ske_mc = (((y_grd - mean_MC) / std_MC) ** 3).mean()
ske_PCE = (((y_pred - mean_PCE) / std_PCE) ** 3).mean()
print('ske_mc:%.8f'%(ske_mc.item()))
print('ske_PCE:%.8f'%(ske_PCE.item()))

# kurtosis
kur_mc = (((y_grd - mean_MC) / std_MC) ** 4).mean()
kur_PCE = (((y_pred - mean_PCE) / std_PCE) ** 4).mean()
print('kur_mc:%.8f'%(kur_mc.item()))
print('kur_PCE:%.8f'%(kur_PCE.item()))

R2 = 1 - torch.sum( (y_grd - y_pred) ** 2) / torch.sum( (y_grd - mean_MC)**2 )
print('R2:%.8f'%(R2))

e3 = torch.sqrt(torch.sum((y_pred - y_grd)**2) / torch.sum(y_grd**2))
print('e3:%.8f'%(e3))
#------------------------------------------------------------------------------------------

# 作图
x_sort = torch.linspace(1, num_pre, num_pre).view(-1, 1)
if not os.path.exists(root_path + 'figs/{}_{}_{}'.format(object_fun, x_num, max_epoch)):
    os.makedirs(root_path + 'figs/{}_{}_{}'.format(object_fun, x_num, max_epoch))

plt.figure(figsize=[12, 8], dpi=360)
plt.plot(x_sort, y_grd, '-', color='red',
         label='True function', alpha=0.5, markersize=5)
plt.plot(x_sort, y_pred, '-', color='black',
         label='Prediction by DL and PCE', alpha=0.5, markersize=5)
plt.title('Model based on DL and PCE by {} data'.format(x_num))
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend()
plt.grid()
plt.savefig(
    root_path + 'figs/{}_{}_{}/plot_pred.png'.format(object_fun, x_num, max_epoch))
plt.show()

# 画统计直方图
plt.figure(figsize=[12, 8], dpi=360)
sns.distplot(y_pred, bins=50, color='red', kde=True,
             label='Predictive distribution')
sns.distplot(y_grd, bins=50, color='green',
             kde=True, label='True distribution')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.title('Model based on DL and PCE by {} data'.format(x_num))
plt.savefig(
    root_path + 'figs/{}_{}_{}/historm_pred.png'.format(object_fun, x_num, max_epoch))
