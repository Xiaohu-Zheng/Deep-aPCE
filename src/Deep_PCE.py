import math
import torch
import numpy as np
from itertools import combinations, permutations

def k_center_moment(x, k):
    '计算x的前k+1阶中心矩（包括0阶矩）'
    mu = torch.zeros((k+1, x.size(1)))
    for i in range(k+1):
        mu[i] = torch.mean(x ** i, dim=0)

    return mu

def order_mat_fun(dim, order):
    order_list = np.array([list(range(order+1))])
    for i in range(dim):
        order_mat_temp = np.repeat(order_list, repeats=(order + 1)**i, axis=1)
        order_mat_i = np.repeat(order_mat_temp, repeats=(order + 1)**(dim-i-1), axis=0).ravel().reshape(-1, 1)
        if i == 0:
            order_mat = order_mat_i
        else:
            order_mat = np.hstack((order_mat, order_mat_i))
    mat_sum = np.sum(order_mat, axis=1)
    index = np.argwhere(mat_sum <= order).ravel().tolist()
    order_mat = torch.from_numpy(order_mat[index, :])
    
    return order_mat

# def order_mat_fun(n, order):
#     '计算aPCE模型的各阶数组合'
#     N = order + 1
#     L = N ** n
#     M = math.factorial(order + n) // (math.factorial(order) * math.factorial(n))
#     order_mat = torch.zeros((M, n))
#     row = 0
#     for j in range(L):
#         order_row = torch.zeros(n)
#         for i in range(n):
#             if i == n-1:
#                 order_row[i] = j % N
#             else: 
#                 z = math.floor(j / (N ** (n - i - 1)))
#                 order_row[i] = z % N
#         if torch.sum(order_row) <= order:
#             order_mat[row] = order_row
#             row +=1
    
#     return order_mat

def univariate_basis_coefficients(mu, order_th):
    mu_mat_temp = np.eye(order_th+1)
    mat0_1 = mu_mat_temp[-1:, :].T
    for j in range(np.shape(mu)[1]):
        for i in range(order_th+1):
            mu_mat_temp[0:order_th, i] = mu[i:(i+order_th),j]
        mu_mat_inv = np.linalg.inv(mu_mat_temp)
        if j ==0:
            p_0_ks = np.matmul(mu_mat_inv, mat0_1)
        else:
            p_0_ks = np.hstack((p_0_ks, np.matmul(mu_mat_inv, mat0_1)))
    p_0_ks = torch.from_numpy(p_0_ks)

    return p_0_ks

def all_orders_univariate_basis_coefficients(mu, order):
    p_orders = []
    for order_th in range(order+1):
        p_order_th = univariate_basis_coefficients(mu, order_th)
        p_orders.append(p_order_th)
    
    return p_orders

def aPCE_poly_basis_fun(x, p_orders, order, norm_factor=None):
    for order_th in range(order+1):
        p_order_th = p_orders[order_th]
        p_order_th = p_order_th.to(device=x.device)
        poly_basis_th = 0
        for k in range(order_th+1):
            poly_basis_th += p_order_th[k] * x ** k
        if order_th == 0:
            poly_basis = poly_basis_th
        else:
            poly_basis = torch.cat((poly_basis, poly_basis_th), dim=0)

    poly_basis = poly_basis.view(order+1, x.size(0), -1)
    
    # import scipy.io as sio
    # root_path = "/mnt/jfs/zhengxiaohu/PCNN/Ishigami_PCNN/"
    # mu_ks_rooth = root_path + f'mu_ks/mu_k.mat'
    # mu_ks = sio.loadmat(mu_ks_rooth)
    # mu_k = torch.from_numpy(mu_ks['mu_k']).float()
    # norm_factor = torch.zeros((order+1, 1, x.size(1)))
    # for i in range(order+1):
    #     p_order_i = p_orders[i]
    #     norm_factor_temp = torch.zeros(x.size(1))
    #     for j in range(i+1):
    #         mu_k_2j = mu_k[2*j]
    #         norm_factor_temp += (p_order_i[j, :] ** 2) * mu_k_2j

    #     if i == 0:
    #         norm_factor[i, 0] = norm_factor_temp
    #     else:
    #         indice = list(range(i+1))
    #         indice_temp = list(combinations(indice, 2))
    #         for li in range(len(indice_temp)):
    #             indice_temp_li = indice_temp[li]
    #             sum_indice_li = sum(indice_temp_li)
    #             norm_factor_temp += 2 * p_order_i[indice_temp_li[0], :] * p_order_i[indice_temp_li[1], :] * mu_k[sum_indice_li, :]
    #         norm_factor[i, 0] = norm_factor_temp
    # norm_factor = norm_factor.to(device=x.device)
    # poly_basis = poly_basis / torch.sqrt(torch.absolute(norm_factor))

    if norm_factor is not None:
        norm_factor = norm_factor.to(device=x.device)
        poly_basis = poly_basis / norm_factor

    return poly_basis

def norm_factor_basis(order, mu_k, p_orders):
    norm_factor = torch.zeros((order+1, 1, mu_k.size(1)))
    for i in range(order+1):
        p_order_i = p_orders[i]
        norm_factor_temp = torch.zeros(mu_k.size(1))
        for j in range(i+1):
            mu_k_2j = mu_k[2*j]
            norm_factor_temp += (p_order_i[j, :] ** 2) * mu_k_2j

        if i == 0:
            norm_factor[i, 0] = norm_factor_temp
        else:
            indice = list(range(i+1))
            indice_temp = list(combinations(indice, 2))
            for li in range(len(indice_temp)):
                indice_temp_li = indice_temp[li]
                sum_indice_li = sum(indice_temp_li)
                norm_factor_temp += 2 * p_order_i[indice_temp_li[0], :] * p_order_i[indice_temp_li[1], :] * mu_k[sum_indice_li, :]
            norm_factor[i, 0] = norm_factor_temp
    norm_factor_basis = torch.sqrt(torch.absolute(norm_factor))

    return norm_factor_basis

def Hermite_orthogonal_poly_basis_fun(x, order):
    if order == 0:
        poly_basis = torch.ones((1, x.size(0),x.size(1))).to(device=x.device)
    elif order == 1:
        X1 = torch.ones((1, x.size(0),x.size(1))).to(device=x.device)
        X2 = x.view(1, x.size(0), -1)
        poly_basis = torch.cat((X1, X2), dim=0)
    elif order >=2:
        for order_th in range(order+1):
            if order_th == 0:
                poly_basis = torch.ones((1, x.size(0),x.size(1))).to(device=x.device)
            elif order_th == 1:
                poly_basis = torch.cat((poly_basis, x.view(1, x.size(0), -1)), dim=0)
            elif order_th >=2:
                poly_basis_th = x.view(1, x.size(0), -1) * poly_basis[-1:, :, :] - (order_th - 1) * poly_basis[-2:-1, :, :]
                poly_basis = torch.cat((poly_basis, poly_basis_th), dim=0)
    
    return poly_basis

def Legendre_orthogonal_poly_basis_fun(x, order):
    if order == 0:
        poly_basis = torch.ones((1, x.size(0),x.size(1))).to(device=x.device)
    elif order == 1:
        X1 = torch.ones((1, x.size(0),x.size(1))).to(device=x.device)
        X2 = x.view(1, x.size(0), -1)
        poly_basis = torch.cat((X1, X2), dim=0)
    elif order >=2:
        for order_th in range(order+1):
            if order_th == 0:
                poly_basis = torch.ones((1, x.size(0),x.size(1))).to(device=x.device)
            elif order_th == 1:
                poly_basis = torch.cat((poly_basis, x.view(1, x.size(0), -1)), dim=0)
            elif order_th >=2:
                poly_basis_th = (2 * (order - 1) + 1) / ((order - 1) + 1) * x.view(1, x.size(0), -1) * poly_basis[-1:, :, :] - (order - 1) / ((order - 1) + 1) * poly_basis[-2:-1, :, :]
                poly_basis = torch.cat((poly_basis, poly_basis_th), dim=0)
    
    return poly_basis

# def Hermite_orthogonal_poly_basis_fun(x, order):
#     X1 = torch.ones(x.size()).to(device=x.device)
#     X2 = x
#     if order == 2:
#         X3 = x ** 2 - 1
#         poly_basis = torch.cat((X1, X2, X3), dim=0).view(order+1, x.size(0), -1)
#     elif order == 3: 
#         X3 = x ** 2 - 1
#         X4 = x ** 3 - 3 * x
#         poly_basis = torch.cat((X1, X2, X3, X4), dim=0).view(order+1, x.size(0), -1)
#     elif order == 4:
#         X3 = x ** 2 - 1
#         X4 = x ** 3 - 3 * x
#         X5 = x ** 4 - 6 * x ** 2 + 3
#         poly_basis = torch.cat((X1, X2, X3, X4, X5), dim=0).view(order+1, x.size(0), -1)
#     elif order == 5:
#         X3 = x ** 2 - 1
#         X4 = x ** 3 - 3 * x
#         X5 = x ** 4 - 6 * x ** 2 + 3
#         X6 = x ** 5 - 10 * x ** 3 + 15 * x
#         poly_basis = torch.cat((X1, X2, X3, X4, X5, X6), dim=0).view(order+1, x.size(0), -1)
#     return poly_basis


def orthogonal_basis(x, order, order_mat, pc='apc', p_orders=None, x_batch=int(1e6), sparse_basis=None, norm_factor=None):
    num_x = x.size(0)
    x_batch = math.ceil(num_x / x_batch)
    x_groups = torch.chunk(x, x_batch)
    for i in range(len(x_groups)):
        # print(i)
        x_i = x_groups[i]
        if pc == 'apc':
            poly_basis = aPCE_poly_basis_fun(x_i, p_orders, order, norm_factor)
        elif pc == 'hpc':
            poly_basis = Hermite_orthogonal_poly_basis_fun(x_i, order)
        elif pc == 'legen-gpc':
            poly_basis = Legendre_orthogonal_poly_basis_fun(x_i, order)
        # Method-1
        index_column = order_mat.int().numpy()
        if sparse_basis is not None:
            index_column = index_column[sparse_basis, :]
        index_column = index_column.tolist()
        index_row = list(range(x_i.size(1)))
        poly_basis = poly_basis.permute(1,2,0)
        poly_basis = poly_basis[:, index_row, index_column]
        phi_x_i = torch.prod(poly_basis, dim=2)
        if i == 0:
            phi_x = phi_x_i
        else:
            phi_x = torch.cat((phi_x, phi_x_i), dim=0) 

    # # Method-2
    # if sparse_basis is not None:
    #     order_mat = order_mat[sparse_basis, :]
    # phi_x = torch.zeros((x.size(0),  order_mat.size(0))).to(device=x.device)
    # for i in range(order_mat.size(0)):
    #     index =  order_mat[i]
    #     inter_basis = torch.ones((x.size(0),1)).to(device=x.device)
    #     for j in range(x.size(1)):
    #         inter_basis = inter_basis * poly_basis[index[j].int()][:, j:(j+1)]
    #     phi_x[:, i:(i+1)] = inter_basis

    return phi_x


def calculate_pce_coefficient(x, y, order, order_mat, pc='apc', p_orders=None, sparse=False, norm_factor=None):
    basis_best = None
    A = orthogonal_basis(x, order, order_mat, pc, p_orders, sparse_basis=None, norm_factor=norm_factor).numpy()
    G = y.numpy()
    if sparse:
        basis_best = sparse_aPC(x, y, order, order_mat, pc, p_orders, norm_factor)
        A = A[:, basis_best]
        coefficient = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T), G)
        return coefficient, basis_best
    
    else:
        coefficient = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T), G)
        return coefficient


def sparse_aPC(x, y, order, order_mat, pc, p_orders, norm_factor):
    var_y = np.var(y.numpy())
    Phi_basis = orthogonal_basis(x, order, order_mat, pc, p_orders, sparse_basis=None, norm_factor=norm_factor).numpy()

    K = min(Phi_basis.shape[1], x.size(0)-1)
    basis_now = []
    residual_error = y.numpy()
    for k in range(K):
        basis_related = np.squeeze(np.absolute(np.matmul(Phi_basis.T, residual_error))) / np.linalg.norm(Phi_basis, ord=0, axis=0)
        if k>=1:
            basis_related[basis_now] = 0
        basis_seclect_k = np.argmax(basis_related)
        basis_now.append(basis_seclect_k)
        A = Phi_basis[:, basis_now]
        coefficient_temp = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T), y.numpy())
        y_pre_now = np.sum(A * coefficient_temp.reshape(1, -1), axis=1).reshape(-1, 1)
        residual_error = y.numpy() - y_pre_now
        H = np.matmul(np.matmul(A, np.linalg.inv(np.matmul(A.T, A))), A.T)
        h = 1 - H.diagonal().reshape(-1, 1)
        e_loo_k = np.mean(((y.numpy() - y_pre_now) / h) ** 2) / var_y
        if k == 0:
            e_loo_best = e_loo_k
            basis_best = basis_now
        else:
            if e_loo_k < e_loo_best:
                basis_best = basis_now.copy()
                e_loo_best = e_loo_k
    basis_best = sorted(basis_best)

    return basis_best

        
def deep_pce_fun(x, c, order, order_mat, pc='apc', p_orders=None, sparse_basis=None, norm_factor=None):
    x = x.to(device=c.device)
    X_basis = orthogonal_basis(x, order, order_mat, pc, p_orders, x.size(0), sparse_basis, norm_factor)
    # file_name = f'DPCE_coefficient/basis_{order}order.mat'
    # path = '/mnt/jfs/zhengxiaohu/Deep_aPCE/oscillator/' + file_name
    # data = {f"basis": X_basis[0].cpu().numpy()}
    # import scipy.io as sio
    # sio.savemat(path, data)
    y = torch.sum(X_basis * c, dim=1).view(-1, 1)

    return y


def prediction_for_Deep_PCE_regression(num, x, net, order, order_mat, pc, p_orders, device, sparse_basis, norm_factor):
    batch_size_now = x.size(0)
    all_prediction = torch.zeros(num, batch_size_now)
    net = net.to(device)
    x = x.to(device)
    with torch.no_grad():
        for i in range(num):
            c = net(x)
            outputs = deep_pce_fun(x, c, order, order_mat, pc, p_orders, sparse_basis, norm_factor)
            output_inter = outputs.view(batch_size_now)
            all_prediction[i, :] = output_inter
            if i == 0:
                c_all = c
            else:
                c_all = torch.cat((c_all, c), dim=0)
    std, mean = torch.std_mean(all_prediction, dim=0)
    c_mean = c_all.mean(dim=0)

    return mean.view(-1, 1), std.view(-1, 1), c_mean.view(1, -1)


class Predict_Deep_PCE_regression():
    def __init__(self, net, order, order_mat, pc='apc', p_orders=None):
        super(Predict_Deep_PCE_regression, self).__init__()
        self.net = net
        self.order_mat = order_mat
        self.order = order
        self.p_orders = p_orders
        self.pc = pc

    def prediction(self, x_pred, num_batch, norm_factor=None, sparse_basis=None):
        x_pred = torch.chunk(x_pred, num_batch)
        for i in range(len(x_pred)):
            # print(i,'/',len(x_pred))
            x_input = x_pred[i]
            with torch.no_grad():
                c_nn_pre = self.net(x_input)
                y_dpce_pre = deep_pce_fun(x_input, c_nn_pre, self.order, self.order_mat, 
                                          self.pc, self.p_orders, sparse_basis, norm_factor)
            if i == 0:
                Y_dpce_pre = y_dpce_pre
            else:
                Y_dpce_pre = torch.cat((Y_dpce_pre, y_dpce_pre), dim=0)

        return Y_dpce_pre


class PCE():
    def __init__(self, c, order, order_mat, pc='apc', p_orders=None, num_batch=1):
        super(PCE, self).__init__()
        self.c = c
        self.order_mat = order_mat
        self.order = order
        self.device = c.device
        self.p_orders = p_orders
        self.pc = pc
        self.num_batch = num_batch

    # def forward(self, x):
    #     if self.pc == 'apc':
    #         poly_basis = aPCE_poly_basis_fun(x, self.p_orders, self.order)
    #     elif self.pc == 'hpc':
    #         poly_basis = Hermite_orthogonal_poly_basis_fun(x, self.order)
    #     index_column = self.order_mat.int().numpy().tolist()
    #     index_row = list(range(x.size(1)))
    #     poly_basis = poly_basis.permute(1,2,0)
    #     poly_basis = poly_basis[:, index_row, index_column]
    #     X_basis = torch.prod(poly_basis, dim=2)
    #     coeff = self.c.repeat(x.size(0),1).to(self.device)
    #     y = torch.sum(X_basis *  coeff, dim=1).view(-1, 1)

    def forward(self, x, sparse_basis=None, norm_factor=None):
        x = torch.chunk(x, self.num_batch)
        for i in range(len(x)):
            # print(i,'/',len(x))
            x_input = x[i]
            if self.pc == 'apc':
                poly_basis = aPCE_poly_basis_fun(x_input, self.p_orders, self.order, norm_factor)
            elif self.pc == 'hpc':
                poly_basis = Hermite_orthogonal_poly_basis_fun(x_input, self.order)
            elif self.pc == 'legen-gpc':
                poly_basis = Legendre_orthogonal_poly_basis_fun(x_input, self.order)
            index_column = self.order_mat.int().numpy()
            if sparse_basis is not None:
                index_column = index_column[sparse_basis, :]
            index_column = index_column.tolist()
            index_row = list(range(x_input.size(1)))
            poly_basis = poly_basis.permute(1,2,0)
            poly_basis = poly_basis[:, index_row, index_column]
            X_basis = torch.prod(poly_basis, dim=2)
            # if sparse_basis is not None:
            #     X_basis = X_basis[:, sparse_basis]
            coeff = self.c.repeat(x_input.size(0),1).to(self.device)
            y_i = torch.sum(X_basis *  coeff, dim=1).view(-1, 1)
            if i == 0:
                y = y_i
            else:
                y = torch.cat((y, y_i), dim=0)

        return y