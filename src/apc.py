import torch
import numpy as np

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


def aPCE_poly_basis_fun(x, p_orders, order):
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
    
    return poly_basis

def orthogonal_basis(x, p_orders, order, order_mat):
    # Method-1
    index_column = order_mat.int().numpy().tolist()
    index_row = list(range(x.size(1)))
    poly_basis = aPCE_poly_basis_fun(x, p_orders, order)
    poly_basis = poly_basis.permute(1,2,0)
    poly_basis = poly_basis[:, index_row, index_column]
    phi_x = torch.prod(poly_basis, dim=2)

    # Method-2
    # poly_basis = aPCE_poly_basis_fun(x, mu, order, x.device)
    # phi_x = torch.zeros((x.size(0),  order_mat.size(0))).to(device=x.device)
    # for i in range(order_mat.size(0)):
    #     index =  order_mat[i]
    #     inter_basis = torch.ones((x.size(0),1)).to(x.device)
    #     for j in range(x.size(1)):
    #         inter_basis = inter_basis * poly_basis[index[j].int()][:, j:(j+1)]
    #     phi_x[:, i:(i+1)] = inter_basis

    return phi_x


def calculate_apc_coefficient(x, y, p_orders, order, order_mat):
    A = orthogonal_basis(x, p_orders, order_mat, order).numpy()
    G = y.numpy()
    coefficient = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T), G)

    return coefficient


def sparse_aPC(x, y, p_orders, order_mat, order):
    Phi_basis = orthogonal_basis(x, p_orders, order_mat, order).numpy()
    H = np.matmul(np.matmul(Phi_basis, np.linalg.inv(np.matmul(Phi_basis.T, Phi_basis))), Phi_basis.T)
    K = min(Phi_basis.shape[1], x.size(0)-1)
    Phi_series_number = list(range(Phi_basis.shape[1]))
    basis_now = []
    basis_prev = []
    residual_error_0 = y.numpy()
    for k in range(K):
        basis_related = np.absolute(np.matmul(Phi_basis.T, residual_error_0)) / np.linalg.norm(Phi_basis, ord=0, axis=1)
        basis_seclect_k = np.argmax(basis_related, axis=0)
        basis_now.append(basis_seclect_k)
        # residual_error_k = 