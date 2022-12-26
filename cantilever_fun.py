
def cantilever_fun(X):
    X_q = X[:, 0:1]
    X_F1 = X[:, 1:2]
    X_F2 = X[:, 2:3]
    X_E = X[:, 3:4]
    X_I = X[:, 4:5]
    X_L = X[:, 5:6]
    X_delta = X[:, 6:7]
    y = X_delta - ((X_q * X_L ** 4) / (8.0 * X_E * X_I) + \
            (5.0 * X_F1 * X_L ** 3) / (48.0 * X_E * X_I) + \
            (X_F2 * X_L ** 3) / (3.0 * X_E * X_I))
    return y