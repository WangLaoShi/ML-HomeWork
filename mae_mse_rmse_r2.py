import numpy as np


# MSE
def mse(y_act, y_pred):
    mse = np.sum(np.power(y_act - y_pred, 2)) / y_act.shape[0]
    return mse


# RMSE
def rmse(y_act, y_pred):
    rmse = np.sqrt(np.sum(np.power(y_act - y_pred, 2)) / y_act.shape[0])
    return rmse


# MAE
def mae(y_act, y_pred):
    mae = np.sum(np.abs(y_act - y_pred)) / y_act.shape[0]
    return mae


# R_squared
def r2(y_act, y_pred):
    r2 = 1 - (np.sqrt(np.sum(np.power(y_act - y_pred, 2)) / y_act.shape[0])) / \
         (np.sqrt(np.sum(np.power(y_act - np.mean(y_act), 2)) / y_act.shape[0]))
    return r2


a = [1, 2, 3]
a = np.array(a)
b = [1, 2, 3]
c = [3, 2, 2]
b = np.array(b)
c = np.array(c)
print(mse(a, b), mse(a, c))
print(rmse(a, b), rmse(a, c))
print(mae(a, b), mae(a, c))
print(r2(a, b), r2(a, c))
