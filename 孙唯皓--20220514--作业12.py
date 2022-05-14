import numpy as np


# true：真实目标变量的数组
# prep：预测数组

def mse(true, pred):
    return np.sum((true - pred) ** 2)

def mae(true, pred):
    return np.sum(np.abs(true - pred))

def rmse(true, pred):
    return np.sqrt(sum((true - pred) ** 2) / len(true))

def R2(true, pred):
    return 1 - ((pred - true)**2).sum() / ((true - true.mean())**2).sum()
