import numpy as np


def mse_value(y_true, y_pred):
    """
    参数:
    y_true -- 测试集目标真实值
    y_pred -- 测试集目标预测值

    返回:
    mse -- MSE 评价指标
    """

    n = len(y_true)
    mse = sum(np.square(y_true - y_pred)) / n
    return mse

def rmse_value(y_true, y_pred):
    """
    参数:
    y_true -- 测试集目标真实值
    y_pred -- 测试集目标预测值

    返回:
    rmse -- RMSE 评价指标
    """

    n = len(y_true)
    mse = (sum(np.square(y_true - y_pred)) / n) ** 0.5
    return mse


def mae_value(y_true, y_pred):
    """
    参数:
    y_true -- 测试集目标真实值
    y_pred -- 测试集目标预测值

    返回:
    mae -- MAE 评价指标
    """
    n = len(y_true)
    mae = sum(np.abs(y_true - y_pred)) / n
    return mae

def R2_value(y_true, y_pred, y_ave):
    """
    参数:
    y_true -- 测试集目标真实值
    y_pred -- 测试集目标预测值
    y_ave -- 测试集目标平均值

    返回:
    r2 -- R_Square 评价指标
    """
    n = len(y_true)
    up = sum(np.square(y_pred - y_true))
    down = sum(np.square(y_ave - y_true))
    r2 = 1 - up / down
    return r2
