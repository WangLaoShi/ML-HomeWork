import numpy as np
import math

def MSE(real, predict):
    MSE_value = 0
    for i in range(len(real)):
        MSE_value += (real[i] - predict[i])**2

    MSE_value /= len(real)
    return MSE_value

def RMSE(real, predict):
    RMSE_value = 0
    for i in range(len(real)):
        RMSE_value += (real[i] - predict[i])**2

    RMSE_value /= len(real)
    RMSE_value = math.sqrt(RMSE_value)
    return RMSE_value

def MAE(real, predict):
    MAE_value = 0
    for i in range(len(real)):
        MAE_value += abs(real[i] - predict[i])

    MAE_value /= len(real)
    return MAE_value

def R(real, predict):
    pass
