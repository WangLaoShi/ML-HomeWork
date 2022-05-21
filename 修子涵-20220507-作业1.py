import numpy as np
from math import sqrt

def MSE(y, t):
    return 0.5 * np.sum((y - t)**2)

def MSE(y_test,y_predict):
    return np.sum((y_test - y_predict) ** 2) / len(y_test)

def RMSE(y_test,y_predict):
    mse = np.sum((y_test - y_predict) ** 2) / len(y_test)
    return sqrt(mse)

def MAE(y_test,y_predict):
    return np.sum(np.absolute(y_test - y_predict)) / len(y_test)

def R2(y_test,y_predict):
    mse = np.sum((y_test - y_predict) ** 2) / len(y_test)
    return 1-mse/ np.var(y_test)

