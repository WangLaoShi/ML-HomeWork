#coding
import numpy as np
201

#MSE

def mse_loss(y_true,y_pred):
    mse_loss = np.sum(np.power(y_true - y_pred, 2))/y_true.shape[0]
    return mse_loss

#RMSE

def rmse_loss(y_true,y_pred):
    rmse_loss = np.squrt(np.sum(np.power(y_true - y_pred, 2))/y_true.shape[0])
    return rmse_loss

#MAE
def mae_loss(y_true,y_pred):
    mae_loss = np.sum(np.abs(y_true - y_pred))/y_true.shape[0]
    return mae_loss

#R_squared

def r2(y_test, y_true):
    r2 =  1 - ((y_test - y_true) ** 2).sum() / ((y_true - np.mean(y_true)) ** 2).sum()
    return r2
