# -*- codeing = utf-8 -*-
# @Time : 2022/5/9 19:04
# @Author : liu
# @File : 刘奥鲜20220509损失函数.py
# @Software: PyCharm
import numpy as np
import math

def Mse(y_test,y_predict):
    mse = np.sum((y_test - y_predict) ** 2) / len(y_test)
    return mse

def Rmse(y_test,y_predict):
    rmse = math.sqrt(Mse(y_test,y_predict))
    return rmse

def Mae(y_test,y_predict):
    mae = np.sum(np.absolute(y_test - y_predict)) / len(y_test)
    return mae

def R2(y_test,y_predict):
    r2 = 1 - Mse(y_test, y_predict) / np.var(y_test)
    return r2

