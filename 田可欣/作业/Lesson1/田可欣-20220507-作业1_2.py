# -*- coding: utf-8 -*-
# @Author: Sherry Tian
# @Date:   2022-05-13 17:31:25
# @Last Modified by:   sherry
# @Last Modified time: 2022-05-13 18:10:24

"""
损失函数

解决回归任务，实际就是找到一条线/超平面来拟合这些样本点，使他们之间的误差尽可能的小。
而不同的线/超平面（不同的参数值）对应着不同的误差，我们需要找到让误差最小的线/超平面。
那么怎么衡量误差呢？
我们需要引入损失函数（又称误差函数）
回归任务常用的损失函数：
1. 均方误差 MSE
2. 均方根误差 RMSE
3. 平均绝对误差 MAE
4. R-squared
5. 平均绝对百分⽐误差 MAPE
"""

import numpy as np


def norm(target, prediction):
	return np.array(target), np.array(prediction)

def MSE(target, prediction):
	target, prediction = norm(target, prediction)
	return np.mean((prediction - target)**2)

def RMSE(target, prediction):
	target, prediction = norm(target, prediction)
	return np.sqrt(np.mean((prediction - target)**2))

def MAE(target, prediction):
	target, prediction = norm(target, prediction)
	return np.mean(np.abs(prediction - target))

def RSquare(target, prediction):
	target, prediction = norm(target, prediction)
	return 1 - np.sum((target - prediction)**2) / np.sum((target - np.mean(target))**2)

def MAPE(target, prediction):
	target, prediction = norm(target, prediction)
	return np.mean(np.abs(prediction - target) / target) * 100


if __name__ == '__main__':

	target = [1.5, 2.1, 3.3, -4.7, -2.3, 0.75]
	prediction = [0.5, 1.5, 2.1, -2.2, 0.1, -0.5]

	print("MSE: %.4f"%MSE(target, prediction))
	print("RMSE: %.4f"%RMSE(target, prediction))
	print("MAE: %.4f"%MAE(target, prediction))
	print("R-Square: %.4f"%RSquare(target, prediction))
	print("MAPE: {:.4}%".format(MAPE(target, prediction)))