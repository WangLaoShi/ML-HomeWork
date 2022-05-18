import numpy as np
from math import sqrt
def MSE(label,prediction):
    return np.sum((label-prediction)**2/len(label))

def RMSE(label,prediction):
    return sqrt(np.sum((label-prediction)**2/len(label)))

def MAE(label,prediction):
    return np.sum(np.absolute(label,prediction))/len(label)

def R2(label,prediction):
    return 1-np.sum((label-prediction)**2/len(label))/np.var(label)