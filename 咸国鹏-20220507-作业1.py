import numpy as np
import math
#均方误差
def MSE(y,y_pre):

    '''
    向量化代替循环;
    :y:实际值;
    :y_pre:预测值;
    :retur MSE;
    '''

    reduce = y - y_pre
    squ = reduce.dot(reduce)
    mse = np.sum(squ) / np.size(squ)

    return mse

#均方根误差
def RMSE(y,y_pre):

    '''
    向量化代替循环;
    :y:实际值;
    :y_pre:预测值;
    :retur RMSE;
    '''

    reduce = y - y_pre
    squ = reduce.dot(reduce)
    rmse = math.sqrt( np.sum(squ) / np.size(squ) )

    return rmse

#平均绝对误差
def MAE(y,y_pre):

    '''
    向量化代替循环;
    :y:实际值;
    :y_pre:预测值;
    :retur MAE;
    '''

    reduce = abs(y - y_pre)
    mae = np.sum(reduce) / np.size(reduce)

    return mae

#R方
def R_squared(y,y_pre):

    '''
    向量化代替循环;
    :y:实际值;
    :y_pre:预测值;
    :return:r_squared:
    '''
    y_mean = np.sum(y) / np.size(y)
    SSR = np.sum( np.square(y_pre - y_mean) )
    SST = np.sum( np.square(y - y_mean) )
    r_squared = SSR / SST

    return r_squared

if __name__ == '__main__':
    #if data is datafram
    #transform datafram to ndarray

    #test date
    y_true = np.array([2,4,6,8])
    y_pre = np.array([2,2,5,9])
    #print result
    print(MSE(y_true,y_pre))
    print(RMSE(y_true, y_pre))
    print(MAE(y_true, y_pre))
    print(R_squared(y_true, y_pre))