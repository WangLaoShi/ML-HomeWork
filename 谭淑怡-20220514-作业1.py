#MSE(mean-square error)均方误差
#RSME(Root Mean Square Error)均方根误差
#MAE(Mean Absolute Error)平均绝对误差
#R2(coefficient of determination)决定系数，也称判定系数或者拟合优度

#sklearn中的MSE，MAE，r2
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
#调用
MSE  = mean_squared_error(y_test,y_predict)
RMSE = sqrt(mean_squared_error(y_test, y_predict))
MAE  = mean_absolute_error(y_test,y_predict)
R2   = r2_score(y_test,y_predict)

#公式写法
MSE = np.sum((y_test - y_predict) ** 2) / len(y_test)
RMSE = sqrt(MSE)
MAE = np.sum(np.absolute(y_test - y_predict)) / len(y_test)
R2 = 1-MSE/ np.var(y_test)




