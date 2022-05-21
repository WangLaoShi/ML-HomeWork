# MSE, MAE, R2, RMSE法一
from sklearn.metrics import mean_squared_error #MSE
from sklearn.metrics import mean_absolute_error #MAE
from sklearn.metrics import r2_score#R 2
import numpy as np

y_test = [1.5, 2.1, 3.3, -4.7, -2.3, 0.75]
y_predict = [0.5, 1.5, 2.1, -2.2, 0.1, -0.5]
#调用
MSE = mean_squared_error(y_test,y_predict)
MAE = mean_absolute_error(y_test,y_predict)
RMSE = np.sqrt(mean_squared_error(y_test,y_predict))  # RMSE
R2 = r2_score(y_test,y_predict)
print(MSE,MAE,RMSE,R2)

'''
# MSE, MAE, R2, RMSE法二
from sklearn import metrics

metrics.mean_squared_error(y_test,y_predict)
metrics.mean_absolute_error(y_test,y_predict)
np.sqrt(metrics.mean_squared_error(y_test,y_predict))  # RMSE
metrics.r2_score(y_test,y_predict)
'''