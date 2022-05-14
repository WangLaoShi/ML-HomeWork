
from sklearn import metrics

metrics.mean_squared_error(y_test,y_predict)#MSE

metrics.mean_absolute_error(y_test,y_predict)#MAE

np.sqrt(metrics.mean_squared_error(y_test,y_predict))  # RMSE

metrics.r2_score(y_test,y_predict)#R2
