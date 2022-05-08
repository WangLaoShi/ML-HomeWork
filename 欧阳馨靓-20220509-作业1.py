from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import xlrd

data_url = r"C:\Users\25772\Desktop\线性回归-代码实现\data\fashion-mnist-demo.csv"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
bos_house_data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
bos_house_target = raw_df.values[1::2, 2]

x_train,x_test,y_train,y_test = train_test_split(bos_house_data,bos_house_target,random_state=41)
forest_reg = RandomForestRegressor(random_state=41)
forest_reg.fit(x_train,y_train)
y_predict = forest_reg.predict(x_test)

from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
print("mean_absolute_error:", mean_absolute_error(y_test, y_predict))
print("mean_squared_error:", mean_squared_error(y_test, y_predict))
print("rmse:", sqrt(mean_squared_error(y_test, y_predict)))
print("r2 score:", r2_score(y_test, y_predict))

mse = np.sum((y_test - y_predict) ** 2) / len(y_test)
rmse = sqrt(mse)
mae = np.sum(np.absolute(y_test - y_predict)) / len(y_test)
r2 = 1-mse/ np.var(y_test)#均方误差/方差
print(" mae:",mae,"mse:",mse," rmse:",rmse," r2:",r2)