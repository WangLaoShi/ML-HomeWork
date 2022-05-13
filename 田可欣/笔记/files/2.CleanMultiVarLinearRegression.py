import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
import sys
print(sys.path)
# plotly.offline.init_notebook_mode()
from LinearRegression import LinearRegression


data = pd.read_csv('../data/world-happiness-report-2017.csv')

## 分离训练集和测试集，8：2

train_data = data.sample(frac=0.8)
test_data = data.drop(train_data.index)

x_1 = 'Economy..GDP.per.Capita.'
x_2 = 'Freedom'
y   = 'Happiness.Score'

x_train = train_data[[x_1, x_2]].values
y_train = train_data[[y]].values

x_test = test_data[[x_1, x_2]].values
y_test = test_data[[y]].values

# 作业 https://www.gingerdoc.com/plotly/plotly_quick_guide
num_iterations = 500 # 要运行的迭代次数
learning_rate = 0.01 # 学习效率
polynomial_degree = 0 # 多项式
sinusoid_degree = 0 # sin 角度

# 线性回归
linear_regression = LinearRegression(x_train, y_train, polynomial_degree, sinusoid_degree)

(theta, cost_history) = linear_regression.train(
    learning_rate,
    num_iterations
)

# 使用了梯度下降
print('开始损失', cost_history[0])
print('结束损失', cost_history[-1])
