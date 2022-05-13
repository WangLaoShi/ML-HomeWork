import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
import sys
print(sys.path)
# plotly.offline.init_notebook_mode()
from LinearRegression import LinearRegression

'''
一个简单的多元线性回归实现
'''

## 世界幸福报告

## 幸福感是根据经济生产、社会支持等因素来得分的。

##《世界幸福报告》是对全球幸福状况的里程碑式调查。第一份报告于2012年发布，2013年第二份，2015年第三份，2016年更新第四份。《2017世界幸福》
# 以其幸福水平排名155个国家，3月20日在联合国举行的庆祝国际幸福日活动上发布。报告继续得到全球的承认，因为各国政府、组织和民间社会越来越多地利用
# 幸福指数来通知其决策决定。跨领域的主要专家——经济学、心理学、调查分析、国家统计、卫生、公共政策等——描述如何有效地利用幸福感衡量来评估国家的进
# 展。这些报告回顾了当今世界上的幸福状况，并展示了新的幸福科学如何解释个人和国家在幸福方面的变化。

## 幸福感得分和排名使用盖洛普世界民意调查的数据。这些分数是基于对民意测验中所问的主要生命评估问题的答案。这个问题，被称为“坎蒂尔阶梯”，要求受
# 访者考虑一个梯子，其中最美好的生活是10岁，最糟糕的生活是0，并以此为尺度对自己的当前生活进行评分。这些分数来自2013-2016年的全国代表性样本，
# 并使用盖洛普权重来代表评估。
# 幸福指数后面的专栏估计了六个因素中的每一个因素——
# 经济生产、
# 社会支持、
# 预期寿命、
# 自由、
# 没有腐败和慷慨——
# 对每个国家的
# 生活评价都比反乌托邦的评价更高，而反乌托邦的生活评价值与世界的价值观相当六个因素中的每一个国家平均值最低。它们对每个国家的总得分没有影响，但
# 它们确实解释了为什么有些国家的排名高于其他国家。

## 关于这个报告 https://zh.m.wikipedia.org/zh/%E4%B8%96%E7%95%8C%E5%BF%AB%E6%A8%82%E5%A0%B1%E5%91%8A

## 作业 1 https://github.com/PhilippeCodes/World-Happiness-Report-Data-Analysis
## 作业 2 ![nPiZ8D](https://upiclw.oss-cn-beijing.aliyuncs.com/uPic/nPiZ8D.png) 使用 np 实现出来

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

# Configure the plot with training dataset.
# 训练集做个图
plot_training_trace = go.Scatter3d(
    x = x_train[:, 0].flatten(),
    y = x_train[:, 1].flatten(),
    z = y_train.flatten(),
    name = '程序开始-训练集',
    mode = 'markers',
    marker = {
        'size': 10,
        'opacity': 1,
        'line': {
            'color': 'rgb(255, 255, 255)',
            'width': 1
        },
    }
)

# 测试集做个图
plot_test_trace = go.Scatter3d(
    x = x_test[:, 0].flatten(),
    y = x_test[:, 1].flatten(),
    z = y_test.flatten(),
    name = '程序开始-测试集',
    mode = 'markers',
    marker = {
        'size': 10,
        'opacity': 1,
        'line': {
            'color': 'rgb(255, 255, 255)',
            'width': 1
        },
    }
)

plot_layout = go.Layout(
    title = 'Date Sets',
    scene = {
        'xaxis': {'title': x_1},
        'yaxis': {'title': x_2},
        'zaxis': {'title': y}
    },
    margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
)

plot_data = [plot_training_trace, plot_test_trace]

plot_figure = go.Figure(data=plot_data, layout=plot_layout)
plot_figure.update_layout(
    title = {
        'text': "原始数据集作图",   # 标题名称
        'y':0.9,  # 位置，坐标轴的长度看做1
        'x':0.5,
        'xanchor': 'center',   # 相对位置
        'yanchor': 'top'})
plotly.offline.plot(plot_figure)


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

# 损失作图显示
plt.plot(range(num_iterations), cost_history)
plt.xlabel('Iterations') # 迭代次数
plt.ylabel('Cost') # 损失，成本
plt.title('Gradient Descent Progress') # 梯度变化图
plt.show()

predictions_num = 10

x_min = x_train[:, 0].min();
x_max = x_train[:, 0].max();

y_min = x_train[:, 1].min();
y_max = x_train[:, 1].max();

x_axis = np.linspace(x_min, x_max, predictions_num)
y_axis = np.linspace(y_min, y_max, predictions_num)

x_predictions = np.zeros((predictions_num * predictions_num, 1))
y_predictions = np.zeros((predictions_num * predictions_num, 1))

x_y_index = 0
for x_index, x_value in enumerate(x_axis):
    for y_index, y_value in enumerate(y_axis):
        x_predictions[x_y_index] = x_value
        y_predictions[x_y_index] = y_value
        x_y_index += 1
# 预测
z_predictions = linear_regression.predict(np.hstack((x_predictions, y_predictions)))

# 预测值作图
plot_predictions_trace = go.Scatter3d(
    x = x_predictions.flatten(),
    y = y_predictions.flatten(),
    z = z_predictions.flatten(),
    name = 'Prediction Plane',
    mode = 'markers',
    marker={
        'size': 1,
    },
    opacity = 0.8,
    surfaceaxis = 2,
)

plot_data = [plot_training_trace, plot_test_trace, plot_predictions_trace]
plot_figure = go.Figure(data=plot_data, layout=plot_layout)
plot_figure.update_layout(
    title = {
        'text': "预测平面作图",   # 标题名称
        'y':0.9,  # 位置，坐标轴的长度看做1
        'x':0.5,
        'xanchor': 'center',   # 相对位置
        'yanchor': 'top'})
plotly.offline.plot(plot_figure)
