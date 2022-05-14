import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
import sys

print(sys.path)
# plotly.offline.init_notebook_mode()


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
# 并使用盖洛普权重来代表评估。幸福指数后面的专栏估计了六个因素中的每一个因素——经济生产、社会支持、预期寿命、自由、没有腐败和慷慨——对每个国家的
# 生活评价都比反乌托邦的评价更高，而反乌托邦的生活评价值与世界的价值观相当六个因素中的每一个国家平均值最低。它们对每个国家的总得分没有影响，但
# 它们确实解释了为什么有些国家的排名高于其他国家。

## 关于这个报告 https://zh.m.wikipedia.org/zh/%E4%B8%96%E7%95%8C%E5%BF%AB%E6%A8%82%E5%A0%B1%E5%91%8A

## 作业 1 https://github.com/PhilippeCodes/World-Happiness-Report-Data-Analysis
## 作业 2 ![nPiZ8D](https://upiclw.oss-cn-beijing.aliyuncs.com/uPic/nPiZ8D.png) 使用 np 实现出来
import numpy as np
import sys

sys.path.append(
    '/Volumes/WD_BLACK/国际人/IPS-Teaching-Material/七龙珠计划/4. 机器学习训练营/IPS-ML-Teaching/2-线性回归代码实现/线性回归-代码实现/utils')
def prepare_for_training(data, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
    # 计算样本总数
    num_examples = data.shape[0]

    data_processed = np.copy(data)

    # 预处理
    features_mean = 0
    features_deviation = 0
    data_normalized = data_processed
    if normalize_data:
        (
            data_normalized,
            features_mean,
            features_deviation
        ) = normalize(data_processed)

        data_processed = data_normalized

    # 特征变换sinusoidal
    if sinusoid_degree > 0:
        sinusoids = generate_sinusoids(data_normalized, sinusoid_degree)
        data_processed = np.concatenate((data_processed, sinusoids), axis=1)

    # 特征变换polynomial
    if polynomial_degree > 0:
        polynomials = generate_polynomials(data_normalized, polynomial_degree, normalize_data)
        data_processed = np.concatenate((data_processed, polynomials), axis=1)

    # 加一列1
    data_processed = np.hstack((np.ones((num_examples, 1)), data_processed))

    return data_processed, features_mean, features_deviation
class LinearRegression:
    '''
    线性回归实现.到目前为止还是一个简单的线性回归方法
    '''

    def __init__(self, data, labels, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
        """
        1.对数据进行预处理操作
        2.先得到所有的特征个数
        3.初始化参数矩阵
        """
        (data_processed,
         features_mean,
         features_deviation) = prepare_for_training(data, polynomial_degree, sinusoid_degree, normalize_data=True)

        self.data = data_processed
        self.labels = labels
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data

        num_features = self.data.shape[1]
        self.theta = np.zeros((num_features, 1))

    def train(self, alpha, num_iterations=500):
        """
        训练模块，执行梯度下降
        """
        cost_history = self.gradient_descent(alpha, num_iterations)
        return self.theta, cost_history

    def gradient_descent(self, alpha, num_iterations):
        """
        实际迭代模块，会迭代 num_iterations 次
        """
        cost_history = []
        for _ in range(num_iterations):
            self.gradient_step(alpha)
            cost_history.append(self.cost_function(self.data, self.labels))
        return cost_history

    def gradient_step(self, alpha):
        """
        梯度下降参数更新计算方法，注意是矩阵运算
        """
        num_examples = self.data.shape[0]
        prediction = LinearRegression.hypothesis(self.data, self.theta)
        delta = prediction - self.labels
        theta = self.theta
        theta = theta - alpha * (1 / num_examples) * (np.dot(delta.T, self.data)).T
        self.theta = theta

    def cost_function(self, data, labels):
        """
        损失计算方法
        """
        num_examples = data.shape[0]
        delta = LinearRegression.hypothesis(self.data, self.theta) - labels
        cost = (1 / 2) * np.dot(delta.T, delta) / num_examples
        return cost[0][0]

    @staticmethod
    def hypothesis(data, theta):
        '''
        得到预测值（假设值）
        :param data:
        :param theta:
        :return:
        '''
        predictions = np.dot(data, theta)
        return predictions

    def get_cost(self, data, labels):
        '''
        获取 cost 成本
        :param data:
        :param labels:
        :return:
        '''
        data_processed = prepare_for_training(data,
                                              self.polynomial_degree,
                                              self.sinusoid_degree,
                                              self.normalize_data
                                              )[0]

        return self.cost_function(data_processed, labels)

    def predict(self, data):
        """
        用训练的参数模型，与预测得到回归值结果
        """
        data_processed = prepare_for_training(data,
                                              self.polynomial_degree,
                                              self.sinusoid_degree,
                                              self.normalize_data
                                              )[0]

        predictions = LinearRegression.hypothesis(data_processed, self.theta)

        return predictions
data = pd.read_csv(r'C:\Users\DELL\Desktop\world-happiness-report-2017.csv')

## 分离训练集和测试集，8：2

train_data = data.sample(frac=0.8)
test_data = data.drop(train_data.index)

input_param_name_1 = 'Economy..GDP.per.Capita.'
input_param_name_2 = 'Freedom'
output_param_name = 'Happiness.Score'

x_train = train_data[[input_param_name_1, input_param_name_2]].values
y_train = train_data[[output_param_name]].values

x_test = test_data[[input_param_name_1, input_param_name_2]].values
y_test = test_data[[output_param_name]].values

# Configure the plot with training dataset.
# 训练集做个图
plot_training_trace = go.Scatter3d(
    x=x_train[:, 0].flatten(),
    y=x_train[:, 1].flatten(),
    z=y_train.flatten(),
    name='Training Set',
    mode='markers',
    marker={
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
    x=x_test[:, 0].flatten(),
    y=x_test[:, 1].flatten(),
    z=y_test.flatten(),
    name='Test Set',
    mode='markers',
    marker={
        'size': 10,
        'opacity': 1,
        'line': {
            'color': 'rgb(255, 255, 255)',
            'width': 1
        },
    }
)

plot_layout = go.Layout(
    title='Date Sets',
    scene={
        'xaxis': {'title': input_param_name_1},
        'yaxis': {'title': input_param_name_2},
        'zaxis': {'title': output_param_name}
    },
    margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
)

plot_data = [plot_training_trace, plot_test_trace]

plot_figure = go.Figure(data=plot_data, layout=plot_layout)

plotly.offline.plot(plot_figure)

num_iterations = 500
learning_rate = 0.01
polynomial_degree = 0
sinusoid_degree = 0

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
plt.xlabel('Iterations')  # 迭代次数
plt.ylabel('Cost')  # 损失，成本
plt.title('Gradient Descent Progress')  # 梯度
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
    x=x_predictions.flatten(),
    y=y_predictions.flatten(),
    z=z_predictions.flatten(),
    name='Prediction Plane',
    mode='markers',
    marker={
        'size': 1,
    },
    opacity=0.8,
    surfaceaxis=2,
)

plot_data = [plot_training_trace, plot_test_trace, plot_predictions_trace]
plot_figure = go.Figure(data=plot_data, layout=plot_layout)
plotly.offline.plot(plot_figure)
