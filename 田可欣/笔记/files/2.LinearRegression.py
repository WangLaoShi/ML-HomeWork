import pprint

import numpy as np
import sys
sys.path.append('/Volumes/WD_BLACK/国际人/IPS-Teaching-Material/七龙珠计划/4. 机器学习训练营/IPS-ML-Teaching/2-线性回归代码实现/线性回归-代码实现/utils')
from features import prepare_for_training

# 原文地址：https://www.cnblogs.com/xiugeng/p/12977373.html

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

        print("-"*50)
        print("~~~~~~~~原始的数据集~~~~~~~~")
        pprint.pprint(data)

        (data_processed,
         features_mean,
         features_deviation) = prepare_for_training(data, polynomial_degree, sinusoid_degree, normalize_data=True)
        print("数据集的平均值" + str(features_mean))
        print("数据集的均差值" + str(features_deviation))
        print("~~~~~~~~处理后数据集~~~~~~~~")
        pprint.pprint(data_processed)
        print("-" * 50)
        self.data = data_processed
        self.labels = labels # 标记
        self.features_mean = features_mean # 均值
        self.features_deviation = features_deviation # 标准差
        self.polynomial_degree = polynomial_degree # 多项式
        self.sinusoid_degree = sinusoid_degree # sin 值
        self.normalize_data = normalize_data # 规范化
        # 获取多少个列作为特征量
        num_features = self.data.shape[1]# 1 是列个数，0 是样本个数
        self.theta = np.zeros((num_features, 1))# 构建θ矩阵

        print("-"*50)
        print("~~~~~~~~~~~~~~~~~~初始化的时候得到了什么东西~~~~~~~~~~~~~~~~~~")
        print("样本数(记录的行数)" + str(self.data.shape[0]))
        print("特征数(记录的列数)" + str(self.data.shape[1]))
        print("最原始的 theta 值")
        print(self.theta)
        print("-"*50)

    def train(self, alpha, num_iterations=500):
        """
        训练模块，执行梯度下降
        :param alpha: α为学习率（步长）
        :param num_iterations: 迭代次数
        """
        cost_history = self.gradient_descent(alpha, num_iterations)
        return self.theta, cost_history

    def gradient_descent(self, alpha, num_iterations):
        """
        实际迭代模块，会迭代 num_iterations 次
        :param alpha: 学习率
        :param num_iterations: 迭代次数
        """
        cost_history = []# 保存损失值
        for _ in range(num_iterations):
            print("第 " + str(_) + " 次循环~~~~")
            self.gradient_step(alpha)# 每次迭代参数更新
            cost_history.append(self.cost_function(self.data, self.labels))
        return cost_history

    # 看懂这个再看函数，![0mx7hh](https://upiclw.oss-cn-beijing.aliyuncs.com/uPic/0mx7hh.png)
    def gradient_step(self, alpha):
        """
        梯度下降参数更新计算方法（核心代码,矩阵运算）
        :param alpha: 学习率
        """
        num_examples = self.data.shape[0] # 样本数
        prediction = LinearRegression.hypothesis(self.data, self.theta) # 预测值
        # 残差 = 预测值-真实值
        delta = prediction - self.labels
        theta = self.theta
        # theta值更新，.T是执行转置
        # ![lsUHor](https://upiclw.oss-cn-beijing.aliyuncs.com/uPic/lsUHor.png)
        theta = theta - alpha * (1 / num_examples) * (np.dot(delta.T, self.data)).T
        self.theta = theta

    def cost_function(self, data, labels):
        """
        损失计算
        :param data: 数据集
        :param labels: 真实值
        :return:
        """
        num_examples = data.shape[0]  # 样本个数
        # 残差 = 预测值-真实值
        delta = LinearRegression.hypothesis(self.data, self.theta) - labels
        cost = (1 / 2) * np.dot(delta.T, delta) / num_examples
        print("👇"*25)
        print("巴拉巴拉小魔仙~~~~~~~~~~~~~~~当前损失~~~~~~~~~~~~~~~")
        print(cost)
        print('👆🏻'*25)
        return cost[0][0]

    @staticmethod
    def hypothesis(data, theta):
        '''
        得到预测值（假设值）
        :param data:
        :param theta:
        :return:
        '''
        # 如果处理的是一维数组，则得到的是两数组的內积；如果处理的是二维数组（矩阵），则得到的是矩阵积
        print("#"*25)
        print("嘛哩嘛哩哄~~~~~~~让我看看每次的 theta 值,它就是我们的 X 的前面的变量,它是靠梯度在改变")
        print(theta)
        print("#" * 25)
        predictions = np.dot(data, theta)
        return predictions

    def get_cost(self, data, labels):
        '''
        获取 cost 成本
        :param data:
        :param labels:
        :return:
        '''
        # 经过处理了的数据
        data_processed = prepare_for_training(data,
                                              self.polynomial_degree,
                                              self.sinusoid_degree,
                                              self.normalize_data
                                              )[0]
        # 返回损失值
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
