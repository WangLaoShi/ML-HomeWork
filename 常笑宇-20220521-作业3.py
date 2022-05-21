import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

#创建一个希伯特矩阵(高度病态，任何一个元素的点发生变动，整个矩阵的行列式的值和逆矩阵都会发生巨大变化)
#这里的加法运算类似于矩阵相乘
X = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
y = np.ones(10)

#计算路径
n_alphas = 200
alphas = np.logspace(-10, -2, n_alphas)
clf = linear_model.Ridge(fit_intercept=False)

coefs = []
for a in alphas:
    clf.set_params(alpha=a)
    clf.fit(X, y)
    coefs.append(clf.coef_)
#图形展示
#设置刻度
ax = plt.gca()
#设置刻度的映射
ax.plot(alphas, coefs)
#设置x轴的刻度显示方式
ax.set_xscale('log')
#翻转x轴
ax.set_xlim(ax.get_xlim()[::-1])
#设置x、y标签以及标题
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
#使得坐标轴最大值和最小值与数据保持一致
plt.axis('tight')
plt.show()