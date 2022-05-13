# 1. 均方误差MSE
# MSE（均方误差）（Mean Square Error）
# MSE是真实值与预测值的差值的平方然后求和平均。
# 范围[0,+∞)，当预测值与真实值完全相同时为0，误差越大，该值越大。
import numpy as np
from sklearn import metrics
y_true = np.array([1.0, 5.0, 4.0, 3.0, 2.0, 5.0, -3.0])
y_pred = np.array([1.0, 4.5, 3.5, 5.0, 8.0, 4.5, 1.0])
print(metrics.mean_squared_error(y_true, y_pred))
print("")
# 结果为：8.107142857142858

# 2. 均方根误差RMSE
# MSE单项的平方根之和
y_true = np.array([1.0, 5.0, 4.0, 3.0, 2.0, 5.0, -3.0])
y_pred = np.array([1.0, 4.5, 3.5, 5.0, 8.0, 4.5, 1.0])
print(np.sqrt(metrics.mean_squared_error(y_true, y_pred)))
print("")
# 结果为：2.847304489713536

# 3. 平均绝对误差MAE
# MSE单项的绝对值之和
y_true = np.array([1.0, 5.0, 4.0, 3.0, 2.0, 5.0, -3.0])
y_pred = np.array([1.0, 4.5, 3.5, 5.0, 8.0, 4.5, 1.0])
print(metrics.mean_absolute_error(y_true, y_pred))
print("")
# 结果为：1.9285714285714286

# 4. R-squard
print("")
from sklearn.metrics import r2_score
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
print(r2_score(y_true, y_pred))

y_true = [[0.5, 1], [-1, 1], [7, -6]]
y_pred = [[0, 2], [-1, 2], [8, -5]]
print(r2_score(y_true, y_pred, multioutput='variance_weighted'))

y_true = [1, 2, 3]
y_pred = [1, 2, 3]
print(r2_score(y_true, y_pred))

y_true = [1, 2, 3]
y_pred = [2, 2, 2]
print(r2_score(y_true, y_pred))

y_true = [1, 2, 3]
y_pred = [3, 2, 1]
print(r2_score(y_true, y_pred))
print("")

y_true = [-2, -2, -2]
y_pred = [-2, -2, -2]
print(r2_score(y_true, y_pred))
print(r2_score(y_true, y_pred, force_finite=False))
print("")

y_true = [-2, -2, -2]
y_pred = [-2, -2, -2 + 1e-8]
print(r2_score(y_true, y_pred))
print(r2_score(y_true, y_pred, force_finite=False))