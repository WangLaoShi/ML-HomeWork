target = [1.5, 2.1, 3.3, -4.7, -2.3, 0.75]  # 样本实际值
prediction = [0.5, 1.5, 2.1, -2.2, 0.1, -0.5]  # 拟合值

error = []
for i in range(len(target)):
    error.append(target[i] - prediction[i])

print("Errors: ", error)

from math import sqrt

# 均方误差MSE
def MSE(self, error):
    squaredError = []
    for val in error:
        squaredError.append(val *  val) # 差值平方和

    print("MSE: ", sum(squaredError) / len(squaredError))

# 均方根误差 RMSE
def RMSE(self):
    print("RMSE = ", sqrt(sum(squaredError) / len(squaredError)))

# 平均绝对误差 MAE
def MAE(self):
    absError = []
    for val in error:
        absError.append(abs(val))  # 误差绝对值
    print("MAE = ", sum(absError) / len(absError))  # 平均绝对误差MAE

def R_squared(self):
    targetDeviation = []
    targetMean = sum(target) / len(target) # 样本均值
    for val in target:
        targetDeviation.append((val - targetMean) * (val - targetMean))  # 拟合值-均值

    R2 = sum(targetDeviation) / (sum(squaredError) / len(squaredError))

    return R2

a = MAE(error)
