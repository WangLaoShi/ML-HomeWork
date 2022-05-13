import numpy as np
testX = np.array([174.5, 171.2, 172.9, 161.6, 123.6, 112.1, 107.1, 98.6, 98.7, 97.5, 95.8, 93.5, 91.1, 85.2, 75.6, 72.7, 68.6,
         69.1, 63.8, 60.1, 65.2, 71, 75.8, 77.8])
testY = np.array([88.3, 87.1, 88.7, 85.8, 89.4, 88, 83.7, 73.2, 71.6, 71, 71.2, 70.5, 69.2, 65.1, 54.8, 56.7, 62, 68.2, 71.1,
         76.1, 79.8, 80.9, 83.7, 85.8])


def MSE(predictions, targets):
    return np.square(targets-predictions).mean()


def MAE(predictions, targets):
    return np.abs(targets-predictions).mean()


def RMSE(predictions, targets):
    return np.sqrt(np.square(targets-predictions).mean())


def computeCorrelation(predictions, targets):
    predictions_mean = np.mean(predictions)
    targets_mean = np.mean(targets)
    SSR = 0
    var_predictions = 0
    var_targets = 0
    for i in range(0, len(predictions)):
        dif_predictions_mean = predictions[i] - predictions_mean
        dif_targets_mean = targets[i] - targets_mean
        SSR += (dif_predictions_mean * dif_targets_mean)
        var_predictions += dif_predictions_mean ** 2
        var_targets += dif_targets_mean ** 2
    SST = np.sqrt(var_predictions * var_targets)
    return "r：", SSR / SST, "r-squared：", (SSR / SST) ** 2


print(computeCorrelation(testX, testY))
print(MAE(testX, testY))
print(MSE(testX, testY))
