"""
Normalize features
规范化特征
"""

import numpy as np


def normalize(features):
    '''
    归一化操作：（原来的值-均值）/ 标准差
    :param features:
    :return:
    '''
    features_normalized = np.copy(features).astype(float)

    # 计算均值
    features_mean = np.mean(features, 0)

    # 计算标准差
    features_deviation = np.std(features, 0)

    # 如果样本数大于 1 ，标准化/归一化操作（原来的值-均值）
    if features.shape[0] > 1:
        features_normalized -= features_mean

    # 防止除以0
    features_deviation[features_deviation == 0] = 1
    features_normalized /= features_deviation

    return features_normalized, features_mean, features_deviation
