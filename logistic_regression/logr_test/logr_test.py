# -*- coding:utf-8 -*-
"""scikit learn逻辑斯蒂回归
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


def initial_data(n=100):
    """ 创建样例数据，包括输入和输出, 让类别接近5:5
    """
    x1_1 = np.random.normal(2.3, 0.3, n).reshape(n, 1)
    x2_1 = np.random.normal(1, 0.3, n).reshape(n, 1)

    y_1 = np.array([1] * n).reshape(n, 1)

    x1_0 = np.random.normal(0.6, 0.1, n).reshape(n, 1)
    x2_0 = np.random.normal(0.3, 0.1, n).reshape(n, 1)

    y_0 = np.array([0] * n).reshape(n, 1)

    x1 = np.concatenate((x1_1, x1_0), axis=0)
    x2 = np.concatenate((x2_1, x2_0), axis=0)
    x0 = np.array([1] * 2 * n).reshape(2 * n, 1)
    y = np.concatenate((y_1, y_0), axis=0)

    x = np.concatenate((x0, x1, x2), axis=1)

    plt.figure(0)

    plt.scatter(x1_0, x2_0, marker='o', color='r', s=50)
    plt.scatter(x1_1, x2_1, marker='o', color='g', s=50)

    return x, y


if __name__ == "__main__":

    # 生成初始数据
    indata, label = initial_data()

    m, n = indata.shape

    indata = np.mat(indata)
    label = np.mat(label)

    weights, his_weights = LogisticRegression(indata, label)

    # 计算出来的权重
    print(weights)

    # 划出决策面 根据 1.0/(1+e^(-z))可知，当z>0时，即x0*w0+x1*w1+x2*w2>0 取1，反之取0
    xx1 = np.arange(0, 3, 0.1)
    xx2 = -(np.array([1.0]*len(xx1)) * weights[0][0] + xx1 * weights[0][1]) / weights[0][2]
    plt.plot(xx1, xx2)
    plt.show()

    # 参数的迭代收敛情况
    plt.figure(1)

    plt.subplot(3, 1, 1)

    plt.plot(his_weights[0])
    plt.ylabel("x0")

    plt.subplot(3, 1, 2)
    plt.plot(his_weights[1])
    plt.ylabel("x1")

    plt.subplot(3, 1, 3)
    plt.plot(his_weights[2])
    plt.ylabel("x2")
    plt.show()
