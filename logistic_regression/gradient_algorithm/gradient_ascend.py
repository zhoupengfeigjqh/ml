# -*- coding:utf-8 -*-
"""基于逻辑斯蒂回归来演示梯度上升算法
"""
import numpy as np
import random
import matplotlib.pyplot as plt

def sigmod(data):
    """计算sigmod值
    """
    return 1.0 / (1 + np.exp(-data))


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


def gradient_descend(indata, label):
    """梯度上升算法
    """
    m, n = indata.shape

    indata = np.mat(indata)
    label = np.mat(label)

    # 初始化权重
    weights = np.ones((n, 1))
    his_weights = [[], [], []]

    # 迭代阈值
    maxiter = 10000
    alpha = 0.001

    # 循环计算
    for k in range(0, maxiter):
        yy = sigmod(indata * weights)
        error = label - yy

        # 由于求最大似然估计，求最大值下的参数, 故采用梯度上升
        weights = weights + alpha * (indata).transpose() * error

        temp = weights.tolist()
        his_weights[0].append(temp[0][0])
        his_weights[1].append(temp[1][0])
        his_weights[2].append(temp[2][0])

    return weights, his_weights


def predict(indata, weights):
    """训练结果带入进行预测，返回值用来和原始标记对比
    """
    y = sigmod(indata * weights.transpose())

    y[y > 0.5] = 1
    y[y <= 0.5] = 0

    return y


if __name__ == "__main__":

    # 生成初始数据
    indata, label = initial_data()

    m, n = indata.shape

    indata = np.mat(indata)
    label = np.mat(label)

    weights, his_weights = gradient_descend(indata, label)

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
