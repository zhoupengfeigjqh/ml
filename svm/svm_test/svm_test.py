# -*- coding:utf-8 -*-
"""
svm-非线性支持向量机
"""
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import random


def initial_data(n=3600):
    """生成一个二维非线性集合
    """
    y = []
    x = []

    for i in range(n):
        x0 = random.uniform(-2, 2)
        x1 = random.uniform(-2, 2)
        if x0*x0 + x1*x1 <= 1:
            x.append([x0, x1])
            y.append(1)

        if (x0*x0 + x1*x1 <= 2) and (x0*x0 + x1*x1 >= 1.3):
            x.append([x0, x1])
            y.append(-1)

    x = np.array(x)
    y = np.array(y)
    x.transpose()

    plt.figure(1)
    plt.scatter(x[y == 1].transpose()[0], x[y == 1].transpose()[1], marker="o", c="b")
    plt.scatter(x[y == -1].transpose()[0], x[y == -1].transpose()[1], marker="o", c="g")

    return x, y


if __name__ == "__main__":
    # 数据初始化
    x, y = initial_data(3600)

    # 构建模型
    model = SVC(kernel="rbf", probability=True)

    # 训练模型
    model.fit(x, y)

    # 支持向量
    v = model.support_vectors_.transpose()
    plt.scatter(v[0], v[1], marker="o", c="y")
    plt.show()


