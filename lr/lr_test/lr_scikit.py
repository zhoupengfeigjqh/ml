# -*- coding:utf-8 -*-
from sklearn import linear_model
import numpy as np


if __name__ == "__main__":

    # 我们人工构建一个数据集合X
    X1 = np.random.normal(0.5, 2, 100)
    X2 = np.random.normal(3, 1.5, 100)
    X3 = np.random.normal(1.7, 1.2, 100)

    # Y
    Y = 2.1 * X1 + 0.5 * X2 + 1.3 * X3 + 13.2

    # 加入扰动
    X1 = X1 + np.random.normal(0, 0.2, 100)
    X2 = X2 + np.random.normal(0, 0.2, 100)
    X3 = X3 + np.random.normal(0, 0.2, 100)

    # 加入两个无关特征
    X4 = np.random.normal(0.3, 2.3, 100)
    X5 = np.random.normal(2.3, 1.3, 100)

    X = np.concatenate(([X1], [X2], [X3], [X4], [X5]), axis=0).transpose()

    # 创建
    # model = linear_model.Ridge()
    # model = linear_model.Lasso()
    model = linear_model.LinearRegression()

    # 拟合
    model.fit(X, Y)

    # 拟合结果
    print(model.coef_, model.intercept_)

    # R2的值 (1-u/v), u=sum((y_true-y_pred)**2), v=sum((y_true-y_true.mean())**2)
    # R2值越大越好
    print("R2 value:", model.score(X, Y))

    # 预测
    print(model.predict([[1, 2, 1.5, 1.3, 2.3]]))






