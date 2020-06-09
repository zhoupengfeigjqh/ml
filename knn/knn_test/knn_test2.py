# -*- coding:utf-8 -*-
"""
  利用sklearn KNeighborsClassifier, 实现实现一个简单的分类
"""
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt


def create_data_set():
    """创建数据集
    """
    # 二维数据
    data = np.array([[13.2, 9.3, 14.3, 27.8, 33.2, 24.3],
                     [0.5, 0.47, 0.23, 2.3, 2.1, 3.3]]).transpose()

    labels = np.array(["A", "A", "A", "B", "B", "B"])

    return data, labels


if __name__ == "__main__":

    # 获取数据
    data, labels = create_data_set()

    # 初始数据分布情况
    plt.scatter(data[labels == "A", 0], data[labels == "A", 1], c="y")
    plt.scatter(data[labels == "B", 0], data[labels == "B", 1], c="r")

    # 将输入数据归一标准化
    minmax_norm = preprocessing.MinMaxScaler()
    data = minmax_norm.fit_transform(data)

    model1 = KNeighborsClassifier(n_neighbors=3)

    model1.fit(data, labels)

    # 预测新数据
    indata = np.array([[15, 1.5]])
    print("new data：", indata)
    plt.scatter(indata[0][0], indata[0][1], c="b")
    plt.xlabel("x1")
    plt.xlabel("x2")
    plt.show()

    # 新数据归一化
    indata = minmax_norm.transform(np.array([[15, 1.5]]))

    print("new label: ", model1.predict(indata)[0])
    print("proba: ", model1.predict_proba(indata)[0])

    # new_label = classify(indata, data, labels, 3)

    # print("new label：", new_label)



