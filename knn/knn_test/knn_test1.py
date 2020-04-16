# -*- coding:utf-8 -*-
"""
   手工实现实现一个简单的分类
"""
import numpy as np
from sklearn import preprocessing
from matplotlib import pyplot as plt


def create_data_set():
    """创建数据集
    """
    # 二维数据
    data = np.array([[13.2, 9.3, 14.3, 27.8, 33.2, 24.3],
                     [0.5, 0.47, 0.23, 2.3, 2.1, 3.3]]).transpose()

    labels = np.array(["A", "A", "A", "B", "B", "B"])

    return data, labels


def classify(indata, data, labels, k):
    """分类
    indata: 输入新数据
    data: 获得的样本数据
    k: K值
    """
    indata = indata.repeat(repeats=data.shape[0], axis=0)

    dist = np.sqrt(np.sum((data - indata)**2, axis=1))

    print(dist)

    ids = dist.argsort()

    class_cnts = {}

    for i in range(0, k):
        cur_label = labels[ids == i][0]
        if cur_label in class_cnts.keys():
            class_cnts[cur_label] = class_cnts[cur_label] + 1
        else:
            class_cnts[cur_label] = 1

    # 排序
    ret = sorted(class_cnts.items(), key=lambda x: x[1], reverse=True)

    # 取得数量最大的一类
    return ret[0][0]


if __name__ == "__main__":

    # 获取数据
    data, labels = create_data_set()

    # 初始数据分布情况
    plt.scatter(data[labels == "A", 0], data[labels == "A", 1], c="y")
    plt.scatter(data[labels == "B", 0], data[labels == "B", 1], c="r")

    # 将输入数据归一标准化
    minmax_norm = preprocessing.MinMaxScaler()
    data = minmax_norm.fit_transform(data)

    # 预测新数据
    indata = np.array([[8, 0.38]])
    print("new data：", indata)
    plt.scatter(indata[0][0], indata[0][1], c="b")
    plt.xlabel("x1")
    plt.xlabel("x2")
    plt.show()

    # 新数据归一化
    indata = minmax_norm.transform(np.array([[8, 0.38]]))
    new_label = classify(indata, data, labels, 3)

    print("new label：", new_label)



