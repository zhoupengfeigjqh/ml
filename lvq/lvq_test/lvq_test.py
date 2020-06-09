# -*- coding:utf-8 -*-
"""
   手写lvq, 实现一个简单的分类
"""
import numpy as np
import matplotlib.pyplot as plt
import random


def distance(x, y):
    """ 计算两个向量之间的距离
    :param x: 第一个向量
    :param y: 第二个向量
    :return: 返回计算值
    """
    return np.sqrt(sum((x-y) ** 2))


def lvq(data: np, k_num: int, k_labels: list, max_iter: int, lr: float):
    """
    :param data: 样本集, 最后一列feature表示原始数据的label
    :param k_num: 表示簇数
    :param k_labels: 表示簇标签
    :param max_iter: 最大迭代数
    :param lr: 学习效率
    :return: 返回向量中心点、簇标记、各个簇中心更新次数
    """
    # 初始化原型向量
    v = np.empty((k_num, data.shape[1]), dtype=np.float32)

    for i in range(k_num):
        # 根据labels, 获取当前label对应的原始数据
        samples = data[data[:, -1] == k_labels[i]]
        # 随机选择一个点作为初始簇心
        v[i] = random.choice(samples)

    # 记录各个中心向量的更新次数
    v_update_cnt = np.zeros(k_num, dtype=np.float32)

    j = 0
    while True:
        # 超过阈值且每个中心向量都更新超过5次则退出
        if j >= max_iter and sum(v_update_cnt > 5) == k_num:
            break

        j = j + 1

        # 随机选择一个样本, 并计算与当前各个簇中心点的距离, 取距离最小的
        sel_sample = random.choice(data)
        min_dist = distance(sel_sample, v[0])
        sel_k = 0
        for ii in range(1, k_num):
            dist = distance(sel_sample, v[ii])
            if min_dist > dist:
                min_dist = dist
                sel_k = ii

        # 根据sel_sample对应样本的标签, 更新v
        if sel_sample[-1] == v[sel_k][-1]:
            v[sel_k][0:-1] = v[sel_k][0:-1] + lr * (sel_sample[0:-1] - v[sel_k][0:-1])
        else:
            v[sel_k][0:-1] = v[sel_k][0:-1] + lr * (sel_sample[0:-1] - v[sel_k][0:-1])

        # v的更新次数
        v_update_cnt[sel_k] = v_update_cnt[sel_k] + 1

    # 更新完毕后, 把各个样本点进行标记, 记录放在categories变量里
    categories = []
    for jj in range(data.shape[0]):
        min_dist = distance(data[jj, :], v[0])
        sel_k = 0
        for kk in range(1, k_num):
            dist = distance(data[jj, :], v[kk])
            if min_dist > dist:
                min_dist = dist
                sel_k = kk
        categories.append(sel_k)

    return v, v_update_cnt, categories


if __name__ == "__main__":
    # 初始数据
    data1 = np.random.normal(4, 1, 30)
    data2 = np.random.normal(6, 0.5, 30)
    data3 = np.random.normal(8, 0.5, 30)
    d1 = np.concatenate((data1, data2, data3), axis=0).reshape(1, 90)

    data1 = np.random.normal(2, 0.51, 30)
    data2 = np.random.normal(4, 0.5, 30)
    data3 = np.random.normal(6, 0.5, 30)
    d2 = np.concatenate((data1, data2, data3), axis=0).reshape(1, 90)

    data1 = np.ones(30) * 1
    data2 = np.ones(30) * 2
    data3 = np.ones(30) * 3
    d3 = np.concatenate((data1, data2, data3), axis=0).reshape(1, 90)

    d = np.concatenate((d1, d2, d3))

    d = d.transpose()

    # print(d)

    v, v_update_cnt, categories = lvq(d, 4, [1, 1, 2, 3], 1500, 0.01)

    # 中心点
    print("*******样本中心*********")
    print(v)
    #
    # 类别样本
    print("*******样本类别集合索引*********")
    print(categories)

    f1 = plt.figure(1)
    plt.title("test lvq")
    plt.xlabel("dimension_1")
    plt.ylabel("dimension_2")
    plt.scatter(d[:, 0], d[:, 1], marker='o', color='g', s=50)
    plt.scatter(v[:, 0], v[:, 1], marker='o', color='r', s=200)
    m, n = np.shape(d)
    # for i in range(m):
        # plt.plot([d[i, 0], v[c[i], 0]], [d[i, 1], v[c[i], 1]], "c--", linewidth=0.3)
    plt.show()



