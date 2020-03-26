# -*- coding:utf-8 -*-
"""
基于逻辑斯蒂回归来演示随机梯度上升算法
随机梯度可以实现在线更新
"""
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd


def sigmod(data):
    """计算sigmod值
    """
    return 1.0 / (1 + np.exp(-data))


def initial_data(n=100):
    """ 创建样例数据，包括输入和输出, 让类别接近5:5
    """
    x1_1 = np.random.normal(2.3, 0.6, n).reshape(n, 1)
    x2_1 = np.random.normal(1, 0.6, n).reshape(n, 1)

    y_1 = np.array([1] * n).reshape(n, 1)

    x1_0 = np.random.normal(1, 0.6, n).reshape(n, 1)
    x2_0 = np.random.normal(1.5, 0.6, n).reshape(n, 1)

    y_0 = np.array([0] * n).reshape(n, 1)

    x1 = np.concatenate((x1_1, x1_0), axis=0)
    x2 = np.concatenate((x2_1, x2_0), axis=0)
    x0 = np.array([1] * 2 * n).reshape(2 * n, 1)
    y = np.concatenate((y_1, y_0), axis=0)

    x = np.concatenate((x0, x1, x2), axis=1)

    plt.figure(0)

    plt.scatter(x1_0, x2_0, marker='o', color='r', s=50)
    plt.scatter(x1_1, x2_1, marker='o', color='g', s=50)

    # plt.show()

    return x, y


def stoc_gradient_descend(indata, label):
    """随机梯度上升算法，每次只取样本中的一个点进行计算
    """
    m, n = indata.shape

    indata = np.mat(indata)
    label = np.mat(label)

    # 初始化权重
    weights = np.ones((1, n))

    his_weights = [[], [], []]

    # 循环选取样本
    for j in range(100):
        ids = list(range(m))
        for i in range(0, m):
            alpha = 1/(1 + i + j) + 0.001

            sel_id = int(random.uniform(0, len(ids)))

            yy = sigmod(indata[ids[sel_id]] * weights.transpose())
            error = label[ids[sel_id]] - yy

            # 由于求最大似然估计，求最大值下的参数, 故采用梯度上升
            weights = weights + alpha * np.array(indata[ids[sel_id]]) * np.array(error)

            del ids[sel_id]

            his_weights[0].append(weights[0][0])
            his_weights[1].append(weights[0][1])
            his_weights[2].append(weights[0][2])

    return weights, his_weights


def predict(indata, weights):
    """训练结果带入进行预测，返回值用来和原始标记对比
    """
    y = sigmod(indata * weights.transpose())

    y[y > 0.5] = 1
    y[y <= 0.5] = 0

    return y


def confusion(indata, weights, label):
    """
    混淆矩阵
    """

    # 预测结果
    label_p = predict(indata, weights)

    # TP FP TN FN
    tp = ((label_p[label == 1] == 1).tolist())[0].count(True)
    fn = ((label_p[label == 1] == 0).tolist())[0].count(True)
    tn = ((label_p[label == 0] == 0).tolist())[0].count(True)
    fp = ((label_p[label == 0] == 1).tolist())[0].count(True)

    c = np.array([[tp, tn], [fp, fn]])

    c = pd.DataFrame(c, index=["T", "F"], columns=["P", "N"])

    print("confusion detail")
    print(c)
    print("precision: ", tp / (tp + fp))
    print("recall: ", tp / (tp + fn))


def roc(indata, weights, label):
    """
    调整阈值，记录不同阈值下的 TPR和FPR
    """
    predict_value = sigmod(indata * weights.transpose())

    tpr = []  # 真阳率 TPR = TP / (TP + FN)
    fpr = []  # 假阳率 FPR = FP / (FP + TN)

    for v in np.arange(0.01, 0.99, 0.01):
        tmp_label = predict_value.copy()
        tmp_label[tmp_label > v] = 1
        tmp_label[tmp_label <= v] = 0

        tp = ((tmp_label[label == 1] == 1).tolist())[0].count(True)
        fn = ((tmp_label[label == 1] == 0).tolist())[0].count(True)
        tn = ((tmp_label[label == 0] == 0).tolist())[0].count(True)
        fp = ((tmp_label[label == 0] == 1).tolist())[0].count(True)

        tpr.append(tp / (tp + fn))
        fpr.append(fp / (fp + tn))

    plt.figure(2)
    plt.scatter(fpr, tpr, marker="o", c="r")
    plt.plot(fpr, tpr, c="b")
    plt.title("ROC")
    plt.show()


def plot_best_fit(weights):
    """在原始数据上划出决策边界
    """
    xx1 = np.arange(0, 3, 0.1)
    xx2 = -(np.array([1.0]*len(xx1)) * weights[0][0] + xx1 * weights[0][1]) / weights[0][2]
    plt.plot(xx1, xx2)
    plt.show()


def param_iteration_detail(his_weights):
    """参数迭代详情
    """
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


if __name__ == "__main__":

    # 生成初始数据
    indata, label = initial_data()

    m, n = indata.shape

    indata = np.mat(indata)
    label = np.mat(label)

    weights, his_weights = stoc_gradient_descend(indata, label)

    # 计算出来的权重
    print(weights)

    # 划出决策面 根据 1.0/(1+e^(-z))可知，当z>0时，即x0*w0+x1*w1+x2*w2>0 取1，反之取0
    plot_best_fit(weights)

    # 参数迭代情况
    param_iteration_detail(his_weights)

    # 预测结果
    label_p = predict(indata, weights)

    confusion(indata, weights, label)

    # ROC曲线
    roc(indata, weights, label)

