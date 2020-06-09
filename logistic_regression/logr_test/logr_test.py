# -*- coding:utf-8 -*-
"""
   利用sklearn LogisticRegressio, 实现一个简单的分类
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics


def initial_data(n=100):
    """ 创建样例数据，包括输入和输出, 让类别接近5:5
    """
    x1_1 = np.random.normal(2.3, 0.3, n).reshape(n, 1)
    x2_1 = np.random.normal(1, 0.3, n).reshape(n, 1)

    y_1 = np.array([1] * n).reshape(n, 1)

    x1_0 = np.random.normal(1, 0.6, n).reshape(n, 1)
    x2_0 = np.random.normal(1.5, 0.6, n).reshape(n, 1)

    y_0 = np.array([0] * n).reshape(n, 1)

    x1 = np.concatenate((x1_1, x1_0), axis=0)
    x2 = np.concatenate((x2_1, x2_0), axis=0)
    x0 = np.array([1] * 2 * n).reshape(2 * n, 1)
    y = np.concatenate((y_1, y_0), axis=0)

    x = np.concatenate((x1, x2), axis=1)

    plt.figure(0)

    plt.scatter(x1_0, x2_0, marker='o', color='r', s=50)
    plt.scatter(x1_1, x2_1, marker='o', color='g', s=50)

    return x, y


def confusion(label, label_p):
    """混淆矩阵计算
    """
    # TP FP TN FN
    tp = (label_p[label == 1] == 1).tolist().count(True)
    fn = (label_p[label == 1] == 0).tolist().count(True)
    tn = (label_p[label == 0] == 0).tolist().count(True)
    fp = (label_p[label == 0] == 1).tolist().count(True)

    c = np.array([[tp, tn], [fp, fn]])

    c = pd.DataFrame(c, index=["T", "F"], columns=["P", "N"])

    print("confusion detail")
    print(c)
    print("precision: ", tp / (tp + fp))
    print("recall: ", tp / (tp + fn))


def plot_best_fit(weights, intercept):
    """在原始数据上划出决策边界
    """
    xx1 = np.arange(0, 3, 0.1)
    xx2 = -(np.array([1.0]*len(xx1)) * intercept[0] + xx1 * weights[0][0]) / weights[0][1]
    plt.plot(xx1, xx2)
    plt.show()


def roc_curve(y_true, y_score):
    """roc 曲线
    """
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_score, drop_intermediate=True)

    plt.figure(1)
    plt.scatter(fpr, tpr, marker="o", c="r")
    plt.plot(fpr, tpr, c="b")
    plt.title("ROC")
    plt.show()

    auc_area(fpr, tpr)


def auc_area(x, y):
    """auc 面积
    """
    auc = sklearn.metrics.auc(x, y)
    print("AUC: ", auc)


if __name__ == "__main__":

    # 生成初始数据
    indata, label = initial_data()

    m, n = indata.shape

    model = LogisticRegression()

    model.fit(indata, label)

    # 计算出来的权重
    print("weights: ", model.coef_)
    print("intercept: ", model.intercept_)

    # 划决策线
    plot_best_fit(model.coef_, model.intercept_)

    # 混淆矩阵
    label_p = model.predict(indata)
    confusion(label.transpose()[0], label_p)

    # ROC 和 AUC
    roc_curve(label, model.decision_function(indata))


