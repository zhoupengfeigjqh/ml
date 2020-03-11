# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import GMM
from sklearn import preprocessing
import matplotlib.pyplot as plt

if __name__ == "__main__":
    """行情分类计算
    """

    path = "C:\\Users\\DELL\\Desktop\\ml\\ml\\gmm\\data\\zz500.csv"

    data = pd.read_csv(path)

    print(data)

    # # 计算过去n天的涨跌幅、以及波动
    n = 5
    v = round((data["highestIndex"] - data["lowestIndex"]) / data["lowestIndex"], 6)
    data["avg_volt_per"] = v.rolling(window=n, min_periods=0).mean()

    p1 = data["closeIndex"][n:].reset_index(drop=True)
    p2 = data["closeIndex"][:-n].reset_index(drop=True)\

    chg = (p2 - p1) / p1

    data = (data.iloc[n:]).reset_index(drop=True)

    data['chg_per'] = chg

    # 分布图
    plt.subplot(2, 1, 1)
    plt.hist(data["chg_per"], bins=50)
    plt.xlabel("chg_per")
    plt.ylabel("count")
    plt.subplot(2, 1, 2)
    plt.hist(data["avg_volt_per"], bins=50)
    plt.xlabel("avg_volt_per")
    plt.ylabel("count")
    plt.show()

    # 箱线图
    plt.subplot(1, 2, 1)
    plt.boxplot(data["chg_per"])
    plt.xlabel("chg_per")
    plt.subplot(1, 2, 2)
    plt.boxplot(data["avg_volt_per"])
    plt.xlabel("avg_volt_per")
    plt.show()

    # chg_per正态性检测
    print(stats.kstest(data["chg_per"], "norm"))

    # volt_per正态性检验
    print(stats.kstest(data["avg_volt_per"], "norm"))

    # 相关性检验
    print(data["avg_volt_per"].corr(data["chg_per"], method="pearson"))

    # 标准化
    ps = preprocessing.StandardScaler().fit(data[["chg_per"]])
    x1 = ps.transform(data[["chg_per"]])
    print(x1.mean(), x1.std())

    ps = preprocessing.StandardScaler().fit(data[["avg_volt_per"]])
    x2 = ps.transform(data[["avg_volt_per"]])
    print(x2.mean(), x2.std())

    d = np.concatenate((x1, x2), axis=1)

    gmm = GMM(n_components=5, covariance_type="full", random_state=0)
    print("*******样本类别*********")
    labels = gmm.fit(d).predict(d)
    print(labels)
    print("*******分布均值*********")
    print(gmm.means_)
    print("*******分布协方差*********")
    print(gmm.covars_)
    print("*******各类分布权重*********")
    print(gmm.weights_)

    f1 = plt.figure(1)
    plt.title("test scikit_learn gmm")
    plt.xlabel("dimension_1")
    plt.ylabel("dimension_2")
    plt.scatter(d[labels == 0, 0], d[labels == 0, 1], marker='o', color='g', s=10)
    plt.scatter(d[labels == 1, 0], d[labels == 1, 1], marker='o', color='r', s=10)
    plt.scatter(d[labels == 2, 0], d[labels == 2, 1], marker='o', color='y', s=10)
    plt.scatter(d[labels == 3, 0], d[labels == 3, 1], marker='o', color='b', s=10)
    plt.scatter(d[labels == 4, 0], d[labels == 4, 1], marker='o', color='k', s=10)
    # plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], marker='o', color='k', s=100)
    plt.show()

    # # 画K线状态序列图
    map_colors = {0: "g", 1: "r", 2: "y", 3: "b", 4: "k"}
    f1 = plt.figure(2)
    plt.title("zz500")
    data["labels"] = labels
    for i in range(0, max(labels) + 1):
        plt.scatter(data[data["labels"] == i].loc[0:].index, data[data["labels"] == i]["closeIndex"].loc[0:],
                    color=map_colors[i], marker="o", s=2)
    plt.savefig('sss', dpi=1000)
    plt.show()

    data.to_csv("./zz500.csv")
