import pandas as pd
import numpy as np
from scipy import stats
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sys


if __name__ == "__main__":
    """行情聚类计算
    """
    sys.path.append(r"C://Users//DELL//Desktop//ml//ml//kmeans//data_pretreat")
    from kline_rebuild import vol_series as vs

    # 读取数据
    file_names = ["deribit_trade_perpetual.BTC.20191201.csv",
                  "deribit_trade_perpetual.BTC.20191202.csv",
                  "deribit_trade_perpetual.BTC.20191203.csv",
                  "deribit_trade_perpetual.BTC.20191204.csv",
                  "deribit_trade_perpetual.BTC.20191205.csv"]

    data = vs.load_data(file_names)
    print(data.describe())

    # 获取K线数据
    # 设置 q_usd 累计阈值20000
    q_usd_vol_threshold = 20000
    data = vs.vol_rebuild(data=data, vtype="usd", threshold=q_usd_vol_threshold)

    # 计算过去k根K线的平均波动率和涨跌幅，用这两个特征进行聚类
    k = 15
    v = round((data["h"] - data["l"]) / data["l"], 6)
    data["avg_volt_per"] = v.rolling(window=k, min_periods=0).mean()

    p1 = data["c"][k:].reset_index(drop=True)
    p2 = data["c"][:-k].reset_index(drop=True)
    chg_per = (p1 - p2) / p2
    data = data.iloc[k:].reset_index(drop=True)
    data["chg_per"] = chg_per

    # 统计信息
    print(data.describe())

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

    # 二元关系 相关性检验
    print(data["avg_volt_per"].corr(data["chg_per"], method="pearson"))

    # 标准化
    ps = preprocessing.StandardScaler().fit(data[["chg_per"]])
    x1 = ps.transform(data[["chg_per"]])
    print(x1.mean(), x1.std())

    ps = preprocessing.StandardScaler().fit(data[["avg_volt_per"]])
    x2 = ps.transform(data[["avg_volt_per"]])
    print(x2.mean(), x2.std())

    d = np.concatenate((x1, x2), axis=1)

    # print(d)
    # # # 涨跌进行聚类
    # inertia_list = []
    # for i in range(2, 10):
    #     km = KMeans(n_clusters=i, n_init=10)
    #     km.fit(d)
    #
    #     print("*******样本误差*********")
    #     print(km.inertia_)
    #     inertia_list.append(km.inertia_)
    #
    # # 不同K值下的样本误差
    # f1 = plt.figure(1)
    # plt.plot(inertia_list)
    # plt.show()

    km = KMeans(n_clusters=4, n_init=10)
    km.fit(d)

    print("*******样本中心*********")
    print(km.cluster_centers_)

    print("*******样本类别*********")
    print(km.labels_)

    print("*******样本误差*********")
    print(km.inertia_)

    print("*******最大迭代数*********")
    print(km.n_iter_)

    f1 = plt.figure(1)
    plt.title("test scikit_learn kmeans")
    plt.xlabel("dimension_1")
    plt.ylabel("dimension_2")
    plt.scatter(x1, x2, marker='o', color='g', s=1)
    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], marker='o', color='r', s=200)
    m, n = np.shape(d)
    # for i in range(m):
    #     plt.plot([d[i, 0], km.cluster_centers_[km.labels_[i], 0]], [d[i, 1], km.cluster_centers_[km.labels_[i], 1]],
    #              "c--", linewidth=0.3)
    plt.show()

