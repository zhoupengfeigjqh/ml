# -*- coding:utf-8 -*-
"""
   利用sklearn，实现一个简单的分类
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

if __name__ == "__main__":
    print("sss")
    # 初始数据
    data1 = np.random.normal(1, 0.3, 200)
    data2 = np.random.normal(3, 1, 400)
    data3 = np.random.normal(5, 2, 500)
    d1 = np.concatenate((data1, data2, data3), axis=0).reshape(1, 1100)

    data1 = np.random.normal(2, 1, 200)
    data2 = np.random.normal(4, 0.3, 400)
    data3 = np.random.normal(6, 0.3, 500)
    d2 = np.concatenate((data1, data2, data3), axis=0).reshape(1, 1100)

    d = np.concatenate((d1, d2))

    d = d.transpose()

    km = KMeans(n_clusters=3)
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
    plt.scatter(d[km.labels_ == 0, 0], d[km.labels_ == 0, 1], marker='o', color='g', s=30)
    plt.scatter(d[km.labels_ == 1, 0], d[km.labels_ == 1, 1], marker='o', color='b', s=30)
    plt.scatter(d[km.labels_ == 2, 0], d[km.labels_ == 2, 1], marker='o', color='y', s=30)
    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], marker='o', color='r', s=200)
    # m, n = np.shape(d)
    # for i in range(m):
    #     plt.plot([d[i, 0], km.cluster_centers_[km.labels_[i], 0]], [d[i, 1], km.cluster_centers_[km.labels_[i], 1]],
    #              "c--", linewidth=0.3)
    plt.show()

