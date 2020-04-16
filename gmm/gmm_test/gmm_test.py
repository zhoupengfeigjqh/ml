# -*- coding:utf-8 -*-
"""
   利用sklearn，实现一个简单的分类
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GMM
from matplotlib.patches import Ellipse
from sklearn.cluster import KMeans


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))


if __name__ == "__main__":
    print("sss")
    # 初始数据
    data1 = np.random.normal(1, 0.3, 50)
    data2 = np.random.normal(3, 1, 450)
    data3 = np.random.normal(5, 2, 500)
    d1 = np.concatenate((data1, data2, data3), axis=0).reshape(1, 1000)

    data1 = np.random.normal(2, 1, 50)
    data2 = np.random.normal(4, 0.3, 450)
    data3 = np.random.normal(6, 0.3, 500)
    d2 = np.concatenate((data1, data2, data3), axis=0).reshape(1, 1000)

    d = np.concatenate((d1, d2))

    d = d.transpose()

    gmm = GMM(n_components=3, covariance_type="full", random_state=0)
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
    plt.scatter(d[labels == 0, 0], d[labels == 0, 1], marker='o', color='g', s=30)
    plt.scatter(d[labels == 1, 0], d[labels == 1, 1], marker='o', color='b', s=30)
    plt.scatter(d[labels == 2, 0], d[labels == 2, 1], marker='o', color='y', s=30)
    plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], marker='o', color='r', s=100)
    # m, n = np.shape(d)
    # for i in range(m):
    #     plt.plot([d[i, 0], km.cluster_centers_[km.labels_[i], 0]], [d[i, 1], km.cluster_centers_[km.labels_[i], 1]],
    #              "c--", linewidth=0.3)
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covars_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)
    plt.show()

    # kmeans
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

    f1 = plt.figure(2)
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


