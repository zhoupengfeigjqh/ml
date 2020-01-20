import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GMM
# 利用sklearn工具包进行测试练习

if __name__ == "__main__":
    print("sss")
    # 初始数据
    data1 = np.random.normal(1, 0.3, 30)
    data2 = np.random.normal(3, 1, 30)
    data3 = np.random.normal(5, 2, 30)
    d1 = np.concatenate((data1, data2, data3), axis=0).reshape(1, 90)

    data1 = np.random.normal(2, 0.5, 30)
    data2 = np.random.normal(4, 0.7, 30)
    data3 = np.random.normal(6, 0.3, 30)
    d2 = np.concatenate((data1, data2, data3), axis=0).reshape(1, 90)

    d = np.concatenate((d1, d2))

    d = d.transpose()

    gmm = GMM(n_components=3, covariance_type="full", random_state=0)
    print(gmm.fit(d).predict(d))

    # print("*******样本中心*********")
    # print(gmm.cluster_centers_)
    #
    # print("*******样本类别*********")
    # print(gmm.labels_)
    #
    # f1 = plt.figure(1)
    # plt.title("test scikit_learn kmeans")
    # plt.xlabel("dimension_1")
    # plt.ylabel("dimension_2")
    # plt.scatter(d[:, 0], d[:, 1], marker='o', color='g', s=50)
    # plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], marker='o', color='r', s=200)
    # m, n = np.shape(d)
    # for i in range(m):
    #     plt.plot([d[i, 0], km.cluster_centers_[km.labels_[i], 0]], [d[i, 1], km.cluster_centers_[km.labels_[i], 1]],
    #              "c--", linewidth=0.3)
    # plt.show()

