import numpy as np
import matplotlib.pyplot as plt
import random
# 根据原理自己写代码来测试


def distance(x, y):
    """ 计算两个向量之间的距离
    :param x: 第一个向量
    :param y: 第二个向量
    :return: 返回计算值
    """
    return np.sqrt(sum((x-y) ** 2))


def kmeans(d, k, max_iter):
    """
    :param d: 样本集
    :param k: 分类数量
    :param max_iter: 最大迭代数
    :return: 返回分类信息
    """
    m, n = np.shape(d)
    print("mxn={0}x{1}".format(m, n))
    if k >= m:
        return d

    # 用于存放初始中心向量 u[0]表示第一个中心点
    u = np.empty((k, n), dtype=np.float)

    # 初始中心向量选取过程 采用随机法
    tmp_cnt = k
    while tmp_cnt > 0:
        s_id = random.randint(0, m-1)
        sample = d[s_id]
        if sample not in u:
            tmp_cnt = tmp_cnt - 1
            u[tmp_cnt] = sample

    # 记录样本点的类别信息，与初始样本点的行相对应
    class_id = np.zeros(m, dtype=np.int)
    cur_iter = max_iter
    while cur_iter:
        cur_iter -= 1
        for i in range(0, m):
            # 遍历sample，计算每个样本点与当前所选中心的距离，取最近的作为本次类别划分
            sample = d[i]
            min_dist = distance(sample, u[0])
            sel_c = 0
            for j in range(1, k):
                tmp_dist = distance(sample, u[j])
                if tmp_dist < min_dist:
                    sel_c = j
                    min_dist = tmp_dist
            class_id[i] = sel_c

        # 计算新的中心点
        new_u = np.empty((k, n), dtype=np.float)
        for i in range(0, k):
            samples = d[class_id == i]
            # 按列求均值
            new_u[i] = np.mean(samples, axis=0)

        # 比较新老中心点的变化，并统计变化次数
        changes = 0
        for i in range(0, k):
            if np.sum(np.abs(new_u[i] - u[i])) > 0.0001:
                u[i] = new_u[i]
                changes += 1

        # 如果change=0，则表示没有变化
        if changes == 0:
            return u, class_id, max_iter - cur_iter

    # 迭代完成后返回
    return u, class_id, max_iter


if __name__ == "__main__":
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

    print(d)

    u, c, iter_cnt = kmeans(d, 3, 100)

    # 中心点
    print("*******样本中心*********")
    print(u)

    # 类别样本
    print("*******样本类别集合索引*********")
    print(c)

    # 计算迭代次数
    print("*******迭代次数*********")
    print(iter_cnt)

    f1 = plt.figure(1)
    plt.title("test kmeans")
    plt.xlabel("dimension_1")
    plt.ylabel("dimension_2")
    plt.scatter(d[:, 0], d[:, 1], marker='o', color='g', s=50)
    plt.scatter(u[:, 0], u[:, 1], marker='o', color='r', s=200)
    m, n = np.shape(d)
    for i in range(m):
        plt.plot([d[i, 0], u[c[i], 0]], [d[i, 1], u[c[i], 1]], "c--", linewidth=0.3)
    plt.show()



