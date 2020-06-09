# -*- coding:utf-8 -*-
"""
   手写最大熵模型, 实现一个简单的概率预测
   数据参考 https://www.researchgate.net/publication/338392540_zuidashangbutongtezhenghanshulizibuchong
"""
import numpy as np
import copy
import math


class MaxEntropy(object):
    def __init__(self, eps=0.01, maxiter=500, func_cnt=1):
        # 最大迭代次数
        self._maxiter = maxiter

        # 收敛判断
        self._eps = eps

        # 初始权值
        self._w = np.zeros(func_cnt)
        # self._w = np.array([math.log(16/5)])

        # 样本类别
        self._categories = []

        # 记录各样本中P'(x,y)的概率
        self._p_xy = {}

        # 记录样本中P'(x)的概率
        self._p_x = {}

        # ep1
        self._ep1 = {}

        # ep2
        self._ep2 = {}

        # z_x
        self._z_x = {}

        # p_y_x
        self._p_y_x = {}

        # sum fi
        self._sum_fi = 0

    @property
    def p_xy(self):
        """获得P'(xi, y)
        """
        return copy.copy(self._p_xy)

    @property
    def p_x(self):
        """获得P'(xi)
        """
        return copy.copy(self._p_x)

    def calc_zx_pyx(self, fc):
        """计算Zw(x)
        """
        # 根据x值，遍历y、特征函数
        for x in self._p_x.keys():
            temp1 = 0
            for y in self._categories:
                temp2 = 0
                find = False
                # 判断x, y和预设的fi(x,y)输入是否一致
                for i, item in fc.items():
                    if find:
                        break
                    for k, v in item.items():  # 满足条件则fi(x,y)=1
                        if k == x and y == v:
                            temp2 = temp2 + self._w[i-1] * 1
                            find = True
                            break
                self._p_y_x[(y, x)] = math.exp(temp2)
                temp1 = temp1 + math.exp(temp2)
            self._z_x[x] = temp1

        # print("zx: ", self._z_x)
        for key in self._p_y_x.keys():
            self._p_y_x[key] = self._p_y_x[key] / self._z_x[key[1]]
        # print("pyx: ", self._p_y_x)

    def update_w(self):
        """计算w的更新尺度delta
        """
        for i in range(0, len(self._w)):
            self._w[i] = self._w[i] + math.log(self._ep1[i] / self._ep2[i]) * 1.0 / self._sum_fi
            print(self._w)

    def calc_ep1_ep2(self, fc):
        """计算Ep'(fi) 和 Ep(fi)
        """
        # 遍历各个特征函数
        for k in fc.keys():
            if k not in self._ep1.keys():
                self._ep1[k-1] = 0.0
            if k not in self._ep2.keys():
                self._ep2[k-1] = 0.0
            items = fc[k]
            for kk in self._p_xy.keys():
                if kk[0] in items.keys() and items[kk[0]] == kk[1]:
                    self._ep1[k-1] += self._p_xy[kk]
                    self._ep2[k-1] += self._p_x[kk[0]] * self._p_y_x[(kk[1], kk[0])]

    def calc_exp_p_sumfi(self, ft, lb, fc):
        """计算P'(x,y) P'(x)
        """
        # 统计x、xy频数
        for x, y in zip(ft, lb):
            if y not in self._categories:
                self._categories.append(y)
            for xi in x:
                # (x,y)
                if (xi, y) in self._p_xy.keys():
                    self._p_xy[(xi, y)] += 1
                else:
                    self._p_xy[(xi, y)] = 1

                # (x)
                if xi in self._p_x.keys():
                    self._p_x[xi] += 1
                else:
                    self._p_x[xi] = 1

        # 求满足特征函数的fi样本累计和
        for k in fc.keys():
            items = fc[k]
            for kk in self._p_xy.keys():
                if kk[0] in items.keys() and items[kk[0]] == kk[1]:
                    self._sum_fi += self._p_xy[kk]

        n = len(list(np.array(ft).flat))

        # 计算概率P'(x,y)
        for key in self._p_xy.keys():
            self._p_xy[key] = self._p_xy[key] / n

        # 计算概率P'(x)
        for key in self._p_x.keys():
            self._p_x[key] = self._p_x[key] / n

    def fit(self, ft, lb, fc):
        """训练
        """
        # 计算经验分布P'(x,y), P'(x)
        self.calc_exp_p_sumfi(ft, lb, fc)

        for i in range(0, self._maxiter):
            # 计算zwx和pyx
            self.calc_zx_pyx(fc)

            # 计算ep1 和 ep2
            self.calc_ep1_ep2(fc)

            # 更新w
            self.update_w()

            print("s")

    def predict(self, x):
        """预测概率
        规则: 若x在观测样本中, 则按照训练生成结果的最大概率; 若x观测样本中, 则按照等概率作为结果(此时熵最大)
        """
        ret = {}

        if x in self._p_x.keys():
            for y in self._categories:
                ret[(x, y)] = round(self._p_y_x[(y, x)], 4)
        else:
            for y in self._categories:
                ret = {(x, y): round(1 / len(self._categories), 4)}

        return ret


if __name__ == "__main__":

    # 二值特征函数的条件 1表示f1 key表示条件，value表示所属类别
    function_condition = {1: {"广告": 0, "传销": 0, "学习": 1, "生活": 1}}
    # function_condition = {1: {"广告": 0, "传销": 0}, 2: {"学习": 1, "生活": 1}}

    # 特征
    features = [["广告", "传销", "广告"],
                ["广告", "传销", "传销"],
                ["学习", "生活", "生活"],
                ["学习", "学习", "生活"],
                ["广告", "学习", "学习"],
                ["传销", "学习", "生活"],
                ["广告", "传销", "学习"]]
    # 类别
    labels = [0, 0, 1, 1, 1, 0, 1]

    # maxentropy
    me = MaxEntropy(func_cnt=len(function_condition.keys()))

    # 训练
    me.fit(features, labels, function_condition)

    print("p_xy: ", me.p_xy)
    print("p_x: ", me.p_x)

    print("zx: ", me._z_x)
    print("pwyx: ", me._p_y_x)

    # 预测
    print(me.predict("广告"))
