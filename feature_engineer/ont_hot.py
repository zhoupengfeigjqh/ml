# -*- coding:utf-8 -*-

from sklearn import preprocessing

enc = preprocessing.OneHotEncoder()
enc.fit([[1, 0, 3, 0], [0, 1, 0, 1], [0, 2, 1, 1], [0, 1, 2, 3]])  # 二维数组，4个特征

array = enc.transform([[0, 2, 3, 1]]).toarray()  # 测试

print(array)

