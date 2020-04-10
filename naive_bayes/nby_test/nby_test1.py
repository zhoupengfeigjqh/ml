# -*- coding:utf-8 -*-
"""naive bayes test
   参考<<机器学习实战>>，手写模型(包括伯努利和多项式)，实现一个简单文本分类
"""

import numpy as np


def load_data_set():
    """加载数据
    """
    data = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'hims'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]

    categary_vec = [0, 1, 0, 1, 0, 1]  # 1代表侮辱文字，0代表正常言论

    return data, categary_vec


def create_vocab_list(data):
    """创建文本单词集，包含文本所有单词且不重复的集合
    """
    vocab_set = set([])  # 无序、不重复
    for doc in data:
        vocab_set = vocab_set.union(set(doc))

    return list(vocab_set)


def set_of_word2vec_bn(vocablist, inputset):
    """伯努利
       根据词汇表和输入文档，生成文档向量
       采用的是伯努利方式，若单词出现在文档中，则记录为1，反之为0，向量形式为[0,1,1,0,...]
    """
    return_vec = [0] * len(vocablist)
    for word in inputset:
        if word in vocablist:
            return_vec[vocablist.index(word)] = 1
        else:
            print("the word: %s is not in my vocabulary" % word)
    return return_vec


def set_of_word2vec_mult(vocablist, inputset):
    """多项式
       根据词汇表和输入文档，生成文档向量
       采用的是累计值形式，若重复出现则累加，向量形式为[0,1,3,2,0,0..]
    """
    return_vec = [0] * len(vocablist)
    for word in inputset:
        if word in vocablist:
            return_vec[vocablist.index(word)] += 1
        else:
            print("the word: %s is not in my vocabulary" % word)
    return return_vec


def calc_bn(data, categary):
    """伯努利
       公式 p(c=i|w1,w2,w3...) = p(w1,w2,w3...|c=i)p(c)/p(w1,w2,w3...)
       p(w1,w2,w3...|c=i)=p(w1|c=i)*p(w2|c=i)*p(w3|c=i)...
       此处需要计算p(wi|c), p(c)的值，同时为避免出现p(wi|c)=0的情况，我们引入了拉普拉斯平滑
    """
    # 先验概率p0=p(c=0) 和 p1=p(c=1)值 p(c=1)=(p(c=1)+1)/(N+2)
    p0 = (sum(categary) + 1) / ((len(categary) * 1.0) + len(np.unique(categary)))
    p1 = 1 - p0

    # 记录各类别下各个单词的频数，初始化为1
    pc0 = np.ones(len(data[0]))
    pc1 = np.ones(len(data[0]))

    # 记录各类别下的单词总数，初始化len(np.unique(categary))
    init_sum0 = len(np.unique(categary))
    init_sum1 = len(np.unique(categary))

    for i in range(0, len(categary)):
        if categary[i] == 1:
            pc1 += data[i]
            init_sum1 += 1

        if categary[i] == 0:
            pc0 += data[i]
            init_sum0 += 1

    # 记录条件概率值 p(w1|c=i)
    pc0 = pc0 * 1.0 / init_sum0
    pc1 = pc1 * 1.0 / init_sum1

    # 为了避免数值下溢，加上了log处理pc0和pc1
    pc0 = np.log(pc0)
    pc1 = np.log(pc1)

    p0 = np.log(p0)
    p1 = np.log(p1)
    return p0, p1, pc0, pc1


def calc_mult(data, categary):
    """多项式
       公式 p(c=i|w1,w2,w3...) = p(w1,w2,w3...|c=i)p(c)/p(w1,w2,w3...)
       p(w1,w2,w3...|c=i)=p(w1|c=i)*p(w2|c=i)*p(w3|c=i)...
       此处需要计算p(wi|c), p(c)的值，同时为避免出现p(wi|c)=0的情况，我们引入了拉普拉斯平滑
    """
    # 先验概率p0=p(c=0) 和 p1=p(c=1)值 p(c=1)=(p(c=1)+1)/(N+2)
    p0 = (sum(categary) + 1) / ((len(categary) * 1.0) + len(np.unique(categary)))
    p1 = 1 - p0

    # 记录各类别下各个单词的频数，初始化为1
    pc0 = np.ones(len(data[0]))
    pc1 = np.ones(len(data[0]))

    # 记录各类别下的单词总数，初始化为len(data[0])
    init_sum0 = 0 + len(data[0])
    init_sum1 = 0 + len(data[0])

    for i in range(0, len(categary)):
        if categary[i] == 1:
            pc1 += data[i]
            init_sum1 += sum(data[i])

        if categary[i] == 0:
            pc0 += data[i]
            init_sum0 += sum(data[i])

    # 记录条件概率值 p(w1|c=i)
    pc0 = pc0 * 1.0 / init_sum0
    pc1 = pc1 * 1.0 / init_sum1

    # 为了避免数值下溢，加上了log处理pc0和pc1
    pc0 = np.log(pc0)
    pc1 = np.log(pc1)

    p0 = np.log(p0)
    p1 = np.log(p1)
    return p0, p1, pc0, pc1


def classify_naive_bayes(data, p0, p1, pc0, pc1):
    """分类预测
    """
    p1c = sum(pc1 * data) + p1
    p0c = sum(pc0 * data) + p0

    if p1c > p0c:
        return 1
    else:
        return 0


if __name__ == "__main__":
    data, categary_vec = load_data_set()
    my_vocablist = create_vocab_list(data)

    data_vec = []

    for doc in data:
        data_vec.append(set_of_word2vec_mult(my_vocablist, doc))

    data_vec = np.array(data_vec)
    categary_vec = np.array(categary_vec)

    p0, p1, pc0, pc1 = calc_mult(data_vec, categary_vec)

    print(p0, '\r\n', p1)
    print(pc0, '\r\n', pc1)

    # 代入预测值
    test_words = ['my', 'stupid', 'worthless', 'love', 'love', 'love']

    words_vec = set_of_word2vec_mult(my_vocablist, test_words)

    print(classify_naive_bayes(words_vec, p0, p1, pc0, pc1))
