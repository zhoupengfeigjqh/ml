# -*- coding:utf-8 -*-
"""naive bayes test
   基于sklearn实现 多项式和伯努利朴素贝叶斯
"""

import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB


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
    """根据词汇表和输入文档，生成文档向量
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
    """根据词汇表和输入文档，生成文档向量
       采用的是累计值形式，若重复出现则累加，向量形式为[0,1,3,2,0,0..]
    """
    return_vec = [0] * len(vocablist)
    for word in inputset:
        if word in vocablist:
            return_vec[vocablist.index(word)] += 1
        else:
            print("the word: %s is not in my vocabulary" % word)
    return return_vec


if __name__ == "__main__":

    mnb = MultinomialNB()
    # mnb = BernoulliNB()

    data, categary_vec = load_data_set()
    my_vocablist = create_vocab_list(data)

    data_vec = []

    for doc in data:
        data_vec.append(set_of_word2vec_mult(my_vocablist, doc))
        # data_vec.append(set_of_word2vec_bn(my_vocablist, doc))

    data_vec = np.array(data_vec)
    categary_vec = np.array(categary_vec)

    mnb.fit(data_vec, categary_vec)

    print(mnb.class_log_prior_, '\r\n')
    print(mnb.feature_count_, '\r\n')
    print(mnb.feature_log_prob_, '\r\n')

    # 代入预测值
    test_words = ['my', 'love', 'worthless']

    words_vec = set_of_word2vec_mult(my_vocablist, test_words)

    print(mnb.predict(np.array([words_vec])))

