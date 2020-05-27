from sklearn.datasets import load_iris
# from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree


if __name__ == "__main__":

    # 获取数据 通过sklearn自带的iris数据
    iris = load_iris()
    data = iris.data
    labels = iris.target

    # 最大深度阈值
    clf = DecisionTreeClassifier(random_state=0, max_depth=3)

    # 不纯度降低阈值
    # clf = DecisionTreeClassifier(random_state=0, min_impurity_decrease=0.05)

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)

    clf.fit(x_train, y_train)

    # 预测正确率
    print(clf.score(x_train, y_train))
    print(clf.score(x_test, y_test))

    with open("test.dot", 'w') as f:
        f = tree.export_graphviz(clf, out_file=f)
