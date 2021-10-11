import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pprint
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier


def distance(x, y, p=2):
    """calculate type of P distance between two points.
    input:
        x: N*M shape array.
        y: 1*M shape array.
        p: type of distance

    output:
        N*1 shape of distance between x and y.
    """
    try:
        d = np.power(np.sum(np.power(np.abs((x - y)), p), 1), 1 / p)  # axis = 1表示按行相加，但向量只有axis = 0
    except:
        d = np.power(np.sum(np.power(np.abs((x - y)), p)), 1 / p)
    return d


# load toy data
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']

print(df.head())

# X, y
data = np.array(df.iloc[:100, [0, 1, -1]])
X, y = data[:,:-1], data[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf_sk = KNeighborsClassifier()
print(clf_sk.fit(X_train, y_train))
print(clf_sk.score(X_test, y_test))

# 构建kd树
class KdTree:
    """
    build kdtree recursively along axis, split on median point.
    k:      k dimensions
    method: alternate/variance, 坐标轴轮替或最大方差轴
    """

    def __init__(self, k=2, method='alternate'):
        self.k = k
        self.method = method

    def build(self, points, depth=0):
        n = len(points)
        if n <= 0:
            return None

        if self.method == 'alternate':
            axis = depth % self.k
        elif self.method == 'variance':
            axis = np.argmax(np.var(points, axis=0), axis=0)

        sorted_points = sorted(points, key=lambda point: point[axis])

        return {
            'point': sorted_points[n // 2],
            'left': self.build(sorted_points[:n // 2], depth + 1),
            'right': self.build(sorted_points[n // 2 + 1:], depth + 1)
        }


data = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])

kd1 = KdTree(k=2, method='alternate')
tree1 = kd1.build(data)

kd2 = KdTree(k=2, method='variance')
tree2 = kd2.build(data)

# friendly print

pp = pprint.PrettyPrinter(indent=4)
pp.pprint(tree1)  # equal to figure. 3.4 《统计学习方法》

pp.pprint(tree2)  # 在该数据集上两种方法结果一样


# 查找kd树

class SearchKdTree:
    """
    search closest point
    """

    def __init__(self, k=2):
        self.k = k

    def __closer_distance(self, pivot, p1, p2):
        if p1 is None:
            return p2
        if p2 is None:
            return p1

        d1 = distance(pivot, p1)
        d2 = distance(pivot, p2)

        if d1 < d2:
            return p1
        else:
            return p2

    def fit(self, root, point, depth=0):
        if root is None:
            return None

        axis = depth % self.k

        next_branch = None
        opposite_branch = None

        if point[axis] < root['point'][axis]:
            next_branch = root['left']
            opposite_branch = root['right']
        else:
            next_branch = root['right']
            opposite_branch = root['left']

        best = self.__closer_distance(point,
                                      self.fit(next_branch,
                                               point,
                                               depth + 1),
                                      root['point'])

        if distance(point, best) > abs(point[axis] - root['point'][axis]):
            best = self.__closer_distance(point,
                                          self.fit(opposite_branch,
                                                   point,
                                                   depth + 1),
                                          best)

        return best


# test
point_ = [3., 4.5]

search = SearchKdTree()
best = search.fit(tree1, point_, depth=0)
print(best)


# force computing
def force(points, point):
    dis = np.power(np.sum(np.power(np.abs((points - point)), 2), 1), 1 / 2)
    idx = np.argmin(dis, axis=0)
    return points[idx]


print(force(data, point_))
