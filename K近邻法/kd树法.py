import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pprint
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier

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