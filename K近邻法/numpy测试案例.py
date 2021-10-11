import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pprint

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from collections import Counter

# load toy data
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']

print(df.head())


# plot dataset

plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0',color = 'r')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1',color = 'b')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()

def distance(x, y, p = 2):
    """calculate type of P distance between two points.
    input:
        x: N*M shape array.
        y: 1*M shape array.
        p: type of distance

    output:
        N*1 shape of distance between x and y.
    """
    try:
        d = np.power(np.sum(np.power(np.abs((x - y)), p), 1), 1 / p) # axis = 1表示按行相加，但向量只有axis = 0
    except:
        d = np.power(np.sum(np.power(np.abs((x - y)), p)), 1 / p)
    return d

# X, y
data = np.array(df.iloc[:100, [0, 1, -1]])
X, y = data[:,:-1], data[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# KNN model
class KNN:
    """
    KNN implementation of violence to calculate.
    """

    def __init__(self, X_train, y_train, n_neighbors = 1, p = 2):
        """
        n_neighbors: k
                  p: type of distance
        """
        self.k = n_neighbors
        self.p = p
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X):
        dis = distance(self.X_train, X, self.p)
        dis_idx = np.argsort(dis)  # return sorted index
        top_k_idx = dis_idx[:self.k]
        top_k_points = self.X_train[top_k_idx]
        top_k_dis = dis[top_k_idx]
        top_k_y = self.y_train[top_k_idx]
        counter = Counter(top_k_y)
        label = counter.most_common()[0][0]
        return label, top_k_points, top_k_dis

    def score(self, X_test, y_test):
        right_count = 0
        for X, y in zip(X_test, y_test):
            label = self.predict(X)[0]
            if label == y:
                right_count += 1
        return right_count / len(X_test)

clf = KNN(X_train, y_train)
print("score:",clf.score(X_test, y_test)) # test on testset

test_point = [4.5, 3.5]
test_label = clf.predict(test_point)[0]
print(test_label)
plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0',color ='r')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1',color = 'b')
if test_label == 0:
    plt.plot(test_point[0], test_point[1], '.', label='test_class:0',color = 'red')
else:
    plt.plot(test_point[0], test_point[1], '.', label='test_class:1',color = 'blue')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()
