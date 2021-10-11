import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from matplotlib import cm

# 载入数据集
iris = load_iris()
df = pd.DataFrame(iris.data,columns=iris.feature_names)
df['label'] = iris.target
df.columns = ['sepal length','sepal width','petal length','petal width','label']

print('各类别的数据量分布：\n',df.label.value_counts())
print('部分数据展示：\n',df)

# 绘制前三类的散点图


fig1 = plt.figure(figsize=(8,6))
ax1 = fig1.gca(projection='3d')
ax1.scatter(df[:50]['sepal length'], df[:50]['sepal width'],df[:50]['petal length'], label='0 class',color = 'blue')
ax1.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'],df[50:100]['petal length'], label='1 class',color = 'red')
ax1.set_zlabel('sepal length', fontdict={'size': 10, 'color': 'k'})
ax1.set_ylabel('sepal width', fontdict={'size': 10, 'color': 'k'})
ax1.set_xlabel('petal length', fontdict={'size': 10, 'color': 'k'})

plt.legend()

plt.show()

# 生成样本
data = np.array(df.iloc[:100, [0, 1, 2, -1]])
X_train, Y = data[:,:-1], data[:,-1]
Y_train = np.array([1 if i == 1 else -1 for i in Y])
print('训练数据X：\n',X_train)
print('训练数据Y：\n',Y_train)
print(np.ones(len(data[0])-1))

# 感知机模型
class Model:
    def __init__(self):
        self.w = np.ones(len(data[0]) - 1,dtype = np.float32)  # [1 1]
        self.b = 0
        self.learning_rate = 0.00008

    # 激活函数
    @staticmethod
    def sign(x, w, b):
        y = np.dot(w, x) + b
        return y

    # 随机梯度下降法
    def fit(self, x_train, y_train):
        is_wrong = False
        fig2 = plt.figure(figsize=(8, 6))
        plt.ion()  # 打开交互模式
        times = 0
        while not is_wrong:
            wrong_count = 0

            for d in range(len(x_train)):
                x = x_train[d]
                y = y_train[d]
                if y * self.sign(x, self.w, self.b) <= 0:
                    self.w = self.w + self.learning_rate * np.dot(y, x)
                    self.b = self.b + self.learning_rate * y
                    wrong_count += 1
            plt.cla()  # 清屏
            x_points = np.linspace(1, 10, 20)
            y_points = np.linspace(1, 10, 20)
            X_, Y_ = np.meshgrid(x_points, y_points)
            Z_ = -(perceptron.w[0] * X_ + perceptron.w[1] * Y_ + perceptron.b) / perceptron.w[2]
            ax2 = fig2.gca(projection='3d')
            ax2.scatter(df[:50]['sepal length'], df[:50]['sepal width'], df[:50]['petal length'], label='0 class',
                        color='blue')
            ax2.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], df[50:100]['petal length'],
                        label='1 class', color='red')
            ax2.set_zlabel('sepal length', fontdict={'size': 10, 'color': 'k'})
            ax2.set_ylabel('sepal width', fontdict={'size': 10, 'color': 'k'})
            ax2.set_xlabel('petal length', fontdict={'size': 10, 'color': 'k'})
            surf = ax2.plot_surface(X_, Y_, Z_, color='purple', linewidth=0, antialiased=False, alpha=0.2,label = 'Hyperplane')
            surf._facecolors2d = surf._facecolor3d
            surf._edgecolors2d = surf._edgecolor3d  # 解决2维与3维绘图的图例不兼容问题
            ax2.set_zlim(0,5)
            times += 1
            Title = 'Iterations = ' + str(times)
            ax2.set_title(Title)
            plt.legend()
            plt.pause(0.01)  # 屏幕刷新率：暂停 秒播放一次
            if wrong_count == 0:
                is_wrong = True
                plt.ioff()
                plt.show()
        return times

    def score(self):
        pass

# 生成模型,训练数据
perceptron = Model()
times = perceptron.fit(X_train,Y_train)
print('w=',perceptron.w,'b=',perceptron.b)


x_points = np.linspace(1,10,20)
y_points = np.linspace(1,10,20)
X_,Y_ = np.meshgrid(x_points,y_points)
Z_ = -(perceptron.w[0] * X_ + perceptron.w[1] * Y_ + perceptron.b) / perceptron.w[2]

fig3 = plt.figure(figsize=(8,6))
ax3 = fig3.gca(projection='3d')
Title = 'Final outcomes: iterations = ' + str(times)
ax3.set_title(Title)
ax3.scatter(df[:50]['sepal length'], df[:50]['sepal width'],df[:50]['petal length'], label='0 class',color = 'blue')
ax3.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'],df[50:100]['petal length'], label='1 class',color = 'red')
ax3.set_zlabel('sepal length', fontdict={'size': 10, 'color': 'k'})
ax3.set_ylabel('sepal width', fontdict={'size': 10, 'color': 'k'})
ax3.set_xlabel('petal length', fontdict={'size': 10, 'color': 'k'})

surf = ax3.plot_surface(X_, Y_, Z_,color='purple', linewidth=0, antialiased=False,alpha = 0.2,label = 'Hyperplane')
surf._facecolors2d = surf._facecolor3d
surf._edgecolors2d = surf._edgecolor3d  # 解决2维与3维绘图的图例不兼容问题
plt.legend()
plt.show()




