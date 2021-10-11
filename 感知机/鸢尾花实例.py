import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# 载入数据集
iris = load_iris()
df = pd.DataFrame(iris.data,columns=iris.feature_names)
df['label'] = iris.target
df.columns = ['sepal length','sepal width','petal length','petal width','label']

print('各类别的数据量分布：\n',df.label.value_counts())
print('部分数据展示：\n',df)

# 绘制前两类的散点图
fig = plt.figure(figsize=(8,6))

plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0 class',color = 'blue')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1 class',color = 'red')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()

# 生成样本
data = np.array(df.iloc[:100, [0, 1, -1]])
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
        self.learning_rate = 0.1

    # 激活函数
    @staticmethod
    def sign(x, w, b):
        y = np.dot(w,x) + b
        return y

    # 随机梯度下降法
    def fit(self,x_train, y_train):
        is_wrong = False
        plt.figure(figsize=(8, 6))
        plt.ion()  # 打开交互模式
        times = 0
        while not is_wrong:
            wrong_count = 0
            for d in range(len(x_train)):
                x = x_train[d]
                y = y_train[d]
                if y * self.sign(x,self.w,self.b) <= 0:
                    self.w = self.w + self.learning_rate * np.dot(y,x)
                    self.b = self.b + self.learning_rate * y
                    wrong_count += 1
            x_points = np.linspace(4, 7, 10)
            plt.cla()  # 清屏
            y_ = -(self.w[0] * x_points + self.b) / self.w[1]
            plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0 class',color = 'blue')
            plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1 class',color = 'red')
            plt.xlabel('sepal length')
            plt.ylabel('sepal width')
            plt.plot(x_points, y_,label = 'Hyperplane',color = 'purple')
            plt.legend(loc = 2)
            times += 1
            Title = 'Iterations = ' + str(times)
            plt.title(Title)
            plt.ylim([1.8,4.7])
            plt.pause(0.001)  # 屏幕刷新率：暂停 秒播放一次

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

x_points = np.linspace(4, 7, 10)
y_ = -(perceptron.w[0] * x_points + perceptron.b) / perceptron.w[1]
plt.figure(figsize=(8,6))
plt.plot(x_points, y_,label = 'Hyperplane',color = 'purple')
plt.title('Final outcomes: iterations = ' + str(times))
plt.plot(data[:50, 0], data[:50, 1], 'bo', color='blue', label='0 class')
plt.plot(data[50:100, 0], data[50:100, 1], 'bo', color='red', label='1 class')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()