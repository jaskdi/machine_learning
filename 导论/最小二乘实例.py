
# 使用最小二乘法拟合正弦函数

import numpy as np
import scipy as sp
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

# 目标函数
def real_func(x):
    return np.sin(2*np.pi*x)

# 多项式
def fit_func(p,x):
    f = np.poly1d(p)
    return f(x)

# 残差
def residuals_func(p,x,y):
    ret = fit_func(p,x) - y
    return ret

# 生成目标函数\生成服从正态噪声的样本
x_points = np.linspace(0,1,1000)
N= 10 # 数据点个数
x = np.linspace(0,1,N)
y_real = real_func(x)
noise = 0.1 # 高斯噪声的大小
y = [np.random.normal(0,noise) + y1 for y1 in y_real]

# M为多项式的次数
def fitting(M = 0):
    # 随机初始化多项式参数
    p_init = np.random.rand(M+1)
    # 最小二乘
    p_lsq = leastsq(residuals_func,p_init,args=(x,y))
    print("Fitting Parameters:",p_lsq)
    return p_lsq


plt.ion() # 打开交互模式
for i in range(N):

    fitted_label = "fitted curve M = " + str(i)

    # 可视化
    plt.cla() # 清屏
    # plt.plot(x_points,real_func(x_points),label = "real")
    plt.plot(x_points,fit_func(fitting(i)[0],x_points),label = fitted_label)
    plt.plot(x,y,'bo',label = 'noise')
    plt.legend()

    plt.pause(0.2)  # 暂停 秒播放一次

plt.ioff()  # 关闭交互模式
plt.show()

# M = N-1 时 fitted curve 通过了所有数据点,但曲线过拟合

# 引入正则化
# 经验风险函数与正则化项之间的调整系数lambda
regularization = 0.0001

# 计算引入正则化后的残差
def residuals_func_regularization(p,x,y):
    ret = fit_func(p,x) - y
    ret = np.append(ret,np.sqrt(0.5*regularization*np.square(p))) # 正则化项为参数向量的L2范数(ridge)
    return ret

# 最小二乘法,加正则化项
p_init = np.random.rand(N + 1)
p_lsq_regularization = leastsq(residuals_func_regularization,p_init,args=(x, y))

# plt.plot(x_points, real_func(x_points), label='real')
plt.plot(x_points, fit_func(fitting(N-1)[0], x_points), label='fitted curve')
plt.plot(x_points,fit_func(p_lsq_regularization[0], x_points),'g',label='regularization')
plt.plot(x, y, 'bo', label='noise')
plt.legend()
plt.show()



