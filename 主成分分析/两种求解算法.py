import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# 载入数据
data = loadmat("C:\\Users\\DELL\\Downloads\\ex7data1.mat")

# 可视化数据
X = data['X']
fig,ax = plt.subplots(figsize = (12,8))
ax.scatter(X[:,0],X[:,1])
# plt.show()

# 相关矩阵的谱分解算法
def pca_eig(x):
    # 数据标准化
    x_norm = (x-np.ones((x.shape[0],1),dtype = int)*np.mean(x,axis=0))/(np.ones((x.shape[0],1),dtype = int)*np.std(x,axis=0))

    # 计算相关矩阵
    x_norm = np.mat(x_norm)
    cov = (x_norm.T * x_norm) / (x.shape[0]-1)

    # 相关矩阵的谱分解
    d, v = np.linalg.eig(cov)
    d = np.diag(d)
    y = x_norm * np.mat(v)
    return d,v,y

# 数据矩阵的奇异值分解算法
def pca_svd(x):
    # 数据标准化
    x_norm = (x-np.ones((x.shape[0],1),dtype = int)*np.mean(x,axis=0))/(np.ones((x.shape[0],1),dtype = int)*np.std(x,axis=0))

    # 定义新矩阵x_norm_,满足x_norm_.T*x_norm_ = cov(x)
    x_norm_ = np.mat(x_norm)/(np.sqrt(np.mat(x_norm).shape[0]-1))

    # 奇异值分解
    # d对角线上的元素λ即为cov(x)的特征值的平方根,v的列向量为λ对应的特征向量
    u,d,v = np.linalg.svd(x_norm_)
    d = np.diag(d)

    # 求样本主成分矩阵
    y = x_norm * np.mat(v)
    return u,d,v,y

D1,V1,Y1 = pca_eig(X)
print(D1)
print(V1)
print(Y1)
U2,D2,V2,Y2 = pca_svd(X)
print(U2.shape)
print(D2)
print(V2)
print(Y2)
## Y1和Y2仅在方向上不同，绝对值相等





