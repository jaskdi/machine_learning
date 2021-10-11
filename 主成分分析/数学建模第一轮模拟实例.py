import numpy as np
import pandas as pd
df = pd.read_excel("C:\\Users\\DELL\\DeskTop\\covid-19.xlsx")
data = df.drop(['index','location'], axis=1)
data = np.mat(data)
for i in range(5):
    data[:,i] = -1*data[:,i]

print(data)
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

# D为相关矩阵的特征值矩阵，V为相关矩阵的单位特征向量矩阵，Y为样本主成分
D,V,Y = pca_eig(data)
# 将对角的特征值矩阵化为向量
D_list = np.diagonal(D)
# 方差贡献率列向量(主成分权重向量)
f = np.mat(D_list).T
f = f/sum(f)
# 原变量权重向量
w = V*f
w = w/sum(w)
# 样本评价分数列向量
score = Y*f
print("方差贡献率矩阵为")
print(D)
print("相关矩阵的特征向量矩阵为")
print(V)
print("样本主成分矩阵为")
print(Y)
print("主成分权重向量为")
print(f)
print("原变量权重向量为")
print(w)
print("样本评价分数为")
print(score)
print()
s = 0
k = 0
for i in range(len(D_list)):
    s += D_list[i]
    t = s/sum(D_list)
    if t >= 0.75:
        print("当取前",i+1,"个主成分时，累积方差贡献率超过 0.75 ，达到",t)
        print("前k列单位特征向量为：\n",V[:,:i+1])
        k = i+1
        break

