"""
=============================
ᕕ(◠ڼ◠)ᕗ
@time:2024/3/15 14:10
@IDE:PyCharm
=============================
"""
import random
import numpy as np

np1 = np.array([[1, 2, 3],
                [2, 4, 5]])
print(np1)
print(np1.ndim) # 维度
print(np1.shape) # 形状
print(np1.size) # 大小

print('----------------------------------------------------------------------')

np2 = np.array([[1, 2, 3]],dtype=np.int64)
print(np2.dtype)

print(np2)
a =np.zeros((3, 4)) #元素全为0 的矩阵
b = np.ones((4, 4)) # 元素全为1
c = np.empty((3, 4))
d = np.arange(16).reshape((4, 4))+1  # 创建范围【0，16）的行矩阵之后转换为4*4的矩阵，每个元素加1
print('------------------------------')
print(b.dot(d)) # dot(b, d) 矩阵乘法
print(b * d) # 元素相乘

e = np.linspace(1,16,4).reshape((2, 2)) # 创建范围【1，16】，分为4段的行矩阵，重塑

print(a)
print(b)
print(c)
print(d)

print(e)

# 创建随机生成的矩阵
np3 = np.random.random((3,4))

print(np3)
print(np.sum(np3, axis=1)) # 每一行的sum
print(np.max(np3, axis=0)) # 每一列的最大值
print(np.min(np3)) # 整个矩阵的最小值

print("***********************next*****************************")

A = np.arange(13, 1,-1).reshape(3, 4)

print(A)
print(np.argmax(A))#矩阵最大的元素索引
print(np.argmin(A))#最小的
print(np.mean(A))#平均值
print(np.average(A))#平均值
print(np.median(A))#求中位数
print(np.cumsum(A)) #累加
print(np.diff(A))# 后一项减去前一项
print(np.nonzero(A))

print(np.sort(A))#每行从小到大排序

# 转置
print(np.transpose(A))
print(A.T)

print(np.clip(A,a_min=5,a_max=9)) #所有大于9的元素等于9，小于5的元素等于5
