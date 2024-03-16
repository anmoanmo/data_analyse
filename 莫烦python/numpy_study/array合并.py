"""
=============================
ᕕ(◠ڼ◠)ᕗ
@time:2024/3/16 16:27
@IDE:PyCharm
=============================
"""
import random

import numpy as np

A = np.ones(3)
B = np.array([2, 2, 2])

C = np.vstack((A, B))# 行合并
print(C)
D = np.hstack((A, B)) # 列合并
print(D)

print(A[np.newaxis,:].shape) # 在行上面加一个维度，使列表变为1行n列的矩阵，从而可以再采用转置变为n行1列的矩阵
#原先维度
print(A.shape)
print('----------------------')
print(np.array([[1],
                [1],
                [1],
                [1]]).shape)
print('-------------------------')
print(A[np.newaxis,:].T)
print(np.vstack((B[np.newaxis, :].T, A[np.newaxis, :].T)))

"""
在 NumPy 中，np.newaxis 用于增加数组的维度。它通常用在数组的索引中。当你在数组的某个轴上使用 np.newaxis 时，它会在那个位置增加一个新的维度。

例如，假设你有一个一维数组 A：

python
Copy code
import numpy as np

A = np.array([1, 2, 3])
print(A.shape)  # 输出: (3,)
这个数组 A 的形状是 (3,)，意味着它是一个一维数组，包含 3 个元素。如果你想将这个一维数组转换为一个二维数组，其中每个原始元素都是新数组的一行，你可以使用 np.newaxis 来增加一个新的维度：

python
Copy code
A_row = A[np.newaxis, :]
print(A_row)
print(A_row.shape)  # 输出: (1, 3)
这里，A[np.newaxis, :] 创建了一个新的二维数组 A_row，其形状是 (1, 3)。这意味着现在有一个维度包含 1 个元素（即一个行），另一个维度包含 3 个元素（即三列）。

同样，你也可以使用 np.newaxis 将一维数组转换为列向量：

python
Copy code
A_col = A[:, np.newaxis]
print(A_col)
print(A_col.shape)  # 输出: (3, 1)
这里，A[:, np.newaxis] 创建了一个新的二维数组 A_col，其形状是 (3, 1)。这意味着现在有一个维度包含 3 个元素（即三行），另一个维度包含 1 个元素（即一列）。

总的来说，np.newaxis 是一个非常有用的工具，可以用来调整数组的维度，使其符合特定的操作或函数的要求。
"""
print('-------------------------------------------------------------------------------')
A = np.ones(3)[:,np.newaxis]
B = np.array([2, 2, 2])[:,np.newaxis]
C = np.concatenate((A, B, B, A),axis=1)#纵向或者横向合并（axis=0）
print(C)
