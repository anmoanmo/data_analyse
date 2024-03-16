"""
=============================
ᕕ(◠ڼ◠)ᕗ
@time:2024/3/16 17:37
@IDE:PyCharm
=============================
"""
import numpy as np

A =np.arange(12).reshape(3,4)
print(A)

print(np.split(A,3,axis=0))#横向分割为3个
print(np.split(A,2,axis=1,))#纵向分割2个
print(np.array_split(A,3,axis=1))#不等分割3个（纵向）

print(np.vsplit(A,3))#横向分割为3个
print(np.hsplit(A,2))#纵向分割2个