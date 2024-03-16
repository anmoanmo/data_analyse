"""
=============================
ᕕ(◠ڼ◠)ᕗ
@time:2024/3/16 16:20
@IDE:PyCharm
=============================
"""

import numpy as np

A = np.arange(3,15).reshape(3,4)
print(A)
print(A[1,2])
print(A[1][2])
print(A[:,2])
print(A[1,1:3])

for row in A:
    print(row)
for column in A.T:
    print(column)

print('-----------------------')
print(A.flatten())

for i in A.flatten():
    print(i)