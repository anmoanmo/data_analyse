"""
=============================
ᕕ(◠ڼ◠)ᕗ
@time:2024/3/16 17:50
@IDE:PyCharm
=============================
"""
import numpy as np

a = np.arange(12)#[np.newaxis,:]
print(a)
b = a
c = a
d = b
a[0] = 24
print(a)
print(b is a)
print(d is a)
d[3] = 22
print(c)

#deep copy
e = a.copy()
print(e is a)