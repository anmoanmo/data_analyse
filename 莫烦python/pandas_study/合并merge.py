"""
=============================
ᕕ(◠ڼ◠)ᕗ
@time:2024/3/16 20:46
@IDE:PyCharm
=============================
"""
import pandas as pd
import numpy as np

left = pd.DataFrame({'a':['a1','a2','a3'],
                     'b':['b1','b2','b3'],
                     'c':['c1','c2','c3'],
                     'k1':['0','1','2'],
                     'k2':['0','1','1']})

right = pd.DataFrame({'a':['a0','a2','a3'],
                      'e':['e1','e2','e3'],
                      'f':['f1','f2','f3'],
                      'k1': ['0', '1', '1'],
                      'k2': ['0', '1', '1'],
                      })
print(left)
print(right)
print('--------------------------------------')
res = pd.merge(left,right,on='a')
print(res)
res = pd.merge(left,right,on=['k1','k2'],how='inner')
print(res)