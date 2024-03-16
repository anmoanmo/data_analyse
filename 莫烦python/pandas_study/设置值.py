"""
=============================
ᕕ(◠ڼ◠)ᕗ
@time:2024/3/16 20:24
@IDE:PyCharm
=============================
"""
import pandas as pd
import numpy as np


datas = pd.date_range('20130101', periods=6)
df1 = pd.DataFrame(np.arange(24).reshape(6,4),index=datas,columns=['a','b','c','d'])

print(df1)
df1.iloc[2,2]=24
print(df1.iloc[2,2])
df1[df1.a>0] = -df1[df1.a>0]
print(df1)
df1['c'] = pd.Series([1,2,3,4,5,6],index=datas)
print(df1)