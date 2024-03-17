"""
=============================
ᕕ(◠ڼ◠)ᕗ
@time:2024/3/16 20:46
@IDE:PyCharm
=============================
"""

import pandas as pd
import numpy as np

df = pd.DataFrame(np.arange(12).reshape(3,4)*0, columns=['A', 'B', 'C', 'D'])
df1 = pd.DataFrame(np.arange(12).reshape(3,4)*1, columns=['A', 'B', 'C', 'D'])
df2 = pd.DataFrame(np.arange(12).reshape(3,4)*2, columns=['A', 'B', 'C', 'D'])
print(df)
print(df1)
print(df2)

print('---------------------------------------')
#concatenating
res = pd.concat([df1, df2, df2], axis=0, ignore_index=True) # 竖向合并
print(res)
print('-----------------------------------------------------')
#join,[inner", 'outer']
df = pd.DataFrame(np.arange(12).reshape(3,4)*0, columns=['A', 'B', 'C', 'D'],index=[1, 2, 3])
df1 = pd.DataFrame(np.arange(12).reshape(3,4)*1, columns=['D', 'C', 'B', 'E'],index=[2, 3, 4])
res = pd.concat([df1, df2])#默认为join='outer'，不相同的部分元素为NaN
print(res)
res = pd.concat([df1, df2], join='inner',ignore_index=True)#只保留相同的部分
print(res)
print('-----------------------------------------------------------')
res = pd.concat([df, df1])
s1 = pd.Series([1, 2, 3, np.nan], index=['A', 'B', 'C', 'E'])  # 添加一个 NaN 值以匹配 'E' 列

print(res)