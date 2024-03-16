"""
=============================
ᕕ(◠ڼ◠)ᕗ
@time:2024/3/16 17:59
@IDE:PyCharm
=============================
"""

import pandas as pd
import numpy as np

s = pd.Series([1,2,3, np.nan,44,1],)
print(s)

datas = pd.date_range('20230316', periods=100)
df = pd.DataFrame(np.random.randn(100,4),index=datas,columns=['a','b','c','d'])
print(datas)
print(df)
df1 = pd.DataFrame(np.arange(12).reshape(3,4))
print(df1)

df2 = pd.DataFrame({'A':[1,2,3],
                    'B':pd.Timestamp('20240316'),
                    'C':'foo',
                    'D':pd.Series(1,index=list(range(1,4)),dtype='int64'),
                 })
print(df2)
print(df2.dtypes)
print(df2.index)
print(df2.columns)
print(df2.values)

print(df2.describe())
"""
df2.describe() 方法用于生成 DataFrame df2 的描述性统计摘要。这通常包括计数、平均值、标准差、
最小值、四分位数和最大值。由于 df2 中只有一列 ('D') 是数值类型，因此 describe() 方法将只针对这一列生成统计摘要。
"""

print('----------------------')
print(df2.sort_index(axis=1,ascending=False)) #按列倒序排序