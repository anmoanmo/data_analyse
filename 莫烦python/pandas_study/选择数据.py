"""
=============================
ᕕ(◠ڼ◠)ᕗ
@time:2024/3/16 18:57
@IDE:PyCharm
=============================
"""
import pandas as pd
import numpy as np


datas = pd.date_range('20130101', periods=6)
df1 = pd.DataFrame(np.arange(24).reshape(6,4),index=datas,columns=['a','b','c','d'])
print(df1)
print('---------------------')
#print(df1['a'],df1.a) #选中目标列
#print(df1[0:3],df1['20130103':'20130105']) # 选中目标行
print(df1.loc["20130102"]) #按照标签选择
print(df1.loc[:,'a']) # 选中所有行“：”，再选择列
#可以采用切片的思路进行选择
print(df1.iloc[3,2])
print(df1.iloc[3:5,2:4])

"""
在 Pandas 中，有两种主要的索引方式：基于标签的索引 (loc) 和基于位置的索引 (iloc)。了解这两种索引方式的区别和注意事项非常重要。

基于标签的索引 (loc)
loc 用于通过行标签和列标签来选择数据。
行标签和列标签可以是任何数据类型，不仅限于整数。
当使用 loc 时，包括起始和结束标签在内的所有数据都会被选中。
基于位置的索引 (iloc)
iloc 用于通过行和列的位置（整数索引）来选择数据。
索引从 0 开始，类似于 Python 中的列表索引。
当使用 iloc 时，包括起始位置但不包括结束位置的数据会被选中。
注意事项
确保在使用 loc 或 iloc 时，指定的标签或位置存在于 DataFrame 中，否则会引发 KeyError 或 IndexError。
在使用切片时，loc 包含结束标签，而 iloc 不包含结束位置。
如果行标签或列标签是整数，那么在使用 loc 时要特别小心，以免与 iloc 混淆。
示例代码
python
Copy code
import pandas as pd

# 创建示例 DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 40],
    'Salary': [50000, 60000, 70000, 80000]
}
df = pd.DataFrame(data)

# 基于标签的索引
print(df.loc[0, 'Name'])  # 输出 Alice
print(df.loc[0:2, ['Name', 'Age']])  # 输出前三行的 Name 和 Age 列

# 基于位置的索引
print(df.iloc[0, 1])  # 输出 25 (第一行第二列的值)
print(df.iloc[0:2, 0:2])  # 输出前两行的前两列
"""
print('--------------------------------------')

print(df1[df1.a>8])