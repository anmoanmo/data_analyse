# Series
Pandas Series 类似表格中的一个列（column），类似于一维数组，可以保存任何数据类型。

```text
Series 特点：
索引： 每个 Series 都有一个索引，它可以是整数、字符串、日期等类型。如果没有显式指定索引，Pandas 会自动创建一个默认的整数索引。

数据类型： Series 可以容纳不同数据类型的元素，包括整数、浮点数、字符串等。
Series 是 Pandas 中的一种基本数据结构，类似于一维数组或列表，但具有标签（索引），使得数据在处理和分析时更具灵活性。

以下是关于 Pandas 中的 Series 的详细介绍： 创建 Series： 可以使用 pd.Series() 构造函数创建一个 Series 对象，传递一个数据数组（可以是列表、NumPy 数组等）和一个可选的索引数组。

```
`pandas.Series( data, index, dtype, name, copy)`
```text
参数说明：

data：一组数据(ndarray 类型)。

index：数据索引标签，如果不指定，默认从 0 开始。

dtype：数据类型，默认会自己判断。

name：设置名称。

copy：拷贝数据，默认为 False。
```
```python
import pandas as pd

a = [1, 2, 3]

myvar = pd.Series(a)

print(myvar)
```
输出结果：
![img.png](img.png)

如果没有指定索引，索引值就从 0 开始，我们可以根据索引值读取数据：
```python
import pandas as pd

a = [1, 2, 3]

myvar = pd.Series(a)

print(myvar[1])
```
```python
import pandas as pd

a = ["Google", "Runoob", "Wiki"]

myvar = pd.Series(a, index = ["x", "y", "z"])

print(myvar)
```
我们也可以使用 key/value 对象，类似字典来创建 Series：
```python
import pandas as pd

sites = {1: "Google", 2: "Runoob", 3: "Wiki"}

myvar = pd.Series(sites)

print(myvar)
```
![img_1.png](img_1.png)

如果我们只需要字典中的一部分数据，只需要指定需要数据的索引即可，如下实例：
```python

import pandas as pd

sites = {1: "Google", 2: "Runoob", 3: "Wiki"}

myvar = pd.Series(sites, index = [1, 2])

print(myvar)
```
```python
import pandas as pd

sites = {1: "Google", 2: "Runoob", 3: "Wiki"}

myvar = pd.Series(sites, index = [1, 2], name="RUNOOB-Series-TEST" )

print(myvar)
```
![img_2.png](img_2.png)

其他操作：
```text
# 获取值
value = series[2]  # 获取索引为2的值

# 获取多个值
subset = series[1:4]  # 获取索引为1到3的值

# 使用自定义索引
value = series_with_index['b']  # 获取索引为'b'的值

# 索引和值的对应关系
for index, value in series_with_index.items():
    print(f"Index: {index}, Value: {value}")
    
# 算术运算
result = series * 2  # 所有元素乘以2

# 过滤
filtered_series = series[series > 2]  # 选择大于2的元素

# 数学函数
import numpy as np
result = np.sqrt(series)  # 对每个元素取平方根

# 获取索引
index = series_with_index.index

# 获取值数组
values = series_with_index.values

# 获取描述统计信息
stats = series_with_index.describe()

# 获取最大值和最小值的索引
max_index = series_with_index.idxmax()
min_index = series_with_index.idxmin()

```
```text
注意事项：

Series 中的数据是有序的。
可以将 Series 视为带有索引的一维数组。
索引可以是唯一的，但不是必须的。
数据可以是标量、列表、NumPy 数组等。
```
# DataFrame
DataFrame 是一个表格型的数据结构，它含有一组有序的列，每列可以是不同的值类型（数值、字符串、布尔型值）。DataFrame 既有行索引也有列索引，它可以被看做由 Series 组成的字典（共同用一个索引）。

```text
DataFrame 特点：

列和行： DataFrame 由多个列组成，每一列都有一个名称，可以看作是一个 Series。同时，DataFrame 有一个行索引，用于标识每一行。

二维结构： DataFrame 是一个二维表格，具有行和列。可以将其视为多个 Series 对象组成的字典。

列的数据类型： 不同的列可以包含不同的数据类型，例如整数、浮点数、字符串等。
```
![img_3.png](img_3.png)
![img_4.png](img_4.png)

`pandas.DataFrame( data, index, columns, dtype, copy)
`
```text
参数说明：

data：一组数据(ndarray、series, map, lists, dict 等类型)。

index：索引值，或者可以称为行标签。

columns：列标签，默认为 RangeIndex (0, 1, 2, …, n) 。

dtype：数据类型。

copy：拷贝数据，默认为 False。
```
```python
import pandas as pd

data = [['Google', 10], ['Runoob', 12], ['Wiki', 13]]

# 创建DataFrame
df = pd.DataFrame(data, columns=['Site', 'Age'])

# 使用astype方法设置每列的数据类型
df['Site'] = df['Site'].astype(str)
df['Age'] = df['Age'].astype(float)

print(df)
```
也可以使用字典来创建：
```python
import pandas as pd

data = {'Site':['Google', 'Runoob', 'Wiki'], 'Age':[10, 12, 13]}

df = pd.DataFrame(data)

print (df)
```
![img_5.png](img_5.png)


以下实例使用 ndarrays 创建，ndarray 的长度必须相同， 如果传递了 index，则索引的长度应等于数组的长度。如果没有传递索引，则默认情况下，索引将是range(n)，其中n是数组长度。
```python
import numpy as np
import pandas as pd

# 创建一个包含网站和年龄的二维ndarray
ndarray_data = np.array([
    ['Google', 10],
    ['Runoob', 12],
    ['Wiki', 13]
])

# 使用DataFrame构造函数创建数据帧
df = pd.DataFrame(ndarray_data, columns=['Site', 'Age'])

# 打印数据帧
print(df)
```
![img_6.png](img_6.png)

从以上输出结果可以知道， DataFrame 数据类型一个表格，包含 rows（行） 和 columns（列）：

![img_7.png](img_7.png)

还可以使用字典（key/value），其中字典的 key 为列名:
```python
import pandas as pd

data = [{'a': 1, 'b': 2},{'a': 5, 'b': 10, 'c': 20}]

df = pd.DataFrame(data)

print (df)
```
没有对应的部分数据为 NaN。

Pandas 可以使用 loc 属性返回指定行的数据，如果没有设置索引，第一行索引为 0，第二行索引为 1，以此类推：
```python
import pandas as pd

data = {
 "calories": [420, 380, 390],
 "duration": [50, 40, 45]
}

# 数据载入到 DataFrame 对象
df = pd.DataFrame(data)

# 返回第一行
print(df.loc[0])
# 返回第二行
print(df.loc[1])
```

**注意**：返回结果其实就是一个 Pandas Series 数据。

也可以返回多行数据，使用 [[ ... ]] 格式，... 为各行的索引，以逗号隔开：

```python
import pandas as pd

data = {
 "calories": [420, 380, 390],
 "duration": [50, 40, 45]
}

# 数据载入到 DataFrame 对象
df = pd.DataFrame(data)

# 返回第一行和第二行
print(df.loc[[0, 1]])
```
Pandas 可以使用 loc 属性返回指定索引对应到某一行：

```python
import pandas as pd

data = {
    "calories": [420, 380, 390],
    "duration": [50, 40, 45]
}

df = pd.DataFrame(data, index = ["day1", "day2", "day3"])

# 指定索引
print(df.loc["day2"])
```
```text
更多 DataFrame 说明
基本操作：

# 获取列
name_column = df['Name']

# 获取行
first_row = df.loc[0]

# 选择多列
subset = df[['Name', 'Age']]

# 过滤行
filtered_rows = df[df['Age'] > 30]
属性和方法：

# 获取列名
columns = df.columns

# 获取形状（行数和列数）
shape = df.shape

# 获取索引
index = df.index

# 获取描述统计信息
stats = df.describe()
数据操作：

# 添加新列
df['Salary'] = [50000, 60000, 70000]

# 删除列
df.drop('City', axis=1, inplace=True)

# 排序
df.sort_values(by='Age', ascending=False, inplace=True)

# 重命名列
df.rename(columns={'Name': 'Full Name'}, inplace=True)
从外部数据源创建 DataFrame：

# 从CSV文件创建 DataFrame
df_csv = pd.read_csv('example.csv')

# 从Excel文件创建 DataFrame
df_excel = pd.read_excel('example.xlsx')

# 从字典列表创建 DataFrame
data_list = [{'Name': 'Alice', 'Age': 25}, {'Name': 'Bob', 'Age': 30}]
df_from_list = pd.DataFrame(data_list)
```
```text
注意事项：

DataFrame 是一种灵活的数据结构，可以容纳不同数据类型的列。
列名和行索引可以是字符串、整数等。
DataFrame 可以通过多种方式进行数据选择、过滤、修改和分析。
通过对 DataFrame 的操作，可以进行数据清洗、转换、分析和可视化等工作。
```
# CSV文件

































































































































