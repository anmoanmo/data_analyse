"""
=============================
ᕕ(◠ڼ◠)ᕗ
@time:2024/3/16 18:56
@IDE:PyCharm
=============================
"""
import pandas as pd
import numpy as np

# 创建示例 DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 40],
    'Salary': [50000, 60000, 70000, 80000]
}
df = pd.DataFrame(data)

# 根据表达式筛选数据
print(df.query('Age > 30'))

# 随机抽取 2 行数据
print(df.sample(2))

# 返回按年龄排序的前 2 行数据
print(df.nlargest(2, 'Age'))

# 返回按薪水排序的后 2 行数据
print(df.nsmallest(2, 'Salary'))

# 将分类变量转换为虚拟/指示变量
print(pd.get_dummies(df['Name']))

# 将薪水限制在 55000 到 75000 范围内
print(df['Salary'].clip(55000, 75000))

# 计算年龄的差异
print(df['Age'].diff())

# 创建条形图
df.plot.bar(x='Name', y='Salary')

# 创建箱形图
df.plot.box()

# 创建散点图
df.plot.scatter(x='Age', y='Salary')

