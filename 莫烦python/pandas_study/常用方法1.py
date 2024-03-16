"""
=============================
ᕕ(◠ڼ◠)ᕗ
@time:2024/3/16 18:54
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

# 查看 DataFrame 的前几行
print(df.head(2))  # 查看前 2 行

# 选择特定的列
print(df[['Name', 'Salary']])

# 选择特定的行和列
print(df.loc[1:2, ['Name', 'Age']])  # 基于标签的索引
print(df.iloc[1:3, 0:2])  # 基于位置的索引

# 添加新列
df['Department'] = ['HR', 'Finance', 'IT', 'Marketing']

# 删除列
df.drop(columns=['Department'], inplace=True)

# 计算统计数据
print(df.describe())  # 描述性统计摘要
print(df.mean())  # 计算每列的平均值

# 过滤数据
filtered_df = df[df['Age'] > 30]

# 分组和聚合
grouped_df = df.groupby('Age').mean()  # 按年龄分组并计算平均薪资

# 排序
sorted_df = df.sort_values(by='Age', ascending=False)

# 数据合并
df2 = pd.DataFrame({
    'Name': ['Eva', 'Frank'],
    'Age': [28, 35],
    'Salary': [55000, 65000]
})
concatenated_df = pd.concat([df, df2], ignore_index=True)  # 纵向合并

# 文件读写
df.to_csv('example.csv')  # 将 DataFrame 写入 CSV 文件
read_df = pd.read_csv('example.csv')  # 从 CSV 文件读取数据

# 显示结果
print(filtered_df)
print(grouped_df)
print(sorted_df)
print(concatenated_df)
print(read_df)

