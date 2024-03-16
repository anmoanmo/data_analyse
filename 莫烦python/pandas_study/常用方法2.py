"""
=============================
ᕕ(◠ڼ◠)ᕗ
@time:2024/3/16 18:55
@IDE:PyCharm
=============================
"""
import pandas as pd
import numpy as np

# 创建示例 DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Alice'],
    'Age': [25, 30, 35, 40, 25],
    'Salary': [50000, 60000, 70000, 80000, 50000]
}
df = pd.DataFrame(data)

# 检查缺失值
print(df.isnull())

# 删除重复行
print(df.drop_duplicates())

# 替换值
print(df.replace('Alice', 'Eve'))

# 转换数据类型
print(df['Age'].astype('float'))

# 应用函数
print(df.apply(lambda x: x['Salary'] * 1.1, axis=1))

# 重新采样时间序列数据
dates = pd.date_range('20230101', periods=6)
ts_df = pd.DataFrame(np.random.randn(6, 1), index=dates, columns=['Value'])
print(ts_df.resample('M').mean())  # 按月重新采样

# 连接 DataFrame
df2 = pd.DataFrame({'Department': ['HR', 'Finance', 'IT', 'Marketing', 'HR']})
print(df.join(df2))

# 计算列之间的相关系数
print(df.corr())

# 计算唯一值的出现次数
print(df['Name'].value_counts())

# 对分组数据应用聚合函数
print(df.groupby('Name').agg({'Salary': 'mean'}))
