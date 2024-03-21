"""
=============================
ᕕ(◠ڼ◠)ᕗ
@time:2024/3/19 14:58
@IDE:PyCharm
=============================
"""

# import pandas as pd
#
# df = pd.read_csv('nba.csv')
#
# print(df.to_string())
#
# with pd.option_context('display.max_columns', None, 'display.max_rows', None):
#     print(df[df['Name'] == 'Kobe Bryant'])

# import pandas as pd
#
# # 三个字段 name, site, age
# nme = ["Google", "Runoob", "Taobao", "Wiki"]
# st = ["www.google.com", "www.runoob.com", "www.taobao.com", "www.wikipedia.org"]
# ag = [90, 40, 80, 98]
#
# # 字典
# dict = {'name': nme, 'site': st, 'age': ag}
# df = pd.DataFrame(dict)
#
# # 保存 dataframe
# df.to_csv('site.csv')
#
# re1 = pd.read_csv('site.csv')
# print(re1)

# import pandas as pd
#
# df = pd.read_json('nested_list.json')
#
# print(df.to_string())


# import pandas as pd
# from glom import glom
#
# df = pd.read_json('nest_deep.json')
#
# data = df['students'].apply(lambda row: glom(row, 'grade.math'))
# print(data)

# import pandas as pd
#
# df = pd.read_csv('property-data.csv')
#
# print(df.to_string())

# import pandas as pd
#
# # 创建一个示例数据框
# data = {'A': [1, 2, 3, 4, 5], 'B': [5, 4, 3, 2, 1]}
# df = pd.DataFrame(data)
#
# # 计算 Pearson 相关系数
# correlation_matrix = df.corr()
# print(correlation_matrix)

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 创建一个示例数据框
data = {'A': [1, 2, 3, 4, 5], 'B': [5, 4, 3, 2, 1]}
df = pd.DataFrame(data)

# 计算 Pearson 相关系数
correlation_matrix = df.corr()
# 使用热图可视化 Pearson 相关系数
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.show()