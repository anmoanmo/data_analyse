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

import pandas as pd

# 三个字段 name, site, age
nme = ["Google", "Runoob", "Taobao", "Wiki"]
st = ["www.google.com", "www.runoob.com", "www.taobao.com", "www.wikipedia.org"]
ag = [90, 40, 80, 98]

# 字典
dict = {'name': nme, 'site': st, 'age': ag}
df = pd.DataFrame(dict)

# 保存 dataframe
df.to_csv('site.csv')

re1 = pd.read_csv('site.csv')
print(re1)