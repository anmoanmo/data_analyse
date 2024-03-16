"""
=============================
ᕕ(◠ڼ◠)ᕗ
@time:2024/3/16 20:31
@IDE:PyCharm
=============================
"""
import pandas as pd
import numpy as np

datas = pd.date_range('20230316',periods=10)

df = pd.DataFrame(np.arange(60).reshape(10,6)+1,index=datas,columns=['a','b','c','d','e','f'])
df.iloc[0,1] = np.nan
df.iloc[1,2] = np.nan
df.iloc[:2,3] = np.nan
print(df)

print(df.dropna(axis=1, how = 'all')) #how = {'any' ,'all'} （一行或一列全为NaN）#丢弃存在缺失数据的行或列
#填入
print('------------------------------填入------------------------------------')
#是否存在缺失值
print(np.any(df.isnull()))
#判断元素是否为缺失值
print(df.isnull())
#将缺失值填充
print(df.fillna(value=0))
