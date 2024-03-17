"""
=============================
ᕕ(◠ڼ◠)ᕗ
@time:2024/3/16 20:45
@IDE:PyCharm
=============================
"""
import numpy as np
import pandas as pd


datas = pd.read_excel('计算机学院2023-2024学年第一学期奖学金名单(1).xlsx', skiprows=3)
#print(datas.columns)
#print(datas.loc[:72, '序号':'获奖金额（元）'])
datas = datas.loc[:72, '序号':'获奖金额（元）']
datas.to_csv('奖学金·2023-2024学年秋.csv')
#

datas = pd.read_csv('奖学金·2023-2024学年秋.csv')
print(datas)