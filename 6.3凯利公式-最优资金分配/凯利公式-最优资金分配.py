# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 14:00:34 2025

@author: zhang
"""

# CASE 6.3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from functools import reduce

'''
读取每个xls文件，只取Date和Adj Close两列
给Adj Close列名加尾缀以区分
按日期将这些dataframes合并
'''

dataframes = {}

for file in Path('.').glob("*.xls"):
    suffix = file.stem  # 去掉扩展名后的文件名
    df = pd.read_excel(file)[['Date','Adj Close']]
    df.columns = ['Date', f'Adj Close_{suffix}']  # rename
    dataframes[suffix] = df  # {key: suffix, value: df}

df = reduce(lambda left, right: pd.merge(left, right, on='Date'), dataframes.values())
df.set_index('Date', inplace=True)
df.sort_index(inplace=True)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
计算各股的Daily Excess Return
'''

r_f = 0.04
for column in df.columns:
    suffix = column.rsplit('_')[-1]
    df[f'Excess Return_{suffix}'] = df[column].pct_change(fill_method=None) - r_f/252

print(df.head())

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
创建一个新的DataFrame，存储各股的每日超额收益率
M = C F
(列向量，矩阵，列向量)
'''

excess_return = df[[col for col in df.columns if col.startswith('Excess Return_')]]

M = 252 * excess_return.mean(axis=0)  # 年化，每列求平均
print('年化平均超额收益：')
print(M)

C = 252 * excess_return.cov()  # 三列excess_return，两两求协方差
print('协方差矩阵：')
print(C)

F = np.dot(np.linalg.inv(C), M)  # F = C^(-1) M
print('最优凯利杠杆比率：')
print(F)

sharpe = np.sqrt(np.dot(F, np.dot(C, F)))  # sharpe = sqrt(F^(T)CF)
print('组合的夏普比率：', sharpe)

g = r_f + sharpe**2/2  # g = r_f + sharpe^2/2
print('组合的最大复合增长率：', g)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
求各股票单独可实现的最大增长率
'''

def get_g_indiv(name, r_f=0.04):
    '''
    sharpe = excess_return / std
    std^2 = C_ii
    '''
    sharpe = (
        M.loc[f'Excess Return_{name}'] /
        np.sqrt(C.loc[f'Excess Return_{name}', f'Excess Return_{name}'])
    )
    g = r_f + sharpe**2/2
    return g

print('OIH的最大复合增长率：', get_g_indiv('OIH'))  # 最大，但依然小于组合的g
print('RKH的最大复合增长率：', get_g_indiv('RKH'))
print('RTH的最大复合增长率：', get_g_indiv('RTH'))