# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 20:35:32 2025

@author: zhang
"""

# CASE 7.5

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.api import OLS

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
同CASE 7.2的代码
线性回归GLD和GDX的价格 (GLD = hedge_ratio * GDX + resid)
得出回归残差resid，用以计算其均值回归的半衰期
'''

df1 = pd.read_csv("GLD.csv")
df2 = pd.read_csv("GDX.csv")
df = pd.merge(df1, df2, on="Date", suffixes=("_GLD", "_GDX"))  # 按日期合并，列名加后缀区分
df.set_index("Date", inplace=True)  # Date列做index
df.sort_index(inplace=True)  # 确保升序
df = df[['Adj Close_GLD','Adj Close_GDX']]  # 选列
df.rename(columns={'Adj Close_GLD': 'GLD','Adj Close_GDX': 'GDX'}, inplace=True)

model1 = OLS(df['GLD'], df['GDX']).fit()
# hedge_ratio = model1.params.iloc[0]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
dz_t = z_t - z_(t-1) = theta * (z_(t-1) - z_mean) + noise
半衰期 = -log(2)/theta
线性回归求 theta (回归强度)
半衰期：价差从其均值偏离最大程度向均值回归一半所需的预期时间
半衰期小 -> 回归快 -> 可套利
半衰期长 -> 不易回归 -> 谨慎交易
实际交易中，无需每日平仓，可持仓至价差回归 (由zscore衡量)
半衰期可用于参考，设置最长持续持仓时间（e.g. 1.5~2 倍半衰期，避免资金拖累、模型失效）
'''

z = model1.resid  # 获得GLD和GDX价格的回归残差（价差）, z_t
prevz = z.shift(1)  # z_(t-1)
dz = z - prevz  # z_t - z_(t-1)

# 去掉首行NaN
prevz = prevz.iloc[1:]  
dz = dz.iloc[1:]

z_mean = prevz.mean()

model2 = OLS(dz, (prevz - z_mean)).fit()  # 无需add_constant(X)

theta = model2.params.iloc[0]  # 用了整个时间序列来决定最优的theta

halflife = -np.log(2)/theta  # days

print(f'halflife = {halflife}')









