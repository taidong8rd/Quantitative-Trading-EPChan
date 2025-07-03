# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 17:31:07 2025

@author: zhang
"""

# CASE 7.3

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.api import OLS, add_constant
from statsmodels.tsa.stattools import coint  # 协整检验用的函数，相当于MATLAB中的cadf
from scipy.stats import pearsonr  # 相关性检验，返回相关系数和对应的pvalue(显著性检验)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
df1 = pd.read_excel("KO.xls")
df2 = pd.read_excel("PEP.xls")
df = pd.merge(df1, df2, on="Date", suffixes=("_KO", "_PEP"))  # 按日期合并，列名加后缀区分
df.set_index("Date", inplace=True)  # Date列做index
df.sort_index(inplace=True)  # 确保升序
df = df[['Adj Close_KO','Adj Close_PEP']]  # 选列
df.rename(columns={'Adj Close_KO': 'KO','Adj Close_PEP': 'PEP'}, inplace=True)

print(df.head())

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
协整检验
'''
tscore, pvalue, crit_values = coint(df['KO'], df['PEP'])

print("CADF t-statistic:", tscore)
print("p-value:", pvalue)
print("Critical Values:", crit_values)

if np.any(tscore < crit_values):  # 只要tscore小于任何一个临界值
    print("通过协整检验，序列协整")
else:
    print("未通过协整检验，序列不协整")
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
线性回归
KO = hedge_ratio * PEP
'''

model = OLS(df['KO'], df['PEP']).fit()
hedge_ratio = model.params[0]
print(hedge_ratio)

residuals = model.resid
residuals.plot()  # 非平稳

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
相关性测试
'''

daily_return = df.pct_change()
corrcoef = daily_return.corr()  # 相关系数corr_ij = cov_ij/(sigma_i*sigma_j)
print(corrcoef)  # 矩阵形式
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
使用pearsonr函数
'''

# 需去除NaN值
r, p = pearsonr(daily_return['KO'].iloc[1:], daily_return['PEP'].iloc[1:])
print(r)  # 传入的两列数的相关系数
print(p)  # 显著性检验：p < 显著性水平 -> 相关










