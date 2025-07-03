# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 15:14:09 2025

@author: zhang
"""

# CASE 7.2

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.api import OLS, add_constant
from statsmodels.tsa.stattools import coint  # 协整检验用的函数，相当于MATLAB中的cadf

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
yfinance下载并保存数据
'''

tickers = ['GLD', 'GDX']
start_date = "2006-05-23"
end_date = "2008-01-19"  # 下载到2008-01-18的数据，左闭右开区间

for ticker in tickers:
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date, interval="1d", auto_adjust=False)
    data.to_csv(f"{ticker}.csv")
    print(f"{ticker}.csv saved successfully with {len(data)} rows.")
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
读取数据、合并、选列创建dataframe（GLD和GDX的调整后收盘价）
'''

df1 = pd.read_csv("GLD.csv")
df2 = pd.read_csv("GDX.csv")
df = pd.merge(df1, df2, on="Date", suffixes=("_GLD", "_GDX"))  # 按日期合并，列名加后缀区分
df.set_index("Date", inplace=True)  # Date列做index
df.sort_index(inplace=True)  # 确保升序
df = df[['Adj Close_GLD','Adj Close_GDX']]  # 选列
df.rename(columns={'Adj Close_GLD': 'GLD','Adj Close_GDX': 'GDX'}, inplace=True)

print(df.head())
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
Cointefrating Augmented Dickey-Fuller检验
使用statsmodels.tsa.stattools.coint

H0：回归残差不平稳 → 没有协整关系
H1：回归残差平稳 → 存在协整关系

tscore: 核心指标，衡量回归残差是否平稳
crit_values: 返回三个临界值，分别对应1%，5%，10%显著水平
若 tscore < 临界值 -> 拒绝原假设 -> 协整(在对应显著水平下)

pvalues: 由t推导得来，应该得出相同结论（在相同显著水平下）
'''

tscore, pvalue, crit_values = coint(df['GLD'], df['GDX'])

print("CADF t-statistic:", tscore)
print("p-value:", pvalue)
print("Critical Values:", crit_values)

if tscore < crit_values[1]:
    print("通过5%显著性水平协整检验，序列协整")
else:
    print("未通过5%显著性水平协整检验，序列不协整")
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
线性回归
GLD = hedge_ratio * GDX
'''

model = OLS(df['GLD'], df['GDX']).fit()
hedge_ratio = model.params[0]
print(hedge_ratio)

residuals = model.resid  # 残差 = GLD - hedge_ratio * GDX
residuals.plot()  # 平稳






