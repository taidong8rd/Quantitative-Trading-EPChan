# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 20:44:05 2025

@author: zhang
"""

# CASE 7.7

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
读取标准普尔500股票的收盘价数据
范围：1999-11-24 - 2007-11-23
'''

cl = pd.read_csv('SPX_20071123.txt', delim_whitespace=True)  # 收盘价

cl['Date'] = pd.to_datetime(cl['Date'], errors='coerce', format='%Y%m%d')

cl.set_index('Date', inplace=True)
cl.sort_index(inplace=True)

print(cl.head())

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
提取每个月最后一天的数据
计算月度收益率
'''

# daily_return = cl.pct_change(fill_method=None)

df = pd.DataFrame(index=cl.index)

df['Month'] = cl.index.month
df['Next Month'] = df['Month'].shift(-1)
df = df.iloc[:-1]  # 去掉最后一行2007-11-23，不然下一步会被选出（该行的'Next Month'为NaN)

# 每月最后一个交易日的日期
lastday_month = df[df['Month'] != df['Next Month']].index

# 月收益率：本月最后一天 - 上月最后一天
monthlyret = (
    (cl.loc[lastday_month] - cl.loc[lastday_month].shift(1))
    / cl.loc[lastday_month].shift(1)
)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
每月使用“去年同月”的月度收益数据对股票排序
选股建立多空组合，用当月真实收益率回测，逐月记录收益
'''

pnl_list = []

# 从第13行(2000-12-29)开始，因为需要回看一年前的同期数据，前12个月无法回测
for i, (index, row) in enumerate(monthlyret.iloc[13:].iterrows()):
    
    print(f'本期: {index.date()}, 同期: {monthlyret.index[i+1].date()}')
    
    prev_monthlyret = monthlyret.iloc[i+1]  # 跳过第0行NaN，第1行是1999-12-31，同比对应
    has_data = np.isfinite(prev_monthlyret)
    prev_monthlyret = prev_monthlyret.loc[has_data]  # 筛选有效股
    
    topN = int(np.floor(len(prev_monthlyret)/10))  # 选前10%和后10%的股，向下取整
    
    longs = prev_monthlyret.nlargest(topN).index  # 去年同期收益最好 -> 今年做多
    shorts = prev_monthlyret.nsmallest(topN).index  # 去年同期收益最差 -> 今年做空
    
    # 构建持仓向量
    positions = pd.Series(0, index=monthlyret.columns)  # 全部500股
    positions.loc[longs] = 1
    positions.loc[shorts] = -1
    
    #exposure = np.abs(positions).sum()  # = 2*topN
    
    # 计算月收益（持仓 x 当前月收益率）
    pnl = (positions * row).sum()
    pnl_list.append(pnl)


pnl_series = pd.Series(pnl_list, index=monthlyret.iloc[13:].index)

avgann_pnl = 12 * pnl_series.mean()
sharpe_ratio = np.sqrt(12) * (pnl_series.mean()/pnl_series.std())

print(f'Average annual return = {avgann_pnl}')
print(f'Sharpe ratio = {sharpe_ratio}')
    






