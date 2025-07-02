# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 14:56:02 2025

@author: zhang
"""

# CASE 3.7

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
读取标准普尔500指数成分股数据(股价)
'''

df = pd.read_csv('spx_20071123.txt', delim_whitespace=True)  # 收盘价
#df = pd.read_csv('spx_op_20071123.txt', delim_whitespace=True)  # 开盘价

df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format='%Y%m%d')

df.set_index('Date', inplace=True)  # inplace=T：直接在原始df上修改，不返回新的
df.sort_index(inplace=True)

print(df.head())

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
基于每日的平均market return给每支股票赋予权重
'''

start = "2006-01-01"
end = "2006-12-31"

# DaraFrame of daily_return of each stock
daily_return = df.pct_change(fill_method=None)  # 禁止自动填充空值计算pct_change

# 逐行求平均，即等权重下的mkt return
# axis='index'/0(以index为轴自上往下每列求平均), 
# axis='columns'/1(以columns为轴自左往右每行求平均)
market_daily_return = daily_return.mean(axis=1)

# 求每行的有效股票数（非空->True, sum([True])=1)，sum([False])=0)
valid_count = df.notna().sum(axis=1)

# 基于相对mkt return的偏离，给每支股票赋予权重(均值回归->负权重)
# 按'index'/0方向（自上往下）填充，即每行return都去减这一行对应的mkt return
# 每一行再去除以这一行的有效股票数
excess_return = daily_return.sub(market_daily_return, axis=0)
weights = - excess_return.div(valid_count, axis=0)

# 去除无效权重
# 当天股价是NaN，或前一天股价是NaN的都无效，因为求当日收益要乘前日权重
invalid_today = df.isna()
invalid_yesterday = df.shift(1).isna()
invalid = invalid_today | invalid_yesterday
weights[invalid] = 0  # 将无效权重置为0

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
求每日收益和夏普比率
'''

# 每个股票的return * 它前一天的权重，然后每行（即每天）求和
daily_pnl = (daily_return * weights.shift(1)).sum(axis=1)

def get_sharpe_ratio(daily_pnl):    
    sharpe_ratio = (
        np.sqrt(252) * 
        (daily_pnl.mean() / daily_pnl.std())
    )
    return sharpe_ratio

sharpe_ratio = get_sharpe_ratio(daily_pnl.loc[start:end])  # 2006

print(f'sharpe ratio of 2006 is {sharpe_ratio}')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
引入交易费用
每次交易（调仓，无论买卖）时，需减去5个基点(0.05%)的交易费用（单边交易成本）
'''

tcost = 0.0005

pnl_less_tcost = (
    daily_pnl - 
    np.abs(weights - weights.shift(1)).sum(axis=1) * tcost  # 每天weights总变动量*0.05%
)

sharpe_ratio_2 = get_sharpe_ratio(pnl_less_tcost.loc[start:end])  # 2006

print(f'with one way transaction cost, sharpe ratio of 2006 is {sharpe_ratio_2}')
