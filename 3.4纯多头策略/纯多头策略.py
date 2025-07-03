# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 13:48:46 2025

@author: zhang
"""

# CASE 3.4

import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
下载 IGE 从 2001-11-26 到 2007-11-14 的历史日线数据（注：左闭右开区间）
auto_adjust = 0/1: 是否根据分红、拆股调整价格

auto_adjust = 0:
返回open, high, low, close（这四个未调整）,
adj close（已调整）, volume, dividends, stock splits, capital gains列
index列是Date
'''

ige = yf.Ticker("IGE")
data = ige.history(start="2001-11-26", end="2007-11-15",  # [start, end)
                   interval="1d", auto_adjust=0)
data.to_csv('IGE.csv')  # save csv会把Date作为普通列来保存，不再是index

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
直接读取保存好的数据（yfinance下载）
'''

data = pd.read_csv('IGE.csv', index_col='Date')  # 指定Date列作index
data.index = pd.to_datetime(data.index, utc=True)  # str -> DatetimeIndex
print(data.columns)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
计算每日的收益率和超额收益率
应当用调整过的收盘价（已考虑拆股和分红对股价的影响，去除了股价的突变）
'''

# 每日收益率 = (今日价格-昨日价格)/昨日价格, fill_method=None:禁止自动用0替代NaN值
data['Daily Return'] = data['Adj Close'].pct_change(fill_method=None)

# 假设年化无风险利率是4%，分布在一年252个交易日中
data['Excess Return'] = data['Daily Return'] - 0.04/252

print(data.head()[['Adj Close','Daily Return','Excess Return']])

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
计算该策略（2001.11.26买入并持有一股IGE，2007.11.14卖出平仓，纯多头）
的年化夏普比率
'''

# = 平均日超额收益率/日标准差
# 年化：分子*252，分母乘sqrt（252），即整体乘sqrt（252）
sharpe_ratio = (
    np.sqrt(252)*
    (data['Excess Return'].mean()/data['Excess Return'].std())
)  

print(sharpe_ratio)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
使用Excess Return计算Cumulative return rate
'''

# cumprod会从第一个非NaN数值开始累乘
data['Cumulative Return'] = (1 + data['Excess Return']).cumprod() - 1
print(data.head()[['Daily Return','Excess Return','Cumulative Return']])

data['Cumulative Return'].plot()
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
（每个交易日结束时的）最高水位：截至当日，出现的最高的账户净值
当前账户净值创新高时，刷新最高水位，因此
账户净值 <= 最高水位
账户净值 = 本金 * (1 + 累计收益率)，本金设为1
'''

data['Net Value'] = 1 + data['Cumulative Return']

data['High Watermark'] = data['Net Value'].cummax()
print(data.head()[['Cumulative Return','Net Value','High Watermark']])

data[['Net Value','High Watermark']].plot()
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
计算回撤与最大回撤
'''

# 回撤（百分比）
data['Drawdown %'] = (
    (data['High Watermark'] - data['Net Value'])
    / data['High Watermark']
) 

# 最大回撤
max_dd = data['Drawdown %'].max()
print(f"最大回撤为：{max_dd}")

max_dd_date = data['Drawdown %'].idxmax()  # 找出最大值对应的index
print(f"发生在：{max_dd_date.date()}")  # 取出date部分（yyyy-mm-dd）

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
'''
计算回撤持续时间、最大回撤持续时间
即从高点到再创新高所用时间（天数）
回撤结束时（即 drawdown = 0），意味着出现了新的最高水位
'''

duration = 0
duration_list = [0]

# 跳过第一行NaN值
for drawdown in data['Drawdown %'].iloc[1:]:
    if drawdown == 0:
        duration = 0
    else:
        duration += 1
    duration_list.append(duration)

data['Drawdown Duration'] = duration_list  # 添加到df，记录回撤持续时间

# 最大回撤持续时间
max_drawdown_duration = data['Drawdown Duration'].max()
print(max_drawdown_duration)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
同时求最大回撤持续时间和对应的起止日期
（copy自下一个python script）
'''

duration = 0
max_duration = 0

start = None
end = None

# 跳过第一行NaN值
for index, row in data.iloc[1:].iterrows():  # 逐行遍历并返回index
    
    if row['Drawdown %'] == 0:  # 回撤没开始，或回撤结束
        duration = 0  # 重置duration
        
    else:
        duration += 1  
        if duration == 1:  # 若是回撤开始第一天，记录日期
            temp = index
        if duration > max_duration:  # 若创最大回撤持续时间记录
            max_duration = duration  # 更新记录
            start = temp  # 记录该回撤的开始日期
            end = index  # 不断更新该回撤的结束日期
        
print(f"最大回撤持续时间为：{max_duration}")
print(f"开始于{start.date()}，止于{end.date()}")

num_trading_days = len(data.loc[start:end].index)  # loc是闭区间操作，iloc左闭右开
print(f"期间共有{num_trading_days}个交易日")





