# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 10:52:13 2025

@author: zhang
"""

# CASE 3.5

import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
多空市场价值中性策略
2001.11.26买入IGE，同时做空同样资金规模SPY作为对冲
2007.11.14将多空双方的头寸平仓
'''

# 使用yfinance下载数据
tickers = ['IGE', 'SPY']
start_date = "2001-11-26"
end_date = "2007-11-14"

for ticker in tickers:
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date, interval="1d", auto_adjust=False)
    data.index.name = 'Date'  # 确保索引列被命名为'Date'再保存

    if data.empty:
        print(f"{ticker}: Data is empty after filtering. Check symbol or date range.")
    else:
        data.to_csv(f"{ticker}.csv")  # 保存为csv
        print(f"{ticker}.csv saved successfully with {len(data)} rows.")
        
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
直接读取保存好的数据（yfinance下载）
'''

# 将'Date'列作为索引
ige = pd.read_csv("IGE.csv", index_col='Date')
spy = pd.read_csv("SPY.csv", index_col='Date')

# 选列组成新DataFrame
data = pd.DataFrame({
    'Close_IGE': ige['Adj Close'],  # Series自带索引（'Date'）
    'Close_SPY': spy['Adj Close']  # pd会自动对齐这些索引，取并集作为新df的索引（默认升序）
})

# str -> DatetimeIndex（支持用字符串切片等）
data.index = pd.to_datetime(data.index, utc=True)

# 确保是升序排列（2001->2007），这样才能算pct_change()
data.sort_index(inplace = True)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
求各自的每日收益、该策略的净收益
'''

data['Return_IGE'] = data['Close_IGE'].pct_change(fill_method=None)
data['Return_SPY'] = data['Close_SPY'].pct_change(fill_method=None)

# R = R_long - R_short
# 希望多头涨，空头跌，所以是多头的正向收益 + 空头的反向收益
# 除以2是因为动用了双倍资金（从总敞口来看），实操中不能用做空所得资金去做多
# 因为相减所以无需先各自减去r_f
data['Net Return'] = (data['Return_IGE'] - data['Return_SPY'])/2
print(data.head())

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
计算该策略的年化夏普比率
'''

sharpe_ratio = (
    np.sqrt(252) * 
    (data['Net Return'].mean() / data['Net Return'].std())
)

print(f'sharpe_ratio = {sharpe_ratio}')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
累计收益曲线，净值曲线，最高水位，回撤，最大回撤及发生日期
'''

# 累计收益曲线
data['Cumulative Return'] = (1 + data['Net Return']).cumprod() - 1

# 净值曲线（初始净值为1）
data['Net Value'] = 1 + data['Cumulative Return']

# 最高水位
data['High Watermark'] = data['Net Value'].cummax()

# 回撤
data['Drawdown'] = (
    (data['High Watermark'] - data['Net Value']) /
    data['High Watermark']
)

# 最大回撤
max_dd = data['Drawdown'].max()
print(f"最大回撤为：{max_dd}")

max_dd_date = data['Drawdown'].idxmax()
print(f"发生在：{max_dd_date.date()}")

# 画图
data[['Net Value','High Watermark']].plot()
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
最大回撤持续时间、起止日期
'''

duration = 0
max_duration = 0

start = None
end = None

# 跳过第一行NaN值
for index, row in data.iloc[1:].iterrows():  # 逐行遍历并返回index
    
    if row['Drawdown'] == 0:  # 回撤没开始，或回撤结束
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

num_trading_days = len(data.loc[start:end])  # loc是闭区间操作，iloc左闭右开
print(f"期间共有{num_trading_days}个交易日")


