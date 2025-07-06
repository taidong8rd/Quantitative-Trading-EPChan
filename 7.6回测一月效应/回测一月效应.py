# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 13:33:02 2025

@author: zhang
"""

# CASE 7.6

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
读取标准普尔600小市值股票的收盘价数据
范围：2004-01-15 - 2008-02-01
'''

cl = pd.read_csv('IJR_20080131.txt', delim_whitespace=True)  # 收盘价

cl['Date'] = pd.to_datetime(cl['Date'], errors='coerce', format='%Y%m%d')

cl.set_index('Date', inplace=True)
cl.sort_index(inplace=True)

print(cl.head())

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
创建一个新dataframe，储存月份，并找出去年年末和今年一月末的日期一一对应
lastday_dec: 2004-12-31, 2005-12-30, ...
lastday_jan: 2005-01-31, 2006-01-31, ...
'''

df = pd.DataFrame(index=cl.index)

df['Month'] = cl.index.month
df['Next Month'] = df['Month'].shift(-1)

lastday_dec = df[(df['Month'] == 12) & (df['Next Month'] == 1)].index  # 保存为DatetimeIndex
lastday_jan = df[(df['Month'] == 1) & (df['Next Month'] == 2)].index

lastday_jan = lastday_jan[1:]  # 删除2004-01-30，因为没有2003年12月最后一天的数据与之对应

print(lastday_jan > lastday_dec)  # 检查1月日期是否晚于12月日期，防止年份错乱
print(f'lastday_dec: {lastday_dec}')
print(f'lastday_jan: {lastday_jan}')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
计算每只股票的年收益和一月收益
'''

daily_return = cl.pct_change(fill_method=None)

# 一月回报（今年1月最后一天 - 去年12月最后一天）
jan_return = (
    (cl.loc[lastday_jan].values - cl.loc[lastday_dec].values)
    / cl.loc[lastday_dec].values  # .values转成np数组，对应位置相减，而不是对应index相减
)
# 转成pd.Series，4行日期，600列股票
jan_return = pd.DataFrame(jan_return, index=lastday_jan, columns=cl.columns)


# 年回报（今年12月最后一天 - 去年12月最后一天）
annual_return = (
    (cl.loc[lastday_dec].values - cl.loc[lastday_dec].shift(1).values)
    / cl.loc[lastday_dec].shift(1).values
)
# 去掉第一行NaN
annual_return = pd.DataFrame(annual_return[1:], index=lastday_dec[1:], columns=cl.columns)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
反转策略：表现最差的股票在下一年一月会反弹(->做多），表现最好的股票会回调(->做空)
在年末，基于这一整年的年收益给股票排序，选股构建多空组合，回测次年一月收益，验证"一月效应"
'''

oneway_tcost = 0.0005

for i, (index, row) in enumerate(annual_return.iterrows()):  # 逐年遍历
    
    # 筛选年收益非NaN的有效股
    has_data = np.isfinite(row)
    valid_return = row.loc[has_data]
    
    topN = round(len(valid_return)/10)  # 选前10%和后10%的股
    
    shorts = valid_return.nlargest(topN).index  # 年收益最好 -> 预计1月回落 -> 做空
    longs = valid_return.nsmallest(topN).index  # 年收益最差 -> 预计1月上涨 -> 做多
    
    # 构建持仓向量，年底建仓，持有至次年一月底
    positions = pd.Series(0, index=annual_return.columns)  # 全部600股
    positions.loc[longs] = 1
    positions.loc[shorts] = -1
    
    exposure = np.abs(positions).sum()  # = 2*topN
    
    # 回测一月收益：乘以次年1月真实收益率
    port_return = (positions * jan_return.iloc[i+1]).sum()  # 按索引（股票代码）对应
    port_return /= exposure  # -> 单位资金收益
    port_return -= 2 * oneway_tcost  # 多头建仓、空头建仓各支付一次
    
    print(f'Last holding date {index.date()}: Portfolio return = {port_return:.5f}')




