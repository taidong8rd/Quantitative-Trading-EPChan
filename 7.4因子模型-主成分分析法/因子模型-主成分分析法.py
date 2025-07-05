# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 12:05:32 2025

@author: zhang
"""

# CASE 7.4

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

'''
读取标准普尔600小市值股票数据
'''

df = pd.read_csv('IJR_20080114.txt', delim_whitespace=True)  # 收盘价

df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format='%Y%m%d')

df.set_index('Date', inplace=True)
df.sort_index(inplace=True)

#df = df.ffill().bfill()  # 先前向填充，再后向填充（如果第一行是NaN）

print(df.head())

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
参数设置
'''

lookback = 252  # 窗口大小(days)
num_factors = 5  # 因子数量
topN = 50  # 选股数量（多头50，空头50）


def calc_R(dailyret_window):
    '''
    构建R: 超额收益率矩阵（本case没减去r_f？）
    去均值: 让协方差矩阵只反映变量间波动的同步性（相对于各自平均水平的波动），而非均值差异
    从而找出能最大程度解释变动的方向的因子。
    '''   
    # 去除有missing returns的股票（列）
    has_data = np.isfinite(dailyret_window).all(axis=0)
    dailyret_window_clean = dailyret_window.loc[:, has_data]  # 有效股

    r_mean = dailyret_window_clean.mean(axis=0)  # 每列求平均（每只股票的平均收益率）
    R = dailyret_window_clean.sub(r_mean, axis=1)  # 去平均化
    R = R.transpose()  # TxN -> NxT (N stocks, T days)
    
    return R, r_mean
    
    
def calc_X(R, num_factors=num_factors):
    '''
    构建X: 因子暴露矩阵
    每一列都是协方差矩阵的特征向量（取特征值最大的几个），代表最大的几个主成分方向（因子）
    np.linalg.eigh要求输入矩阵是实对称（A = A^T，比如协方差矩阵）或复共轭对称的
    返回特征值（升序排列）和对应的特征向量(每一列)
    '''
    cov_matrix = R @ R.T / R.shape[1]  # normalise by T
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    X = eigvecs[:, -num_factors:]  # 选出倒数n列（对应n个最大的特征值）
    
    return X


def calc_b(R, X):
    '''
    构建b: 因子收益向量
    用最后一天各股的收益向量r（因变量）、X矩阵（自变量）做线性回归
    r = Xb + u
    求出b = (b1, b2, b3, b4, b5, ...)^T
    '''
    r = R.iloc[:,-1]
    #X = sm.add_constant(X)
    model = sm.OLS(r, X).fit()
    b = model.params
    residual = model.resid
    
    return b, residual


def get_positions(dailyret_window, topN=topN):
    '''
    构建持仓向量
    使用给定dataframe窗口，求R，X，进而线性回归求b
    用b预测窗口下一天各股的收益：
    r_exp = r_mean + Xb, r_mean是窗口内各股的平均收益
    根据预测的r_exp来建仓（后续将乘真实下一天收益r_next，回测pnl）
    '''
    # 求R, X, b
    R, r_mean = calc_R(dailyret_window)
    X = calc_X(R)
    b, residual = calc_b(R,X)
    
    r_exp = r_mean + np.dot(X,b)  # 预测窗口下一天各有效股的收益
    
    # 根据预测建仓，positions将记录全部600股的持仓情况
    positions = pd.Series(0, index=dailyret_window.columns)  # 列名（股票代码）做索引
    
    longs = r_exp.nlargest(topN).index
    shorts = r_exp.nsmallest(topN).index

    positions.loc[longs] = 1   # r_exp最高的topN个做多
    positions.loc[shorts] = -1  # r_exp最低的topN个做空，其余股仓位0（含无效股）
                                     
    return positions
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
滚动回测窗口：
每天用过去lookback(e.g.252)天(含当天T)的数据构造R, X, b, 并预测下一天(T+1)的各股收益
在T日收盘后确定T+1的持仓（positions），并准备次日开盘建仓
T+1开盘建仓，T+1收盘平仓，获得该日pnl
pnl通过持仓点乘T+1的真实收益率得出
'''

dailyret = df.pct_change(fill_method=None)

pnl_list = []

for i in range(1, len(df)-lookback):
    
    # 创建窗口
    # first window [1:253)，不含df第一天（因为dailyret都是NaN）
    # last window [753:1005)，不含df的最后一天（因为没有下一天的真实return）
    print(f'Window [{i}:{lookback+i})')
    dailyret_window = dailyret.iloc[i:lookback+i]
    
    r_next = dailyret.iloc[lookback+i]
    
    positions = get_positions(dailyret_window)
                                  
    pnl = (positions * r_next).dropna().sum()  # 去掉r_next为NaN的
    pnl_list.append(pnl)
    
#    exposure = np.abs(positions).sum()  # = 100
#    pnl /= exposure  # -> 单位资金的收益


avg_pnl = np.mean(pnl_list)
avg_pnl_annual = avg_pnl * 252

print(f'avg_pnl = {avg_pnl}')
print(f'avg_pnl_annual = {avg_pnl_annual}')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
画PnL图
'''

dates = df.index[lookback+1:len(df)]
pnl_series = pd.Series(pnl_list, index=dates)

pnl_series.plot(label="Daily PnL")
pnl_series.cumsum().plot(label="Cumulative PnL")
plt.legend()
plt.show()



















