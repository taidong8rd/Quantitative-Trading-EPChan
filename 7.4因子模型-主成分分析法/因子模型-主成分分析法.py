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

df = df.ffill().bfill()  # 先前向填充，再后向填充（如果第一行是NaN）

print(df.head())

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
参数设置
'''

lookback = 252  # 窗口大小(days)
num_factors = 5  # 因子数量
topN = 50  # 选股数量


def calc_R(df):
    '''
    构建R: 超额收益率矩阵（本case没减去r_f？）
    去均值: 让协方差矩阵只反映变量间波动的同步性（相对于各自平均水平的波动），而非均值差异
    从而找出能最大程度解释变动的方向的因子。
    '''
    daily_return = df.pct_change(fill_method=None)  # TxN (T days, N stocks)
    daily_return = daily_return.iloc[1:]  # 去掉第一行NaN值，便于后续求X
    mean_return = daily_return.mean(axis=0)  # 每只股票的平均收益率
    R = daily_return.sub(mean_return, axis=1)  # 去均值化，每股的收益-均值
    R = R.transpose()  # -> Nx(T-1)
    
    return R


def calc_X(R, num_factors=num_factors):
    '''
    构建X: 因子暴露矩阵
    每一列都是协方差矩阵的特征向量（取特征值最大的几个），代表最大的几个主成分方向
    np.linalg.eigh要求输入矩阵是实对称（A = A^T，比如协方差矩阵）或复共轭对称的
    返回特征值（升序排列）和对应的特征向量(每一列)
    '''
    cov_matrix = R @ R.T / R.shape[1]  # normalise by T-1
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    X = eigvecs[:, -num_factors:]  # 选出倒数n列（对应n个最大的特征值）
    
    return X


def calc_b(R, X):
    '''
    构建b: 因子收益向量
    用最后一天各股的收益向量r、X矩阵做线性回归
    r = Xb + u
    求出b = (b1, b2, b3, b4, b5, ...)^T
    '''
    r = R.iloc[:,-1]
    # X = sm.add_constant(X)
    model = sm.OLS(r, X).fit()
    b = model.params
    residual = model.resid
    
    return b, residual


def get_positions(df_window, topN=topN):
    '''
    构建持仓向量
    使用给定dataframe窗口，求R，X，进而线性回归求b
    用b预测窗口下一天各股的收益：
    r_next = r_mean + Xb, r_mean是窗口内各股的平均收益
    根据预测的r_next决定今天持仓情况
    '''
    R = calc_R(df_window)
    X = calc_X(R)
    b, residual = calc_b(R,X)
    
    # 窗口范围内每只股票的mean return
    r_mean = df_window.pct_change(fill_method=None).mean(axis=0)  
    r_next = r_mean + np.dot(X,b)  # 预测窗口下一天各股的收益
    
    # 根据预测在今日建仓
    positions = np.zeros_like(r_next)
    
    # 例: argsort([3,1,2]) -> [1,2,0] (返回数组升序排序后的，各数在原数组中的索引)
    # sort_idx的最后topN个数就是r_next最大的topN个股票的index
    # 要把positions向量和r_next向量的索引对应起来
    sort_idx = np.argsort(r_next)  
    
    positions[sort_idx[-topN:]] = 1   # r_next最高的topN个做多
    positions[sort_idx[0:topN]] = -1  # r_next最低的topN个做空
                                      # 其余股仓位0
    return positions
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
滚动窗口
模拟每天用过去lookback(e.g.252)天的数据重新建模、预测下一天的股票表现并在今日选股建仓
每个窗口内都是求R、X、b、positions
累加每日pnl，pnl用今日持仓（基于历史窗口算出）与下一天真实的收益率计算
'''

daily_return = df.pct_change(fill_method=None)

pnl_list = []

for i in range(0, len(df)-lookback):
    
    print(f'Window [{i}:{lookback+i})')  # last window [753:1005)，不含df的最后一天
    df_window = df.iloc[i:lookback+i]
   
    positions = get_positions(df_window)
#    exposure = np.abs(positions).sum()  # = 100
    
    next_return = daily_return.iloc[lookback+i]  # 下一天真实收益率  
                                  
    pnl = np.dot(positions, next_return)  # 回测
#    pnl /= exposure  # -> 单位资金的收益
    pnl_list.append(pnl)

avg_pnl = np.mean(pnl_list)
avg_pnl_annual = avg_pnl * 252

print(f'avg_pnl = {avg_pnl}')
print(f'avg_pnl_annual = {avg_pnl_annual}')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
画PnL图
'''

dates = df.index[lookback:len(df)]
pnl_series = pd.Series(pnl_list, index=dates)

pnl_series.plot(label="Daily PnL")
pnl_series.cumsum().plot(label="Cumulative PnL")
plt.legend()
plt.show()



















