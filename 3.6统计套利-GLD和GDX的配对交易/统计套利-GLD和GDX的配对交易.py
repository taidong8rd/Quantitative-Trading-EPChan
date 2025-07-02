# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 17:30:58 2025

@author: zhang
"""

# CASE 3.6

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.api import OLS, add_constant

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
使用yfinance获取目标股票数据
'''

tickers = ['GLD', 'GDX']
start_date = "2006-05-23"
end_date = "2007-11-30"

# from curl_cffi import requests
# session = requests.Session(impersonate="chrome")

for ticker in tickers:
    stock = yf.Ticker(ticker)
    #stock = yf.Ticker(ticker, session = session)   
    data = stock.history(start=start_date, end=end_date, interval="1d", auto_adjust=False)
    data.index.name = 'Date'
    
    if data.empty:
        print(f"{ticker}: Data is empty!")
    else:
        data.to_csv(f"{ticker}.csv")  # 保存为csv
        print(f"{ticker}.csv saved successfully with {len(data)} rows.")
        
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
读取数据、创建dataframe、画GLD和GDX收盘价折线图
'''

# 注；这里导入的是书中使用的数据，与yfinance下载的略有不同，范围2006-05-23到2007-11-30
df1 = pd.read_excel("GLD.xls")
df2 = pd.read_excel("GDX.xls")
df = pd.merge(df1, df2, on="Date", suffixes=("_GLD", "_GDX"))  # 按日期合并，列名加后缀区分
df.set_index("Date", inplace=True)  # Date列做index
df.sort_index(inplace=True)  # 确保升序
df = df[['Adj Close_GLD','Adj Close_GDX']]  # 选列
df.rename(columns={'Adj Close_GLD': 'Close_GLD','Adj Close_GDX': 'Close_GDX'}, inplace=True)

print(df.head())

'''
GLD和GDX间存在稳定统计关系  -> 线性回归等数学模型
当二者价格短期偏离这种关系时最终会均值回归
偏离发生时存在套利机会  -> long & short（配对交易）-> 偏离消失时平仓获利
不论大盘涨跌，只要二者间统计关系稳定，就可用此法套利
'''
df[['Close_GLD','Close_GDX']].plot()  # 观察
plt.xticks(rotation=45)
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
线性回归模型，估计对冲比率
构造对冲组合（买1股GLD，卖hedge_ratio股GDX），对冲系统性风险
GLD = hedge_ratio * GDX（股价）
没有引入截距，因为会过拟合（we want GLD = 0 when GDX = 0）
'''

close_gld = df['Close_GLD'].values  # convert to arrays
close_gdx = df['Close_GDX'].values

# 划分训练集和测试集
trainset = np.arange(0, 252)  # trainset index (用前252个)
testset = np.arange(252, len(df))  # testset index (用剩下的)

#X_train = add_constant(close_gdx[trainset])  # 引入a -> a + b * GDX（这里不用）
X_train = close_gdx[trainset]
Y_train = close_gld[trainset]
model = OLS(Y_train, X_train).fit()  # 线性回归（基于训练集）

hedge_ratio = model.params[0]  # get the parameter(s)
print(f'hedge_ratio = {hedge_ratio}')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
构造spread（价差）和z-score（衡量价差的偏移程度）
'''

# spread不等于0，len(spread) = len(完整数据集)，在训练集上平均spread接近0
spread = close_gld - hedge_ratio * close_gdx  # spread = GLD - b * GDX

spread_train = spread[trainset]  # 训练集上两者的价差

# 基于训练集求出价差的mean和std
spread_mean = np.mean(spread_train)
spread_std = np.std(spread_train)
print(f'spread_mean = {spread_mean}, spread_std = {spread_std}')

# 覆盖整个数据集的zscore，衡量spread相对于其mean的偏移
zscore = (spread - spread_mean) / spread_std

df['Spread'] = spread
df['Z-score'] = zscore

# 绘制价差的变化
df['Spread'].plot()
plt.xticks(rotation=45)
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
生成交易信号（T/F组成的一维数组，记录应该执行的操作）
交易信号取决于zscore(spread的偏离程度)
spread = GLD - b * GDX
zscore = (spread - spread_mean)/std
'''

# 若价差低于均值很多 -> 此时gld偏低，gdx偏高 -> 预期gld涨（long），gdx跌（short）
longs = zscore <= -2  # long gld

shorts = zscore >= 2  # short gld

# 价差回归到均值附近，套利空间消失，不再持仓（锁定利润）
exits = np.abs(zscore) <= 1

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
建立持仓
'''

# 每个交易日一行，每行两列（分别记gld和gdx的持仓），初始值为NaN
positions = np.full((len(df), 2), np.nan)


'''
Dollar-neutral（多头投入资金 = 空头投入资金）
仓位向量记录的是单位资金，而不是股数！
'''
positions[longs] = [1, -1]  # long gld, short gdx
positions[shorts] = [-1, 1]  # short gld, long gdx
positions[exits] = [0, 0]  # 平仓


'''
Beta-neutral（严格按照hedge_ratio建仓）
仓位向量记录的是股数！
'''
# positions[longs] = [1, -hedge_ratio]  # long 1 gld, short hedge_ratio * gdx
# positions[shorts] = [-1, hedge_ratio]  # short 1 gld, long hedge_ratio * gdx
# positions[exits] = [0, 0]  # 平仓


# 对于 1<abs(zscore)<2 的交易日，无操作，保持前日持仓
positions = pd.DataFrame(positions).ffill().values  # 用df的ffill功能

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
计算收益PnL
PnL_t = Return_t * position_(t-1) * price_t, then sum over all stocks
归一化：除以总敞口 -> 单位资金收益
'''

df['Return_GLD'] = df['Close_GLD'].pct_change(fill_method=None)
df['Return_GDX'] = df['Close_GDX'].pct_change(fill_method=None)
df['Position_GLD'] = positions[:,0]
df['Position_GDX'] = positions[:,1]


'''
如果position记的是单位资金投入，如[1, -1]:
'''
df['PnL'] = (
    df['Return_GLD'] * df['Position_GLD'].shift(1) +  # positions自带正负号
    df['Return_GDX'] * df['Position_GDX'].shift(1)
)

df['Exposure'] = (
    (df['Position_GLD'].shift(1)).abs() +  # 总敞口（多头和空头头寸的绝对值之和）
    (df['Position_GDX'].shift(1)).abs()
)


'''
如果position记的是股数，如[1, -hedge_ratio]，需乘价格:
'''
# df['PnL'] = (
#     df['Return_GLD'] * df['Position_GLD'].shift(1) * df['Close_GLD'] +
#     df['Return_GDX'] * df['Position_GDX'].shift(1) * df['Close_GDX']
# )

# df['Exposure'] = (
#     (df['Position_GLD'].shift(1) * df['Close_GLD']).abs() +
#     (df['Position_GDX'].shift(1) * df['Close_GDX']).abs()
# )


# 用总敞口归一化得到单位资金的收益
# 前一天仓位[0,0] -> 总敞口0 -> PnL:NaN，实际上这种情况PnL应=0
df['PnL'] = (df['PnL'] / df['Exposure']).fillna(0)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
计算夏普比率
'''

# 注意：Series.std() != np.std(Series)，前者/(n-1)，后者/n
# 用Series.std()以防低估风险！
def get_sharpe_ratio(dataset):    
    sharpe_ratio = (
        np.sqrt(252) * 
        (dataset['PnL'].mean() / dataset['PnL'].std())
    )
    return sharpe_ratio

sharpe_ratio_train = get_sharpe_ratio(df.iloc[trainset])
print(f'sharpe_ratio_train = {sharpe_ratio_train}')

sharpe_ratio_test = get_sharpe_ratio(df.iloc[testset])
print(f'sharpe_ratio_test = {sharpe_ratio_test}')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
计算Cumulative P&L
累加该策略在每一天的收益
持仓情况在变，不是持有单一资产，所以没有复利累计
'''

df['Cumulative PnL'] = df['PnL'].cumsum()

# 在训练集+测试集上的表现
df['Cumulative PnL'].plot()
plt.xticks(rotation=45)
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
检查策略是否存在未来数据偏差
即在计算某一天的持仓时，是否使用了"未来"的数据
'''

# 封装成函数
def get_positions(df):
    
    trainset = np.arange(0, 252)
#    testset = np.arange(252, len(df))
    
    close_gdx = df['Close_GDX'].values
    close_gld = df['Close_GLD'].values
    
#    X_train = add_constant(close_gdx[trainset])
    X_train = close_gdx[trainset]
    Y_train = close_gld[trainset]
    model = OLS(Y_train, X_train).fit()
    
    spread = close_gld - model.params[0] * close_gdx

    spread_train = spread[trainset]

    spread_mean = np.mean(spread_train)
    spread_std = np.std(spread_train)
    
    zscore = (spread - spread_mean) / spread_std
    
    longs = zscore <= -2  
    shorts = zscore >= 2
    exits = np.abs(zscore) <= 1

    positions = np.full((len(df), 2), np.nan)
    positions[shorts] = [-1, 1]
    positions[longs] = [1, -1]
    positions[exits] = [0, 0]
    # positions[longs] = [1, -hedge_ratio]
    # positions[shorts] = [-1, hedge_ratio]
    # positions[exits] = [0, 0]
    positions = pd.DataFrame(positions).ffill().values
    
    return positions

cut = 60  # 裁掉数据集中最后60天的数据
df_cut = df[:-cut]

positions_cut = get_positions(df_cut)  # 用该数据集求每日持仓情况

assert positions_cut.shape == positions[:-cut].shape, "Shape mismatch!"

#与原positions数组比较
if not np.array_equal(positions_cut, positions[:-cut]):
    print("Look-ahead bias detected!")
else:
    print("No look-ahead bias detected.")












