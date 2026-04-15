# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 16:38:41 2026

@author: anque
"""

import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from matplotlib import font_manager


plt.rcParams['font.sans-serif'] = ['SimHei']    
plt.rcParams['axes.unicode_minus'] = False      


# ==================== 1. 读取CSV文件 ====================

ppi = pd.read_csv(r"C:\Users\anque\Downloads\PPI_data.csv")
ip = pd.read_csv(r"C:\Users\anque\Downloads\IP_data.csv")



print("PPI文件列名：", ppi.columns.tolist())
print("IP文件列名：", ip.columns.tolist())

# ==================== 2. 数据清洗与合并 ====================
# 使用列名 'observation_date'
ppi['observation_date'] = pd.to_datetime(ppi['observation_date'])
ip['observation_date'] = pd.to_datetime(ip['observation_date'])

ppi = ppi.set_index('observation_date')
ip = ip.set_index('observation_date')

# 按日期合并
df = ppi.join(ip, how='inner')

# 重命名列名
df = df.rename(columns={
    df.columns[0]: 'PPI_YoY',      # 制造业成本（PPI同比）
    df.columns[1]: 'IP_YoY'        # 工业生产同比
})

# ==================== 3. 处理缺失值 ====================
print("\n合并前形状：", ppi.shape, "+", ip.shape)
print("合并后形状：", df.shape)

print("\n缺失值数量：\n", df.isnull().sum())

df = df.dropna()   # 删除缺失行

# ==================== 4. 检查结果 ====================
print("\n数据清洗与合并完成！")
print("时间范围：", df.index.min().strftime('%Y-%m'), " 到 ", df.index.max().strftime('%Y-%m'))
print("总月度观测数：", len(df))

print("\n列名含义：")
print("- PPI_YoY：生产者价格指数同比增速 (%) —— 制造业成本")
print("- IP_YoY：工业生产同比增速 (%) —— 经济周期")

print("\n描述性统计：")
print(df.describe().round(3))

print("\nPPI 与 IP 的相关系数：")
print(df.corr().round(3))

# ==================== 5. 保存文件 ====================
df.to_csv('manufacturing_cost_vs_cycle_cleaned.csv', index=True)
print("\n已保存为 manufacturing_cost_vs_cycle_cleaned.csv")

# ==================== 6. 快速趋势图 ====================
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['PPI_YoY'], label='PPI YoY (制造业成本)', color='blue', linewidth=2)
plt.plot(df.index, df['IP_YoY'], label='IP YoY (经济周期)', color='orange', linewidth=2)
plt.title('中国制造业成本 vs 经济周期（2015年后）')
plt.ylabel('同比增速 (%)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
import statsmodels.api as sm

# 线性回归：PPI ~ IP
X = sm.add_constant(df['IP_YoY'])   # 添加常数项
y = df['PPI_YoY']
model = sm.OLS(y, X).fit()

print(model.summary())

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ====================== 1. 读取之前清理好的数据 ======================
df = pd.read_csv('manufacturing_cost_vs_cycle_cleaned.csv', index_col='observation_date', parse_dates=True)

print("原始数据形状：", df.shape)

# ====================== 2. 创建滞后变量 ======================
df['IP_lag1'] = df['IP_YoY'].shift(1)   # 滞后1期
df['IP_lag2'] = df['IP_YoY'].shift(2)   # 滞后2期

# 删除因创建滞后产生的缺失值
df = df.dropna()

print("加入滞后项后数据形状：", df.shape)
print("时间范围：", df.index.min().strftime('%Y-%m'), "到", df.index.max().strftime('%Y-%m'))

# ====================== 3. 改进版多元回归 ======================
# 模型1：原始简单回归（做对比使用）
X1 = sm.add_constant(df['IP_YoY'])
model1 = sm.OLS(df['PPI_YoY'], X1).fit()

# 模型2：加入滞后1期，这里使用上一个月的工业生产增速（即 t-1 期的 IP）
X2 = sm.add_constant(df[['IP_YoY', 'IP_lag1']])
model2 = sm.OLS(df['PPI_YoY'], X2).fit()

# 模型3：加入滞后1期和2期（使用上上一个月，同上，不做过多赘述）
X3 = sm.add_constant(df[['IP_YoY', 'IP_lag1', 'IP_lag2']])
model3 = sm.OLS(df['PPI_YoY'], X3).fit()

# ====================== 4. 输出结果 ======================
print("\n" + "="*60)
print("模型1：简单回归 (仅当前IP)")
print(model1.summary())

print("\n" + "="*60)
print("模型2：加入滞后1期")
print(model2.summary())

print("\n" + "="*60)
print("模型3：加入滞后1期和2期（推荐）")
print(model3.summary())

# ====================== 5. 保存回归结果 ======================
with open('regression_results.txt', 'w', encoding='utf-8') as f:
    f.write("模型1：简单回归\n")
    f.write(str(model1.summary()))
    f.write("\n\n模型3：多元回归（推荐）\n")
    f.write(str(model3.summary()))

print("\n 回归结果已保存为 regression_results.txt")

#中美分析对比

import pandas as pd
from fredapi import Fred
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. 设置FRED API Key ====================
fred = Fred(api_key='40d97e088570cf58c45e6b0bba93ec51')   
# ==================== 2. 获取中美数据 ====================
print("正在从FRED获取中美数据...")

# 中国（已为YoY）
cn_ppi = fred.get_series('CHNPIEATI01GYM', observation_start='2015-01-01')
cn_ip = fred.get_series('CHNPRINTO01IXPYM', observation_start='2015-01-01')

# 美国（需额外计算YoY）
us_ppi = fred.get_series('PPIFIS', observation_start='2015-01-01')      # 水平指数
us_ip = fred.get_series('INDPRO', observation_start='2015-01-01')       # 水平指数

# 计算美国同比增速（12个月滚动）
us_ppi_yoy = us_ppi.pct_change(12) * 100
us_ip_yoy = us_ip.pct_change(12) * 100

# ==================== 3. 合并数据 ====================
df = pd.DataFrame({
    'CN_PPI': cn_ppi,
    'CN_IP': cn_ip,
    'US_PPI_YoY': us_ppi_yoy,
    'US_IP_YoY': us_ip_yoy
}).dropna()

df.index = pd.to_datetime(df.index)

print("数据合并完成！")
print("时间范围：", df.index.min().strftime('%Y-%m'), "到", df.index.max().strftime('%Y-%m'))
print("观测数：", len(df))

# ==================== 4. 相关性分析 ====================
print("\n中美相关系数矩阵：")
print(df.corr().round(3))

# ==================== 5. 可视化对比 ====================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# 图1：PPI对比
axs[0,0].plot(df.index, df['CN_PPI'], label='中国PPI', color='blue')
axs[0,0].plot(df.index, df['US_PPI_YoY'], label='美国PPI', color='red')
axs[0,0].set_title('中美制造业成本（PPI同比）对比')
axs[0,0].legend()
axs[0,0].grid(True)

# 图2：工业生产对比
axs[0,1].plot(df.index, df['CN_IP'], label='中国工业生产', color='blue')
axs[0,1].plot(df.index, df['US_IP_YoY'], label='美国工业生产', color='red')
axs[0,1].set_title('中美经济周期（工业生产同比）对比')
axs[0,1].legend()
axs[0,1].grid(True)

# 图3：中美PPI散点+回归
sns.regplot(x='US_PPI_YoY', y='CN_PPI', data=df, ax=axs[1,0], scatter_kws={'alpha':0.6})
axs[1,0].set_title('中美PPI相关性（散点+回归）')

# 图4：中美IP散点+回归
sns.regplot(x='US_IP_YoY', y='CN_IP', data=df, ax=axs[1,1], scatter_kws={'alpha':0.6})
axs[1,1].set_title('中美工业生产相关性')

plt.tight_layout()
plt.show()

# ==================== 6. 各国简单回归 ====================
print("\n中国回归（PPI ~ IP）：")
X_cn = sm.add_constant(df['CN_IP'])
model_cn = sm.OLS(df['CN_PPI'], X_cn).fit()
print(model_cn.summary().tables[1])

print("\n美国回归（PPI ~ IP）：")
X_us = sm.add_constant(df['US_IP_YoY'])
model_us = sm.OLS(df['US_PPI_YoY'], X_us).fit()
print(model_us.summary().tables[1])

# 保存所有的结果
df.to_csv('中美制造业成本周期对比_cleaned.csv')
plt.savefig('中美对比图.png', dpi=300, bbox_inches='tight')
print("\n项目完成！文件已保存")



















