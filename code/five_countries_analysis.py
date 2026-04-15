import pandas as pd
from fredapi import Fred
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. 设置API Key ====================
fred = Fred(api_key='40d97e088570cf58c45e6b0bba93ec51')  

# ==================== 2. 获取五国数据 + 汇率 ====================
print("正在获取中美欧日韩数据及汇率...")

# 中国
cn_ppi = fred.get_series('CHNPIEATI01GYM', observation_start='2015-01-01')
cn_ip  = fred.get_series('CHNPRINTO01IXPYM', observation_start='2015-01-01')
cn_ex  = fred.get_series('DEXCHUS', observation_start='2015-01-01')   # CNY per USD

# 美国
us_ppi = fred.get_series('PPIFIS', observation_start='2015-01-01')
us_ip  = fred.get_series('INDPRO', observation_start='2015-01-01')
us_ppi_yoy = us_ppi.pct_change(12) * 100
us_ip_yoy  = us_ip.pct_change(12) * 100

# 欧元区
ea_ppi = fred.get_series('PIEAMP02EZM659N', observation_start='2015-01-01')
ea_ip  = fred.get_series('PRMNTO01EZQ657S', observation_start='2015-01-01')
ea_ex  = fred.get_series('DEXUSEU', observation_start='2015-01-01')   # USD per EUR

# 日本
jp_ppi = fred.get_series('JPNPIEATI02GYM', observation_start='2015-01-01')
jp_ip  = fred.get_series('JPNPROINDMISMEI', observation_start='2015-01-01')
jp_ip_yoy = jp_ip.pct_change(12) * 100
jp_ex  = fred.get_series('DEXJPUS', observation_start='2015-01-01')   # JPY per USD

# 韩国
kr_ppi = fred.get_series('KORPPDMMINMEI', observation_start='2015-01-01')
kr_ip  = fred.get_series('KORPRINTO01GYSAM', observation_start='2015-01-01')
kr_ppi_yoy = kr_ppi.pct_change(12) * 100
kr_ex  = fred.get_series('DEXKOUS', observation_start='2015-01-01')   # KRW per USD

# 计算汇率YoY波动（正值 = 本币贬值）
cn_ex_yoy = cn_ex.pct_change(12) * 100
ea_ex_yoy = ea_ex.pct_change(12) * 100   # 这里EUR为反向
jp_ex_yoy = jp_ex.pct_change(12) * 100
kr_ex_yoy = kr_ex.pct_change(12) * 100

# ==================== 3. 合并数据 ====================
df = pd.DataFrame({
    'CN_PPI': cn_ppi, 'CN_IP': cn_ip, 'CN_EX_YoY': cn_ex_yoy,
    'US_PPI': us_ppi_yoy, 'US_IP': us_ip_yoy,
    'EA_PPI': ea_ppi, 'EA_IP': ea_ip, 'EA_EX_YoY': ea_ex_yoy,
    'JP_PPI': jp_ppi, 'JP_IP': jp_ip_yoy, 'JP_EX_YoY': jp_ex_yoy,
    'KR_PPI': kr_ppi_yoy, 'KR_IP': kr_ip, 'KR_EX_YoY': kr_ex_yoy
}).dropna()

df.index = pd.to_datetime(df.index)

print(" 五国+汇率数据合并完成！")
print("时间范围：", df.index.min().strftime('%Y-%m'), "到", df.index.max().strftime('%Y-%m'))

# ==================== 4. 相关性（新增汇率部分） ====================
print("\nPPI与汇率波动相关系数：")
print(df[['CN_PPI','US_PPI','EA_PPI','JP_PPI','KR_PPI',
          'CN_EX_YoY','EA_EX_YoY','JP_EX_YoY','KR_EX_YoY']].corr().round(3))

# ==================== 5. 可视化（新增汇率图） ====================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig, axs = plt.subplots(3, 1, figsize=(16, 12))

# PPI
for c, color, label in zip(['CN','US','EA','JP','KR'], ['blue','red','green','purple','orange'], ['中国','美国','欧元区','日本','韩国']):
    axs[0].plot(df.index, df[f'{c}_PPI'], label=label, color=color)
axs[0].set_title('五国制造业成本 (PPI同比)')
axs[0].legend(); axs[0].grid(True)

# 工业生产
for c, color, label in zip(['CN','US','EA','JP','KR'], ['blue','red','green','purple','orange'], ['中国','美国','欧元区','日本','韩国']):
    axs[1].plot(df.index, df[f'{c}_IP'], label=label, color=color)
axs[1].set_title('五国经济周期 (工业生产同比)')
axs[1].legend(); axs[1].grid(True)

# ==================== 汇率波动图（修复版） ====================
axs[2].plot(df.index, df['CN_EX_YoY'], label='中国 (CNY贬值 %)', color='blue')
axs[2].plot(df.index, df['JP_EX_YoY'], label='日本 (JPY贬值 %)', color='purple')
axs[2].plot(df.index, df['KR_EX_YoY'], label='韩国 (KRW贬值 %)', color='orange')

# 欧元区：转为欧元贬值（取负值，使方向一致）
if 'EA_EX_YoY' in df.columns:
    axs[2].plot(df.index, -df['EA_EX_YoY'], label='欧元区 (EUR贬值 %)', color='green')

#这里美元为本币，仅做参照，不加入汇率波动分析

axs[2].set_title('五国汇率波动 (YoY%，正值=本币贬值)')
axs[2].legend()
axs[2].grid(True)
plt.tight_layout()
plt.show()

# ==================== 6. 分国家滞后回归（此处增加了汇率变量） ====================
def lagged_regression_with_ex(country_ppi, country_ip, country_ex, country_name):
    print(f"\n{'='*70}\n{country_name} 滞后回归 (PPI ~ IP + lag1 + lag2 + EX_YoY)\n{'='*70}")
    temp = pd.DataFrame({
        'PPI': country_ppi, 
        'IP': country_ip, 
        'EX_YoY': country_ex
    }).dropna()
    temp['IP_lag1'] = temp['IP'].shift(1)
    temp['IP_lag2'] = temp['IP'].shift(2)
    temp = temp.dropna()
    
    X = sm.add_constant(temp[['IP', 'IP_lag1', 'IP_lag2', 'EX_YoY']])
    model = sm.OLS(temp['PPI'], X).fit()
    print(model.summary().tables[1])   # 打印系数表
    return model

# 运行五个国家
models = {}
models['中国'] = lagged_regression_with_ex(df['CN_PPI'], df['CN_IP'], df['CN_EX_YoY'], '中国')
models['美国'] = lagged_regression_with_ex(df['US_PPI'], df['US_IP'], df['US_PPI']*0, '美国')  # 美国汇率基准
models['欧元区'] = lagged_regression_with_ex(df['EA_PPI'], df['EA_IP'], df['EA_EX_YoY'], '欧元区')
models['日本'] = lagged_regression_with_ex(df['JP_PPI'], df['JP_IP'], df['JP_EX_YoY'], '日本')
models['韩国'] = lagged_regression_with_ex(df['KR_PPI'], df['KR_IP'], df['KR_EX_YoY'], '韩国')

df.to_csv('五国制造业成本周期汇率对比.csv')
plt.savefig('五国对比图.png', dpi=300)
print("\n✅ 项目全部完成！CSV、图片、5个国家回归表格均已输出")