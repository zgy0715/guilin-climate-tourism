#!/usr/bin/env python3
"""
使用2013-2025年月度旅游数据重新分析问题三
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import os, sys

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "数据")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from cci_core import read_weather_data, add_comfort_scores, calc_cci

print("=" * 70)
print("问题三：基于月度数据的气候对旅游影响分析")
print("=" * 70)

# ═════════════════════════════════════════════════════════════
# 1. 读取并合并数据
# ═════════════════════════════════════════════════════════════
print("\n1. 读取月度旅游气候合并数据...")
df = pd.read_csv(os.path.join(DATA_DIR, "月度旅游气候合并数据.csv"))
print(f"   合并数据: {len(df)}行, 年份{int(df['Year'].min())}-{int(df['Year'].max())}")

# 有效数据（有旅游数据的月份）
valid = df.dropna(subset=["tourists", "revenue"]).copy()
print(f"   有效数据(有旅游值): {len(valid)}行")

# ═════════════════════════════════════════════════════════════
# 2. 灰色关联分析（月度）
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2. 灰色关联分析（月度数据）")
print("=" * 70)

from sklearn.preprocessing import MinMaxScaler

def grey_relation(reference, factors, rho=0.5):
    ref = np.array(reference, dtype=float)
    fac = np.array(factors, dtype=float)
    scaler = MinMaxScaler()
    ref_norm = scaler.fit_transform(ref.reshape(-1, 1)).flatten()
    fac_norm = scaler.fit_transform(fac)
    diff = np.abs(fac_norm - ref_norm.reshape(-1, 1))
    d_min, d_max = diff.min(), diff.max()
    coefficients = (d_min + rho * d_max) / (diff + rho * d_max)
    return coefficients.mean(axis=0)

factor_cols = ["CCI", "Tavg", "Tmax", "Tmin", "rain_days", "hot_days", "cold_days"]
factor_names = ["CCI指数", "平均气温", "最高气温", "最低气温", "降雨天数", "高温天数", "低温天数"]

gra_tourists = grey_relation(valid["tourists"], valid[factor_cols])
gra_revenue = grey_relation(valid["revenue"], valid[factor_cols])

gra_df = pd.DataFrame({
    "因素": factor_names,
    "游客量关联度": gra_tourists.round(4),
    "旅游收入关联度": gra_revenue.round(4),
}).sort_values("游客量关联度", ascending=False)
print(gra_df.to_string(index=False))

# ═════════════════════════════════════════════════════════════
# 3. 月度回归分析
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3. 月度回归分析")
print("=" * 70)

# 创建月度虚拟变量
valid["month_sin"] = np.sin(2 * np.pi * valid["Month"] / 12)
valid["month_cos"] = np.cos(2 * np.pi * valid["Month"] / 12)
valid["pandemic"] = ((valid["Year"] >= 2020) & (valid["Year"] <= 2022)).astype(int)

# 模型1: 仅CCI
X1 = sm.add_constant(valid[["CCI"]])
y_t = valid["tourists"]
m1 = sm.OLS(y_t, X1).fit()
print(f"\n模型1 (仅CCI): R2={m1.rsquared:.4f}, 调整R2={m1.rsquared_adj:.4f}")

# 模型2: CCI + 季节
X2 = sm.add_constant(valid[["CCI", "month_sin", "month_cos"]])
m2 = sm.OLS(y_t, X2).fit()
print(f"模型2 (CCI+季节): R2={m2.rsquared:.4f}, 调整R2={m2.rsquared_adj:.4f}")

# 模型3: CCI + 季节 + 疫情
X3 = sm.add_constant(valid[["CCI", "month_sin", "month_cos", "pandemic"]])
m3 = sm.OLS(y_t, X3).fit()
print(f"模型3 (CCI+季节+疫情): R2={m3.rsquared:.4f}, 调整R2={m3.rsquared_adj:.4f}")

# 模型4: 多气候变量 + 季节 + 疫情
X4 = sm.add_constant(valid[["CCI", "Tavg", "rain_days", "hot_days", "month_sin", "month_cos", "pandemic"]])
m4 = sm.OLS(y_t, X4).fit()
print(f"模型4 (多变量+季节+疫情): R2={m4.rsquared:.4f}, 调整R2={m4.rsquared_adj:.4f}")

print(f"\n最优模型（模型4）回归结果:")
print(m4.summary().tables[1])

# ═════════════════════════════════════════════════════════════
# 4. 月度游客量指数分析
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4. 月度游客量指数分析")
print("=" * 70)

# 计算月度游客量指数
yearly_total = valid.groupby("Year")["tourists"].sum().reset_index()
yearly_total.columns = ["Year", "yearly_total"]
valid2 = pd.merge(valid, yearly_total, on="Year")
valid2["tourist_index"] = valid2["tourists"] / valid2["yearly_total"] * 100

# 按月份统计平均游客量指数
monthly_avg = valid2.groupby("Month")["tourist_index"].mean()
print("\n各月平均游客量指数(%):")
for m in range(1, 13):
    if m in monthly_avg.index:
        print(f"  {m:2d}月: {monthly_avg[m]:.1f}%")

# ═════════════════════════════════════════════════════════════
# 5. 年度趋势分析
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("5. 年度趋势分析")
print("=" * 70)

yearly = valid.groupby("Year").agg(
    tourists=("tourists", "sum"),
    revenue=("revenue", "sum"),
    CCI=("CCI", "mean"),
    Tavg=("Tavg", "mean"),
    rain_days=("rain_days", "sum"),
).reset_index()

print(yearly.to_string(index=False))

# ═════════════════════════════════════════════════════════════
# 6. 两阶段对比
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("6. 两阶段对比分析")
print("=" * 70)

y1 = valid[valid["Year"] <= 2018]
y2 = valid[valid["Year"] >= 2019]

indicators = ["tourists", "revenue", "CCI", "Tavg", "rain_days"]
names = ["月均游客量(万)", "月均收入(亿)", "月均CCI", "月均气温(°C)", "月均降雨天数"]

print(f"\n{'指标':<15} {'前段(≤2018)':<12} {'后段(≥2019)':<12} {'差值':<10} {'p值':<10} {'显著性'}")
print("-" * 75)
for ind, name in zip(indicators, names):
    v1 = y1[ind].mean()
    v2 = y2[ind].mean()
    t, p = stats.ttest_ind(y1[ind].dropna(), y2[ind].dropna())
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"{name:<15} {v1:<12.2f} {v2:<12.2f} {v2-v1:<+10.2f} {p:<10.4f} {sig}")

# ═════════════════════════════════════════════════════════════
# 7. 保存结果
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("7. 保存结果")
print("=" * 70)

gra_df.to_csv(os.path.join(DATA_DIR, "月度灰色关联分析.csv"), index=False, encoding="utf-8-sig")
yearly.to_csv(os.path.join(DATA_DIR, "年度旅游气候汇总.csv"), index=False, encoding="utf-8-sig")
monthly_avg.reset_index().to_csv(os.path.join(DATA_DIR, "月度游客量指数.csv"), index=False, encoding="utf-8-sig")

print("结果已保存!")
print("\n分析完成！")
