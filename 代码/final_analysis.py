# -*- coding: utf-8 -*-
"""
桂林气候舒适度综合分析 —— 整合多种方法
1. OWCI：最高温非线性惩罚
2. SI：加权绝对值偏差
3. 三指数综合 + 虚拟变量回归
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import os

# ─── 导入公共模块 ─────────────────────────────────────────────
from cci_core import (
    PROJECT_ROOT, DATA_DIR, IMG_DIR,
    read_weather_data, add_comfort_scores, calc_cci,
    classify_percentile, weather_score, wind_score_simple,
    wind_score_owci,
)

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

print("=" * 60)
print("桂林气候舒适度综合分析")
print("=" * 60)

# ═════════════════════════════════════════════════════════════
# 1. 读取数据
# ═════════════════════════════════════════════════════════════

df = read_weather_data()
print(f"数据范围: {df['Date'].min().date()} ~ {df['Date'].max().date()}")
print(f"数据量: {len(df)} 天")

# ═════════════════════════════════════════════════════════════
# 2. 多种舒适度指数计算
# ═════════════════════════════════════════════════════════════
print("\n计算气候舒适度指数...")

# --- 方法1: 原始 CCI（熵权 TOPSIS）---
df = add_comfort_scores(df)
df, w_entropy = calc_cci(df)

# --- 方法2: SI 公式（改进版，无湿度）---
df["SI"]      = 0.68 * np.abs(df["Tavg"] - 24) + 0.5 * np.abs(df["S_wind"] * 5 - 2)
df["SI_score"] = 100 - df["SI"] / df["SI"].max() * 100

# --- 方法3: OWCI 思路（最高温非线性惩罚）---
df["CCI_owci"] = 100 * (
    np.exp(-((df["Tavg"] - 21) / 8) ** 2)
    * (1 / (1 + np.exp(0.5 * (df["Tmax"] - 35)))) * 0.6
    + df["S_weather"] * 0.15
    + np.array([wind_score_owci(t, w) for t, w in zip(df["Tmax"], df["wind_power"])]) * 0.25
)

# ═════════════════════════════════════════════════════════════
# 3. 百分位数法等级划分
# ═════════════════════════════════════════════════════════════
print("进行等级划分...")
df["CCI_level"]  = classify_percentile(100 - df["CCI"])
df["SI_level"]   = classify_percentile(df["SI"])
df["OWCI_level"] = classify_percentile(100 - df["CCI_owci"])

for col in ["CCI_level", "SI_level", "OWCI_level"]:
    print(f"\n{col}:")
    print(df[col].value_counts().sort_index())

# ═════════════════════════════════════════════════════════════
# 4. 月度分析
# ═════════════════════════════════════════════════════════════
monthly = df.groupby("Month").agg(
    CCI=("CCI", "mean"), SI=("SI", "mean"),
    OWCI=("CCI_owci", "mean"),
    Tmax=("Tmax", "mean"), Tavg=("Tavg", "mean"),
    Days=("Date", "count"),
).round(2)

best_cci  = monthly["CCI"].idxmax()
best_owci = monthly["OWCI"].idxmax()
print(f"\n最舒适月份(CCI): {best_cci}月 (CCI={monthly.loc[best_cci, 'CCI']:.2f})")
print(f"最舒适月份(OWCI): {best_owci}月 (OWCI={monthly.loc[best_owci, 'OWCI']:.2f})")

# ═════════════════════════════════════════════════════════════
# 5. 时间序列趋势分析
# ═════════════════════════════════════════════════════════════
print("\n趋势分析...")
yearly = df.groupby("Year").agg(
    CCI=("CCI", "mean"), OWCI=("CCI_owci", "mean"),
    Tmax=("Tmax", "mean"), Tavg=("Tavg", "mean"),
).round(2)


def calc_trend(series):
    x = np.arange(len(series))
    slope, intercept, rv, pv, se = stats.linregress(x, series)
    return {"slope": slope, "r_squared": rv**2, "p_value": pv,
            "trend": "上升" if slope > 0 else "下降"}


cci_t  = calc_trend(yearly["CCI"])
owci_t = calc_trend(yearly["OWCI"])
print(f"CCI 年趋势: {cci_t['trend']} ({cci_t['slope']:.4f}/年), R^2={cci_t['r_squared']:.4f}")
print(f"OWCI 年趋势: {owci_t['trend']} ({owci_t['slope']:.4f}/年), R^2={owci_t['r_squared']:.4f}")

df["CCI_9ma"]  = df["CCI"].rolling(window=9 * 365, center=True).mean()
df["OWCI_9ma"] = df["CCI_owci"].rolling(window=9 * 365, center=True).mean()

# ═════════════════════════════════════════════════════════════
# 6. 灰色关联分析（使用真实年度旅游数据）
# ═════════════════════════════════════════════════════════════
print("\n灰色关联分析...")

# 读取真实旅游数据
yearly_tourism = pd.read_excel(
    os.path.join(DATA_DIR, "年度旅游气候数据.xlsx"),
    usecols=["Year", "tourists_total", "tourism_revenue"],
)
has_real_data = True

monthly_all = df.groupby(["Year", "Month"]).agg(
    CCI=("CCI", "mean"),
    Tmax=("Tmax", "mean"),
    Tavg=("Tavg", "mean"),
    rain_days=("day_weather", lambda x: sum("雨" in str(i) for i in x)),
    hot_days=("Tmax", lambda x: sum(x >= 35)),
).reset_index()


def grey_relation(reference, factors, rho=0.5):
    ref = np.array(reference)
    fac = np.array(factors)
    scaler = MinMaxScaler()
    ref_norm = scaler.fit_transform(ref.reshape(-1, 1)).flatten()
    fac_norm = scaler.fit_transform(fac)
    diff = np.abs(fac_norm - ref_norm.reshape(-1, 1))
    d_min, d_max = diff.min(), diff.max()
    coef = (d_min + rho * d_max) / (diff + rho * d_max)
    return coef.mean(axis=0)


yearly_climate = df.groupby("Year").agg(
    CCI_mean=("CCI", "mean"),
    Tmax_mean=("Tmax", "mean"),
    Tavg_mean=("Tavg", "mean"),
    rain_days=("day_weather", lambda x: sum("雨" in str(i) for i in x)),
    hot_days=("Tmax", lambda x: sum(x >= 35)),
    cold_days=("Tmin", lambda x: sum(x <= 5)),
).reset_index()

yearly_merged = pd.merge(yearly_tourism, yearly_climate, on="Year")
factors_df = yearly_merged[["CCI_mean", "Tmax_mean", "Tavg_mean", "rain_days", "hot_days", "cold_days"]]
relation = grey_relation(yearly_merged["tourists_total"], factors_df)

gra_result = pd.DataFrame({
    "因素": ["CCI(熵权)", "最高气温", "平均气温", "降雨天数", "高温天数", "低温天数"],
    "关联系数": relation.round(4),
}).sort_values("关联系数", ascending=False)
print("\n灰色关联分析结果:")
print(gra_result.to_string(index=False))

# ═════════════════════════════════════════════════════════════
# 7. 虚拟变量回归分析
# ═════════════════════════════════════════════════════════════
print("\n虚拟变量回归分析...")

monthly_all["summer_7"]     = (monthly_all["Month"] == 7).astype(int)
monthly_all["summer_8"]     = (monthly_all["Month"] == 8).astype(int)
monthly_all["golden_week"]  = (monthly_all["Month"] == 10).astype(int)
monthly_all["spring_feast"] = monthly_all["Month"].isin([1, 2]).astype(int)

monthly_all["holiday_score"] = (
    monthly_all["summer_7"] * 0.5
    + monthly_all["summer_8"] * 1.0
    + monthly_all["golden_week"] * 1.0
    + monthly_all["spring_feast"] * (-1.0)
)

X = sm.add_constant(monthly_all[["CCI", "Tmax", "rain_days", "hot_days", "holiday_score"]])
# 注意：此脚本的月度数据中无旅游指标，此处为自回归演示
# 正式分析请使用 complete_analysis.py 中的虚拟变量回归（含季度旅游数据）
y = monthly_all["CCI"]  # 演示用：CCI自回归，R²无实际意义

model = sm.OLS(y, X).fit()
print(f"R^2 = {model.rsquared:.4f}, 调整R^2 = {model.rsquared_adj:.4f}")



# ═════════════════════════════════════════════════════════════
# 8. 保存结果
# ═════════════════════════════════════════════════════════════
print("\n保存结果...")
df.to_csv(os.path.join(DATA_DIR, "每日综合数据.csv"), index=False, encoding="utf-8-sig")
monthly.to_csv(os.path.join(DATA_DIR, "每月综合数据.csv"), encoding="utf-8-sig")
yearly.to_csv(os.path.join(DATA_DIR, "年度趋势.csv"), encoding="utf-8-sig")
gra_result.to_csv(os.path.join(DATA_DIR, "灰色关联结果.csv"), index=False, encoding="utf-8-sig")

with open(os.path.join(DATA_DIR, "回归结果.txt"), "w", encoding="utf-8") as f:
    f.write(str(model.summary()))

summary = pd.DataFrame({
    "指标": ["CCI均值", "CCI范围", "SI均值", "OWCI均值", "最舒适月份", "年变化趋势"],
    "值": [
        f"{df['CCI'].mean():.2f}",
        f"{df['CCI'].min():.2f} ~ {df['CCI'].max():.2f}",
        f"{df['SI'].mean():.2f}",
        f"{df['CCI_owci'].mean():.2f}",
        f"{best_cci}月",
        f"CCI {cci_t['trend']}, OWCI {owci_t['trend']}",
    ],
})
summary.to_csv(os.path.join(DATA_DIR, "analysis_summary.csv"), index=False, encoding="utf-8-sig")

print(f"\n所有结果已保存到: {DATA_DIR}")
print("=" * 60)
print("分析完成!")
print("=" * 60)
