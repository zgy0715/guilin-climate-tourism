# -*- coding: utf-8 -*-
"""
桂林气候舒适度优化模型 —— 基于 OWCI 论文改进
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
import os

# ─── 导入公共模块 ─────────────────────────────────────────────
from cci_core import (
    PROJECT_ROOT, DATA_DIR,
    read_weather_data, extract_temp, weather_score,
    wind_score_simple, wind_score_owci,
)

os.makedirs(DATA_DIR, exist_ok=True)

# ═════════════════════════════════════════════════════════════
# 1. 读取数据
# ═════════════════════════════════════════════════════════════
print("正在读取天气数据...")
df = read_weather_data()

# ═════════════════════════════════════════════════════════════
# 2. 优化的温度舒适度（OWCI 思路）
# ═════════════════════════════════════════════════════════════
def comfort_temp(tavg, tmax):
    """基于 OWCI 的温度舒适度：最适 21°C，高温时最高温非线性惩罚"""
    base = np.exp(-((tavg - 21) / 8) ** 2)
    tmax_penalty = 1 / (1 + np.exp(0.5 * (tmax - 35)))
    return base * tmax_penalty

df["S_temp"] = comfort_temp(df["Tavg"], df["Tmax"])

# ═════════════════════════════════════════════════════════════
# 3. 天气评分（使用公共模块 weather_score）
# ═════════════════════════════════════════════════════════════
df["S_weather"] = (
    df["day_weather"].apply(weather_score)
    + df["night_weather"].apply(weather_score)
) / 2

# ═════════════════════════════════════════════════════════════
# 4. 风速评分（OWCI 思路）
# ═════════════════════════════════════════════════════════════
df["S_wind"] = [wind_score_owci(t, w) for t, w in zip(df["Tmax"], df["wind_power"])]

# ═════════════════════════════════════════════════════════════
# 5. 熵权法确定权重
# ═════════════════════════════════════════════════════════════
print("\n正在计算熵权法权重...")
X = df[["S_temp", "S_weather", "S_wind"]].values
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-6)
P = X / X.sum(axis=0)
E = -np.sum(P * np.log(P + 1e-6), axis=0) / np.log(len(df))
w = (1 - E) / (1 - E).sum()
print(f"熵权: S_temp={w[0]:.4f}, S_weather={w[1]:.4f}, S_wind={w[2]:.4f}")

# ═════════════════════════════════════════════════════════════
# 6. TOPSIS 计算 CCI
# ═════════════════════════════════════════════════════════════
print("正在计算 TOPSIS...")
Z = X * w
Z_plus  = Z.max(axis=0)
Z_minus = Z.min(axis=0)

D_plus  = np.sqrt(((Z - Z_plus) ** 2).sum(axis=1))
D_minus = np.sqrt(((Z - Z_minus) ** 2).sum(axis=1))
df["CCI"] = D_minus / (D_plus + D_minus) * 100

print(f"CCI 范围: {df['CCI'].min():.2f} ~ {df['CCI'].max():.2f}")
print(f"CCI 均值: {df['CCI'].mean():.2f}")

# ═════════════════════════════════════════════════════════════
# 7. 对比旧模型
# ═════════════════════════════════════════════════════════════
df["CCI_old"] = 100 * (
    0.60 * np.exp(-((df["Tavg"] - 22) / 7.5) ** 2)
    + 0.25 * df["S_weather"]
    + 0.15 * df["S_wind"]
)
print(f"\n旧 CCI 范围: {df['CCI_old'].min():.2f} ~ {df['CCI_old'].max():.2f}")
print(f"旧 CCI 均值: {df['CCI_old'].mean():.2f}")

# ═════════════════════════════════════════════════════════════
# 8. 月度分析
# ═════════════════════════════════════════════════════════════
monthly = df.groupby("Month").agg(
    CCI_new=("CCI", "mean"),
    CCI_old=("CCI_old", "mean"),
    Tmax=("Tmax", "mean"),
    Tavg=("Tavg", "mean"),
).round(2)
print("\n月度 CCI 对比:")
print(monthly)

# ═════════════════════════════════════════════════════════════
# 9. 保存结果
# ═════════════════════════════════════════════════════════════
df.to_csv(os.path.join(DATA_DIR, "优化CCI.csv"), index=False, encoding="utf-8-sig")
monthly.to_csv(os.path.join(DATA_DIR, "月度对比.csv"), encoding="utf-8-sig")
print(f"\n数据已保存到: {DATA_DIR}")
print("\n优化完成!")
