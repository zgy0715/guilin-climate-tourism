# -*- coding: utf-8 -*-
"""
CCI核心模块 —— 公共函数库
被 complete_analysis.py / charts_optimized.py / problem3_analysis.py
  / final_analysis.py / optimized_CCI.py 统一引用

用法:
    from cci_core import PROJECT_ROOT, DATA_DIR, read_weather_data, calc_cci, ...
"""
import os
import re
import numpy as np
import pandas as pd

# ─── 路径常量（均相对于本模块位置自动计算）────────────────────────────
PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)
DATA_DIR   = os.path.join(PROJECT_ROOT, "数据")
IMG_DIR    = os.path.join(PROJECT_ROOT, "图片")
B_DIR      = os.path.join(PROJECT_ROOT, "B题")
PAPER_DIR  = os.path.join(PROJECT_ROOT, "论文")

# ─── 温度提取 ────────────────────────────────────────────────
def extract_temp(x):
    """从天气记录的字符串中提取数字温度值"""
    m = re.search(r"-?\d+", str(x))
    return float(m.group()) if m else np.nan

# ─── 天气评分 ────────────────────────────────────────────────
def weather_score(x):
    """白天/夜间天气 → 0-1 舒适度评分"""
    for k, v in {"晴": 1.0, "多云": 0.95, "阴": 0.8,
                 "小雨": 0.65, "阵雨": 0.6, "中雨": 0.45,
                 "大雨": 0.25, "暴雨": 0.1, "雷阵雨": 0.35}.items():
        if k in str(x):
            return v
    return 0.7

# ─── 风力评分（简单版）─────────────────────────────────────────
def wind_score_simple(x):
    """风力描述 → 0-1 舒适度评分（适用于 CCI）"""
    s = str(x)
    if "微风" in s or "1级" in s or "2级" in s:
        return 1.0
    elif "3级" in s:
        return 0.85
    elif "4级" in s:
        return 0.65
    elif "5级" in s:
        return 0.45
    return 0.7

# ─── 风力评分（OWCI 版）────────────────────────────────────────
def wind_score_owci(tmax, wind_str):
    """OWCI 风力评分（考虑温度对风效的调节）"""
    s = str(wind_str)
    if "微风" in s or "1级" in s or "2级" in s:
        V = 1.5
    elif "3级" in s:
        V = 3.5
    elif "4级" in s:
        V = 5.5
    elif "5级" in s:
        V = 7.5
    else:
        V = 2.0
    numerator   = (tmax - 26) * (tmax - 36)
    denominator = np.abs(tmax - 31) + 6.0
    score = 0.5 + (numerator / denominator * np.sqrt(V)) / 20
    return np.clip(score, 0.1, 1.0)

# ─── 读取天气数据（通用）───────────────────────────────────────
def read_weather_data():
    """
    返回 DataFrame，包含列:
    Date, Year, Month, Quarter,
    Tmax, Tmin, Tavg,
    day_weather, night_weather, wind_power,
    raw_date, tmax, tmin, wind_dir
    """
    raw = pd.read_excel(
        os.path.join(B_DIR, "附件2：2011-2025桂林天气记录（每日）.xlsx"),
        sheet_name="Sheet1", header=None,
    )
    df = raw[pd.to_numeric(raw[0], errors="coerce").notna()].copy()
    df.columns = [
        "raw_date", "tmax", "tmin",
        "day_weather", "night_weather",
        "wind_dir", "wind_power",
    ]
    df["Tmax"]  = df["tmax"].apply(extract_temp)
    df["Tmin"]  = df["tmin"].apply(extract_temp)
    df = df.dropna(subset=["Tmax", "Tmin"]).reset_index(drop=True)

    df["Date"]    = pd.date_range("2011-01-01", periods=len(df), freq="D")
    df["Year"]    = df["Date"].dt.year
    df["Month"]   = df["Date"].dt.month
    df["Quarter"] = df["Date"].dt.quarter
    df["Tavg"]    = (df["Tmax"] + df["Tmin"]) / 2
    return df

# ─── 为 df 添加舒适度评分列 ────────────────────────────────────
def add_comfort_scores(df):
    """原地添加 S_temp, S_weather, S_wind 三列"""
    df["S_temp"] = np.exp(-((df["Tavg"] - 22) / 7.5) ** 2)
    df["S_weather"] = (
        df["day_weather"].apply(weather_score)
        + df["night_weather"].apply(weather_score)
    ) / 2
    df["S_wind"] = df["wind_power"].apply(wind_score_simple)
    return df

# ─── 熵权 TOPSIS 法计算 CCI ──────────────────────────────────
def calc_cci(df, score_cols=None):
    """
    基于已有 S_temp / S_weather / S_wind 列计算 CCI。
    返回 (df, weights)，df 新增 'CCI' 列，weights 为长度为 3 的数组。
    """
    if score_cols is None:
        score_cols = ["S_temp", "S_weather", "S_wind"]
    X = df[score_cols].values

    X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-6)
    P = X_norm / X_norm.sum(axis=0)
    E = -np.sum(P * np.log(P + 1e-6), axis=0) / np.log(len(df))
    weights = (1 - E) / (1 - E).sum()

    Z = X_norm * weights
    D_plus  = np.sqrt(((Z - Z.max(axis=0)) ** 2).sum(axis=1))
    D_minus = np.sqrt(((Z - Z.min(axis=0)) ** 2).sum(axis=1))
    df["CCI"] = D_minus / (D_plus + D_minus) * 100
    return df, weights

# ─── 百分位数法等级划分 ──────────────────────────────────────
def classify_percentile(series):
    """
    将连续值序列按 P10 / P30 / P70 / P90 切为 5 档。
    返回 Series（字符串标签）。
    注意：CCI 越大越舒适，调用时请传入 (100 - series) 以反转。
    """
    p10, p30, p70, p90 = series.quantile([0.1, 0.3, 0.7, 0.9])
    conditions = [
        series <= p10,
        (series > p10) & (series <= p30),
        (series > p30) & (series <= p70),
        (series > p70) & (series <= p90),
        series > p90,
    ]
    choices = ["1-最舒适", "2-较舒适", "3-正常", "4-较不舒适", "5-最不舒适"]
    return pd.Series(np.select(conditions, choices, default="3-正常"), index=series.index)

# ─── 风力映射为风速（用于SI公式）─────────────────────────────
def wind_to_speed(x):
    """风力描述 → 近似风速(m/s)，参考 Beaufort 风级与气象规范"""
    s = str(x)
    if "微风" in s:
        return 1.5
    elif "1-2级" in s or "1级" in s or "2级" in s:
        return 1.5
    elif "3级" in s or "3-4级" in s:
        return 4.0
    elif "4级" in s or "4-5级" in s:
        return 6.0
    elif "5级" in s or "5-6级" in s:
        return 8.5
    return 2.0  # 默认轻风


# ─── SI 指数（简化版，无湿度）──────────────────────────────────
def calc_si(df):
    """
    计算 SI 指数简化版。
    原公式: SI = 0.68×|Tm-24| + 0.07×|Hu-70| + 0.5×|V-2.0|
    简化版: SI = 0.68×|Tavg-24| + 0.5×|V-2.0|  （无湿度数据）

    SI 值越小越舒适；SI_score 为正向指标（越大越舒适）。
    原地添加 'wind_speed', 'SI', 'SI_score' 三列。
    """
    df["wind_speed"] = df["wind_power"].apply(wind_to_speed)
    df["SI"] = 0.68 * np.abs(df["Tavg"] - 24) + 0.5 * np.abs(df["wind_speed"] - 2.0)
    si_max = df["SI"].max()
    df["SI_score"] = 100 - df["SI"] / si_max * 100  # si_max 恒 > 0
    return df


# ─── 一键读取 + 评分 + CCI ──────────────────────────────────
def load_and_calc_cci():
    """读取天气数据 → 加评分列 → 计算 CCI → 返回 (df, weights)"""
    df = read_weather_data()
    df = add_comfort_scores(df)
    df, weights = calc_cci(df)
    return df, weights
