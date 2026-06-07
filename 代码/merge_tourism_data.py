#!/usr/bin/env python3
"""
整合2013-2025年桂林旅游数据，生成统一的月度和年度数据文件
"""
import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "数据")

# ═════════════════════════════════════════════════════════════
# 1. 读取各年份旅游数据
# ═════════════════════════════════════════════════════════════
print("读取各年份旅游数据...")

files = {
    2013: "2013年桂林旅游数据.csv",
    2014: "2014年桂林旅游数据.csv",
    2015: "2015年桂林旅游数据.csv",
    2016: "2016年桂林旅游数据.csv",
    2017: "2017年桂林旅游数据.csv",
    2018: "2018年桂林旅游数据.csv",
    2019: "2019年桂林旅游数据.csv",
    2020: "2020年桂林旅游数据.csv",
    2021: "2021年桂林旅游数据.csv",
    2022: "2022年桂林旅游数据.csv",
    2023: "2023年桂林旅游数据.csv",
    2024: "2024年桂林旅游数据.csv",
    2025: "2025年桂林旅游数据.csv",
}

all_monthly = []
all_quarterly = []

for year, fname in files.items():
    fpath = os.path.join(DATA_DIR, fname)
    if not os.path.exists(fpath):
        print(f"  [跳过] {fname} 不存在")
        continue
    df = pd.read_csv(fpath, on_bad_lines="skip")
    # 统一列名
    if "tourists万人次" in df.columns:
        df = df.rename(columns={"tourists万人次": "tourists", "revenue亿元": "revenue"})
    # 只取月度行（Month为数字1-12）
    monthly = df[df["Month"].apply(lambda x: str(x).strip().isdigit() and 1 <= int(str(x).strip()) <= 12)].copy()
    monthly["Year"] = year
    all_monthly.append(monthly[["Year", "Quarter", "Month", "tourists", "revenue"]])
    # 取季度行（Month为-）
    quarterly = df[df["Month"].astype(str).str.strip() == "-"].copy()
    quarterly["Year"] = year
    all_quarterly.append(quarterly[["Year", "Quarter", "Month", "tourists", "revenue"]])
    print(f"  [OK] {year}: {len(monthly)}个月度 + {len(quarterly)}个季度")

# ═════════════════════════════════════════════════════════════
# 2. 合并月度数据
# ═════════════════════════════════════════════════════════════
print("\n合并月度数据...")
monthly_df = pd.concat(all_monthly, ignore_index=True)
monthly_df["Month"] = monthly_df["Month"].astype(int)
monthly_df["Quarter"] = monthly_df["Quarter"].astype(int)

# 确保数值列为数值类型
for col in ["tourists", "revenue"]:
    if col in monthly_df.columns:
        monthly_df[col] = pd.to_numeric(monthly_df[col], errors="coerce")

# 按年月排序
monthly_df = monthly_df.sort_values(["Year", "Month"]).reset_index(drop=True)

# 保存月度数据
monthly_path = os.path.join(DATA_DIR, "月度旅游数据_2013-2025.csv")
monthly_df[["Year", "Quarter", "Month", "tourists", "revenue"]].to_csv(
    monthly_path, index=False, encoding="utf-8-sig"
)
print(f"月度数据已保存: {monthly_path} ({len(monthly_df)}行)")

# 统计覆盖情况
print(f"\n月度数据覆盖:")
for y in range(2013, 2026):
    n = len(monthly_df[monthly_df["Year"] == y])
    print(f"  {y}: {n}个月")

# ═════════════════════════════════════════════════════════════
# 3. 合并季度数据
# ═════════════════════════════════════════════════════════════
print("\n合并季度数据...")
quarterly_df = pd.concat(all_quarterly, ignore_index=True)
quarterly_df["Quarter"] = pd.to_numeric(quarterly_df["Quarter"], errors="coerce")
quarterly_df = quarterly_df.dropna(subset=["Quarter"])
quarterly_df["Quarter"] = quarterly_df["Quarter"].astype(int)

for col in ["tourists", "revenue"]:
    if col in quarterly_df.columns:
        quarterly_df[col] = pd.to_numeric(quarterly_df[col], errors="coerce")

quarterly_df = quarterly_df.sort_values(["Year", "Quarter"]).reset_index(drop=True)

quarterly_path = os.path.join(DATA_DIR, "季度旅游数据_2013-2025.csv")
quarterly_df[["Year", "Quarter", "tourists", "revenue"]].to_csv(
    quarterly_path, index=False, encoding="utf-8-sig"
)
print(f"季度数据已保存: {quarterly_path} ({len(quarterly_df)}行)")

# ═════════════════════════════════════════════════════════════
# 4. 生成年度汇总
# ═════════════════════════════════════════════════════════════
print("\n生成年度汇总...")
yearly = monthly_df.groupby("Year").agg(
    tourists_total=("tourists", "sum"),
    tourism_revenue=("revenue", "sum"),
    months_count=("Month", "count"),
).reset_index()

# 补充缺失年份的年度数据（从年度文件读取）
yearly_file = os.path.join(DATA_DIR, "年度旅游气候数据.xlsx")
if os.path.exists(yearly_file):
    yearly_official = pd.read_excel(yearly_file, usecols=["Year", "tourists_total", "tourism_revenue"])
    # 对于月度数据不完整的年份，用官方年度数据
    for _, row in yearly_official.iterrows():
        y = int(row["Year"])
        if y in yearly["Year"].values:
            idx = yearly[yearly["Year"] == y].index[0]
            if yearly.loc[idx, "months_count"] < 12:
                yearly.loc[idx, "tourists_total"] = row["tourists_total"]
                yearly.loc[idx, "tourism_revenue"] = row["tourism_revenue"]
                print(f"  {y}: 用官方年度数据替换 ({row['tourists_total']:.2f}万, {row['tourism_revenue']:.2f}亿)")
        else:
            yearly = pd.concat([yearly, pd.DataFrame([{
                "Year": y,
                "tourists_total": row["tourists_total"],
                "tourism_revenue": row["tourism_revenue"],
                "months_count": 12,
            }])], ignore_index=True)

yearly = yearly.sort_values("Year").reset_index(drop=True)
yearly_path = os.path.join(DATA_DIR, "年度旅游汇总_2013-2025.csv")
yearly.to_csv(yearly_path, index=False, encoding="utf-8-sig")
print(f"年度汇总已保存: {yearly_path}")

print("\n年度数据:")
print(yearly[["Year", "tourists_total", "tourism_revenue"]].to_string(index=False))

# ═════════════════════════════════════════════════════════════
# 5. 读取气候数据并合并
# ═════════════════════════════════════════════════════════════
print("\n读取气候数据并合并...")
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from cci_core import read_weather_data, add_comfort_scores, calc_cci

weather = read_weather_data()
weather = add_comfort_scores(weather)
weather_df, weights = calc_cci(weather)

# 计算月度气候指标
weather_df["Year"] = weather_df["Year"]
weather_df["Month"] = weather_df["Month"]
weather_df["Quarter"] = weather_df["Quarter"]

monthly_climate = weather_df.groupby(["Year", "Month"]).agg(
    Tavg=("Tavg", "mean"),
    Tmax=("Tmax", "mean"),
    Tmin=("Tmin", "mean"),
    rain_days=("day_weather", lambda x: sum("雨" in str(i) for i in x)),
    hot_days=("Tmax", lambda x: sum(x >= 35)),
    cold_days=("Tmin", lambda x: sum(x <= 5)),
    CCI=("CCI", "mean"),
).reset_index()

print(f"月度气候数据: {len(monthly_climate)}行")

# 合并旅游+气候数据
merged = pd.merge(monthly_df[["Year", "Month", "tourists", "revenue"]],
                  monthly_climate,
                  on=["Year", "Month"], how="inner")
merged = merged.sort_values(["Year", "Month"]).reset_index(drop=True)

merged_path = os.path.join(DATA_DIR, "月度旅游气候合并数据.csv")
merged.to_csv(merged_path, index=False, encoding="utf-8-sig")
print(f"合并数据已保存: {merged_path} ({len(merged)}行)")

# ═════════════════════════════════════════════════════════════
# 6. 统计摘要
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("数据整合完成！")
print("=" * 60)
print(f"\n生成文件:")
print(f"  1. 月度旅游数据_2013-2025.csv ({len(monthly_df)}行)")
print(f"  2. 季度旅游数据_2013-2025.csv ({len(quarterly_df)}行)")
print(f"  3. 年度旅游汇总_2013-2025.csv ({len(yearly)}行)")
print(f"  4. 月度旅游气候合并数据.csv ({len(merged)}行)")
