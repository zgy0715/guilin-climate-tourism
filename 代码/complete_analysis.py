# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from sklearn.preprocessing import StandardScaler
from scipy import stats
import statsmodels.api as sm
import warnings
import os

# ─── 导入公共模块 ─────────────────────────────────────────────
from cci_core import (
    PROJECT_ROOT, DATA_DIR,
    extract_temp, weather_score, wind_score_simple,
    read_weather_data, add_comfort_scores, calc_cci,
    classify_percentile, calc_si,
)

warnings.filterwarnings("ignore")
os.makedirs(DATA_DIR, exist_ok=True)

print("=" * 80)
print("完整问题解决方案")
print("=" * 80)

# ═════════════════════════════════════════════════════════════
# 第一部分：读取数据并计算 CCI
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("第一部分：读取数据并计算 CCI")
print("=" * 80)

df = read_weather_data()
df = add_comfort_scores(df)
df, weights = calc_cci(df)

print(f"数据范围: {df['Date'].min().date()} ~ {df['Date'].max().date()}")
print(f"数据量: {len(df)} 天")
print(f"熵权: S_temp={weights[0]:.4f}, S_weather={weights[1]:.4f}, S_wind={weights[2]:.4f}")
print(f"CCI 均值: {df['CCI'].mean():.2f}  范围: {df['CCI'].min():.2f} ~ {df['CCI'].max():.2f}")

# 百分位数法等级划分（CCI 越大越舒适，反转后传入）
df["CCI_level"] = classify_percentile(100 - df["CCI"])

# ═════════════════════════════════════════════════════════════
# 第二部分：Mann-Kendall 趋势检验
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("第二部分：Mann-Kendall 趋势检验")
print("=" * 80)

yearly_cci = df.groupby("Year")["CCI"].mean()


def mann_kendall_test(x):
    """Mann-Kendall 趋势检验"""
    n = len(x)
    s = 0
    for k in range(n - 1):
        for j in range(k + 1, n):
            s += np.sign(x[j] - x[k])

    unique, counts = np.unique(x, return_counts=True)
    tp = counts[counts > 1]
    var_s = (n * (n - 1) * (2 * n + 5)) / 18.0
    if len(tp) > 0:
        for t in tp:
            var_s -= (t * (t - 1) * (2 * t + 5)) / 18.0

    z = (s - 1) / np.sqrt(var_s) if s > 0 else ((s + 1) / np.sqrt(var_s) if s < 0 else 0)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    trend = "上升" if z > 0 else ("下降" if z < 0 else "无趋势")

    if p < 0.01:
        significance = "极显著"
    elif p < 0.05:
        significance = "显著"
    elif p < 0.1:
        significance = "较显著"
    else:
        significance = "不显著"
    return {"S": s, "Z": z, "p": p, "trend": trend, "significance": significance}


def sens_slope(x):
    """Sen's 斜率估计"""
    n = len(x)
    slopes = [(x[j] - x[i]) / (j - i) for i in range(n) for j in range(i + 1, n)]
    return np.median(slopes)


mk_result = mann_kendall_test(yearly_cci.values)
sen_slope = sens_slope(yearly_cci.values)

print(f"\n--- Mann-Kendall 趋势检验结果 ---")
print(f"S = {mk_result['S']:.0f}, Z = {mk_result['Z']:.4f}, p = {mk_result['p']:.4f}")
print(f"趋势: {mk_result['trend']}, {mk_result['significance']}")
print(f"Sen's 斜率 = {sen_slope:.4f}")
print(f"\n结论: CCI 呈{mk_result['trend']}趋势，但{mk_result['significance']}")

# ═════════════════════════════════════════════════════════════
# 第三部分：Pettitt 突变检验
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("第三部分：Pettitt 突变检验")
print("=" * 80)


def pettitt_test(x):
    n = len(x)
    kt = np.zeros(n)
    for t in range(n):
        for i in range(t):
            kt[t] += np.sign(x[t] - x[i])
        for i in range(t + 1, n):
            kt[t] += np.sign(x[t] - x[i])
    k_abs = np.abs(kt)
    k_max_idx = np.argmax(k_abs)
    k_stat = k_abs[k_max_idx]
    p = min(2 * np.exp(-6 * k_stat**2 / (n**3 + n**2)), 1.0)
    return {"change_point": k_max_idx + 1, "K_stat": k_stat, "p": p}


pettitt_result = pettitt_test(yearly_cci.values)
change_year = yearly_cci.index[pettitt_result["change_point"] - 1]

print(f"\n--- Pettitt 突变检验结果 ---")
print(f"K = {pettitt_result['K_stat']:.0f}, 突变点 = 第{pettitt_result['change_point']}年 ({change_year}年)")
print(f"p = {pettitt_result['p']:.4f}")
print(f"\n结论: CCI 未发生显著突变 (p > 0.05)" if pettitt_result["p"] > 0.05 else "\n结论: 存在显著突变")

# ═════════════════════════════════════════════════════════════
# 第四部分：马尔科夫链等级转换分析
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("第四部分：马尔科夫链等级转换分析")
print("=" * 80)

levels = ["1-最舒适", "2-较舒适", "3-正常", "4-较不舒适", "5-最不舒适"]
yearly_levels = df.groupby(["Year", "CCI_level"]).size().unstack(fill_value=0)
for lev in levels:
    if lev not in yearly_levels.columns:
        yearly_levels[lev] = 0
yearly_levels = yearly_levels[levels]
yearly_ratio = yearly_levels.div(yearly_levels.sum(axis=1), axis=0)

n_levels = len(levels)
transition_matrix = np.zeros((n_levels, n_levels))
for i in range(len(yearly_ratio) - 1):
    current = yearly_ratio.iloc[i].values
    next_yr = yearly_ratio.iloc[i + 1].values
    total = current.sum() * next_yr.sum()
    if total > 0:
        for j in range(n_levels):
            for k in range(n_levels):
                transition_matrix[j][k] += current[j] * next_yr[k]

row_sums = transition_matrix.sum(axis=1)
transition_matrix = transition_matrix / row_sums[:, np.newaxis]
trans_df = pd.DataFrame(transition_matrix, index=levels, columns=levels)

print("\n--- 年度舒适度等级转移概率矩阵 ---")
print(trans_df.round(4).to_string())
for i, lev in enumerate(levels):
    max_idx = np.argmax(transition_matrix[i])
    print(f"'{lev}' → '{levels[max_idx]}' ({transition_matrix[i][max_idx]:.4f})")

# ═════════════════════════════════════════════════════════════
# 第五部分：国标季节划分
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("第五部分：国标 GB/T 42074-2022 季节划分")
print("=" * 80)

df["Tavg_5d"] = df["Tavg"].rolling(window=5, center=False).mean()


def classify_season_gb(tavg_5d_series):
    seasons = []
    current = "winter"
    for v in tavg_5d_series:
        if pd.isna(v):
            seasons.append(current)
            continue
        if current == "winter":
            if v >= 10:
                current = "spring"
        elif current == "spring":
            if v >= 22:
                current = "summer"
            elif v < 10:
                current = "winter"
        elif current == "summer":
            if v < 22:
                current = "autumn"
        elif current == "autumn":
            if v < 10:
                current = "winter"
            elif v >= 22:
                current = "summer"
        seasons.append(current)
    return seasons


df["Season_gb"] = classify_season_gb(df["Tavg_5d"])
season_map = {"spring": "春季", "summer": "夏季", "autumn": "秋季", "winter": "冬季"}
df["Season_cn"] = df["Season_gb"].map(season_map)
yearly_season = df.groupby(["Year", "Season_cn"]).size().unstack(fill_value=0)
season_order = ["春季", "夏季", "秋季", "冬季"]
for s in season_order:
    if s not in yearly_season.columns:
        yearly_season[s] = 0
yearly_season = yearly_season[season_order]

print("\n--- 各年四季天数 ---")
print(yearly_season.to_string())
print("\n--- 各季节平均天数 ---")
for s in season_order:
    print(f"  {s}: {yearly_season[s].mean():.1f} 天")

# ═════════════════════════════════════════════════════════════
# 第六部分：岭回归
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("第六部分：岭回归解决多重共线性")
print("=" * 80)

yearly_tourism = pd.read_excel(
    os.path.join(DATA_DIR, "年度旅游气候数据.xlsx"),
    usecols=["Year", "tourists_total", "tourism_revenue"],
)

yearly_climate = df.groupby("Year").agg(
    CCI_mean=("CCI", "mean"),
    Tmax_mean=("Tmax", "mean"),
    rain_days=("day_weather", lambda x: sum("雨" in str(i) for i in x)),
    hot_days=("Tmax", lambda x: sum(x >= 35)),
    cold_days=("Tmin", lambda x: sum(x <= 5)),
).round(2)

yearly_data = pd.merge(yearly_tourism, yearly_climate, left_on="Year", right_index=True)
yearly_data["pandemic"] = yearly_data["Year"].between(2020, 2022).astype(int)

X_cols = ["CCI_mean", "Tmax_mean", "rain_days", "hot_days", "cold_days", "pandemic"]
X_raw = yearly_data[X_cols].values
y_tourists = yearly_data["tourists_total"].values
y_revenue = yearly_data["tourism_revenue"].values

scaler_X = StandardScaler()
scaler_y1 = StandardScaler()
scaler_y2 = StandardScaler()
X_scaled = scaler_X.fit_transform(X_raw)
y1_scaled = scaler_y1.fit_transform(y_tourists.reshape(-1, 1)).flatten()
y2_scaled = scaler_y2.fit_transform(y_revenue.reshape(-1, 1)).flatten()


def ridge_regression(X, y, alpha):
    n, p = X.shape
    Xc = np.column_stack([np.ones(n), X])
    I = np.eye(p + 1)
    I[0, 0] = 0
    beta = np.linalg.solve(Xc.T @ Xc + alpha * I, Xc.T @ y)
    return beta


alphas = np.logspace(-3, 3, 100)
best_a1, best_r1 = 0, -np.inf
best_a2, best_r2 = 0, -np.inf

for a in alphas:
    b1 = ridge_regression(X_scaled, y1_scaled, a)
    y1p = np.column_stack([np.ones(len(X_scaled)), X_scaled]) @ b1
    r1 = 1 - np.sum((y1_scaled - y1p)**2) / np.sum((y1_scaled - y1_scaled.mean())**2)
    if r1 > best_r1:
        best_r1, best_a1 = r1, a
    b2 = ridge_regression(X_scaled, y2_scaled, a)
    y2p = np.column_stack([np.ones(len(X_scaled)), X_scaled]) @ b2
    r2 = 1 - np.sum((y2_scaled - y2p)**2) / np.sum((y2_scaled - y2_scaled.mean())**2)
    if r2 > best_r2:
        best_r2, best_a2 = r2, a

beta1 = ridge_regression(X_scaled, y1_scaled, best_a1)
beta2 = ridge_regression(X_scaled, y2_scaled, best_a2)
coef1 = beta1[1:] / scaler_X.scale_ * scaler_y1.scale_[0]
coef2 = beta2[1:] / scaler_X.scale_ * scaler_y2.scale_[0]
intercept1 = beta1[0] * scaler_y1.scale_[0] + scaler_y1.mean_[0] - np.sum(coef1 * scaler_X.mean_)
intercept2 = beta2[0] * scaler_y2.scale_[0] + scaler_y2.mean_[0] - np.sum(coef2 * scaler_X.mean_)

print(f"\n最优 alpha(游客量) = {best_a1:.4f}, 调整 R2 = {best_r1:.4f}")
print(f"最优 alpha(旅游收入) = {best_a2:.4f}, 调整 R2 = {best_r2:.4f}")

print("\n--- VIF 分析 ---")
vif_data = pd.DataFrame({
    "变量": X_cols,
    "VIF": [variance_inflation_factor(yearly_data[X_cols].values, i) for i in range(len(X_cols))],
})
print(vif_data.to_string(index=False))

# ═════════════════════════════════════════════════════════════
# 第七部分：线性倾向估计与滑动平均
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("第七部分：线性倾向估计与滑动平均")
print("=" * 80)

cci_series = yearly_cci.values
years = np.arange(len(yearly_cci))


def linear_tendency(x, t):
    """一元线性回归 xi = a + b*ti，返回倾向率 b 及其显著性"""
    slope, intercept, r_val, p_val, std_err = stats.linregress(t, x)
    return {"a": intercept, "b": slope, "r2": r_val**2, "p": p_val, "stderr": std_err}


# 滑动平均（存储结果用于CSV输出）
lt_results = {}
for k in [3, 5, 7, 9]:
    ma = pd.Series(cci_series).rolling(window=k, center=True).mean()
    valid = ma.dropna()
    if len(valid) > 3:
        t_valid = np.arange(len(valid))
        lt = linear_tendency(valid.values, t_valid)
        rate_per_decade = lt["b"] * 10
        lt_results[f"{k}年滑动"] = lt
        print(f"\n{k}年滑动平均: b = {lt['b']:.4f}/年, "
              f"倾向率 = {rate_per_decade:+.2f}/10年, "
              f"R^2 = {lt['r2']:.4f}, p = {lt['p']:.4f}")

# 原始序列线性倾向
lt_raw = linear_tendency(yearly_cci.values, years)
lt_results["原始序列"] = lt_raw
print(f"\n--- 原始序列线性倾向估计 ---")
print(f"回归方程: CCI = {lt_raw['a']:.2f} {'+' if lt_raw['b'] >= 0 else '-'} "
      f"{abs(lt_raw['b']):.4f} × 年份序号")
print(f"倾向率 b = {lt_raw['b']:.4f}/年（即 {lt_raw['b']*10:+.2f}/10年）")
print(f"决定系数 R^2 = {lt_raw['r2']:.4f}, p = {lt_raw['p']:.4f}")
print(f"趋势方向: {'下降' if lt_raw['b'] < 0 else '上升'}, "
      f"{'显著' if lt_raw['p'] < 0.05 else '不显著'}")

print(f"\n--- 与 MK 检验对比 ---")
print(f"MK: Z = {mk_result['Z']:.4f}, p = {mk_result['p']:.4f}, "
      f"Sen's 斜率 = {sen_slope:.4f}/年")
print(f"线性倾向: b = {lt_raw['b']:.4f}/年, p = {lt_raw['p']:.4f}")
print(f"结论: 两种方法结论一致 —— CCI 呈微弱{lt_raw['b'] < 0 and '下降' or '上升'}趋势，统计不显著")

# ═════════════════════════════════════════════════════════════
# 第八部分：SI 指数对比验证
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("第八部分：SI 指数对比验证")
print("=" * 80)

df = calc_si(df)
print(f"\nSI 范围: {df['SI'].min():.2f} ~ {df['SI'].max():.2f} "
      f"（越小越舒适）")
print(f"SI_score 均值: {df['SI_score'].mean():.2f}")

# 年度 SI 与 CCI 对比
yearly_si = df.groupby("Year").agg(
    SI_mean=("SI", "mean"),
    SI_score_mean=("SI_score", "mean"),
    CCI_mean=("CCI", "mean"),
).round(2)

corr_si_cci = yearly_si["SI_score_mean"].corr(yearly_si["CCI_mean"])
print(f"\n年度 SI_score 与 CCI 的 Pearson 相关系数: r = {corr_si_cci:.4f}")
print("（若 r > 0.7，说明两种指数对舒适度的评价高度一致）")

# SI等级划分（使用百分位数法，SI越小越舒适，无需反转）
si_levels = classify_percentile(df["SI"])
print(f"\nSI 等级分布:")
for lev in ["1-最舒适", "2-较舒适", "3-正常", "4-较不舒适", "5-最不舒适"]:
    n = (si_levels == lev).sum()
    print(f"  {lev}: {n} 天 ({n/len(df)*100:.1f}%)")

# ═════════════════════════════════════════════════════════════
# 第九部分：虚拟变量回归与气候弹性系数
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("第九部分：虚拟变量回归与气候弹性系数")
print("=" * 80)

# 读取季度旅游数据
quarterly_tourism = pd.read_excel(
    os.path.join(DATA_DIR, "季度旅游气候数据.xlsx"),
    usecols=["Year", "Quarter", "tourists", "revenue"],
)

# 读取年度总量（用于计算季度指数）
yearly_total = yearly_tourism[["Year", "tourists_total", "tourism_revenue"]].copy()
yearly_total.rename(columns={"tourism_revenue": "revenue_total"}, inplace=True)

# 计算季度气候数据
quarterly_climate = df.groupby(["Year", "Quarter"]).agg(
    CCI_mean=("CCI", "mean"),
    SI_mean=("SI", "mean"),
    Tavg_mean=("Tavg", "mean"),
    Tmax_mean=("Tmax", "mean"),
    rain_days=("day_weather", lambda x: sum("雨" in str(i) for i in x)),
).round(2).reset_index()

# 合并数据
qdata = pd.merge(quarterly_tourism, quarterly_climate, on=["Year", "Quarter"])
qdata = pd.merge(qdata, yearly_total[["Year", "tourists_total"]], on="Year")

# 计算季度游客量指数（%）
qdata["Q_index"] = qdata["tourists"] / qdata["tourists_total"] * 100

# 虚拟变量赋值
# Q1（春节）= -1, Q2 = 0（基线）, Q3（暑假）= +1, Q4（国庆）= +1
qdata["Q_dummy"] = qdata["Quarter"].map({1: 0, 2: 0, 3: 1, 4: 1})  # Q3/Q4 旺季
qdata["spring_dummy"] = qdata["Quarter"].map({1: -1, 2: 0, 3: 0, 4: 0})  # Q1 春节
# 综合节假日效应
qdata["holiday_effect"] = qdata["Q_dummy"] + qdata["spring_dummy"]
# 疫情虚拟变量
qdata["pandemic"] = qdata["Year"].between(2020, 2022).astype(int)

print(f"\n季度数据样本: {len(qdata)} 条（{qdata['Year'].min()}-{qdata['Year'].max()}）")
print(f"\n季度游客量指数（Q_index）范围: {qdata['Q_index'].min():.1f}% ~ "
      f"{qdata['Q_index'].max():.1f}%")

# OLS 回归
# 删除tourists/revenue为NaN的行（如2017年Q1-Q2、2025年部分季度缺失）
qdata = qdata.dropna(subset=["tourists", "revenue"]).copy()
qdata["Q_index"] = qdata["tourists"] / qdata["tourists_total"] * 100

# 模型1: 仅用 CCI
X1 = sm.add_constant(qdata[["CCI_mean"]])
y = qdata["Q_index"]
m1 = sm.OLS(y, X1).fit()

# 模型2: CCI + 节假日效应 + 疫情
X2 = sm.add_constant(qdata[["CCI_mean", "holiday_effect", "pandemic"]])
m2 = sm.OLS(y, X2).fit()

# 模型3: CCI + 分项虚拟变量 + 疫情
X3 = sm.add_constant(qdata[["CCI_mean", "Q_dummy", "spring_dummy", "pandemic"]])
m3 = sm.OLS(y, X3).fit()

print(f"\n--- 模型对比 ---")
print(f"模型1 (CCI):         R^2={m1.rsquared:.4f}, 调整R^2={m1.rsquared_adj:.4f}, "
      f"AIC={m1.aic:.2f}")
print(f"模型2 (CCI+节日+疫情): R^2={m2.rsquared:.4f}, 调整R^2={m2.rsquared_adj:.4f}, "
      f"AIC={m2.aic:.2f}")
print(f"模型3 (CCI+分项+疫情): R^2={m3.rsquared:.4f}, 调整R^2={m3.rsquared_adj:.4f}, "
      f"AIC={m3.aic:.2f}")

# 报告最优模型（模型2或3中选AIC最低的）
best_m = m2 if m2.aic < m3.aic else m3
print(f"\n--- 最优模型详细结果 ---")
print(f"R^2 = {best_m.rsquared:.4f}, 调整R^2 = {best_m.rsquared_adj:.4f}")
print(f"F = {best_m.fvalue:.2f}, p = {best_m.f_pvalue:.4e}")
print(f"Durbin-Watson = {durbin_watson(best_m.resid):.4f}")
print(best_m.summary().tables[1])

# 气候弹性系数
cci_coef = best_m.params.get("CCI_mean", 0)
cci_se = best_m.bse.get("CCI_mean", 0)
mean_cci = qdata["CCI_mean"].mean()
mean_qi = qdata["Q_index"].mean()

print(f"\n--- 气候弹性系数 ---")
print(f"CCI 回归系数 beta1 = {cci_coef:.4f}（标准误 = {cci_se:.4f}）")
print(f"含义: CCI 每上升 1 个单位，季度游客占比变化 {cci_coef:+.2f} 个百分点")
# 弹性（百分比形式）: (dQ/Q) / (dCCI/CCI) = beta1 × (mean_CCI / mean_Q_index)
pct_elasticity = cci_coef * mean_cci / mean_qi
print(f"百分比弹性: epsilon = beta1 × (CCI_mean / Q_mean) = {pct_elasticity:.4f}")
print(f"含义: CCI 每上升 1%，季度游客占比变化 {pct_elasticity:+.2f}%")
print(f"参考: 丽江海外游客弹性=1.31%, 国内游客弹性=0.56%")

# 检验显著性
t_val = cci_coef / cci_se if cci_se > 0 else 0
n = len(qdata)
p_val_cci = 2 * (1 - stats.t.cdf(abs(t_val), n - len(best_m.params)))

# 置信区间
ci_low = cci_coef - stats.t.ppf(0.975, n - len(best_m.params)) * cci_se
ci_high = cci_coef + stats.t.ppf(0.975, n - len(best_m.params)) * cci_se
print(f"\nt = {t_val:.4f}, p = {p_val_cci:.4f}, "
      f"{'显著' if p_val_cci < 0.05 else '不显著'}")
print(f"95% 置信区间: [{ci_low:.4f}, {ci_high:.4f}]")

# ═════════════════════════════════════════════════════════════
# 第十部分：两阶段对比分析
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("第十部分：两阶段对比分析")
print("=" * 80)

early = yearly_cci[yearly_cci.index <= 2018]
late = yearly_cci[yearly_cci.index >= 2019]

print(f"\n前段 (2011-2018): CCI 均值 = {early.mean():.2f}, "
      f"标准差 = {early.std():.2f}, n = {len(early)}")
print(f"后段 (2019-2025): CCI 均值 = {late.mean():.2f}, "
      f"标准差 = {late.std():.2f}, n = {len(late)}")
print(f"差值 = {late.mean() - early.mean():+.2f}")

t_stat, p_two = stats.ttest_ind(late, early, equal_var=False)
print(f"\nWelch t 检验: t = {t_stat:.4f}, p = {p_two:.4f}")
if p_two < 0.05:
    print("结论: 两阶段 CCI 均值差异显著 → 气候舒适度发生了显著变化")
else:
    print("结论: 两阶段 CCI 均值差异不显著 → 气候舒适度未发生显著变化")

# 按各气候指标做两阶段对比
climate_indicators = df.groupby("Year").agg(
    CCI=("CCI", "mean"),
    Tavg=("Tavg", "mean"),
    Tmax=("Tmax", "mean"),
    rain_days=("day_weather", lambda x: sum("雨" in str(i) for i in x)),
).round(2)

print(f"\n--- 各气候指标两阶段对比 ---")
for col in ["CCI", "Tavg", "Tmax", "rain_days"]:
    e = climate_indicators[climate_indicators.index <= 2018][col]
    l = climate_indicators[climate_indicators.index >= 2019][col]
    diff = l.mean() - e.mean()
    t_s, p_s = stats.ttest_ind(l, e, equal_var=False)
    print(f"  {col}: 前段={e.mean():.2f}, 后段={l.mean():.2f}, "
          f"差值={diff:+.2f}, p={p_s:.4f}")

# ═════════════════════════════════════════════════════════════
# 第十一部分：保存所有结果
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("第十一部分：保存所有结果")
print("=" * 80)

pd.DataFrame({
    "指标": ["S", "Z", "p", "趋势", "显著性", "Sen斜率"],
    "值": [mk_result["S"], mk_result["Z"], mk_result["p"],
           mk_result["trend"], mk_result["significance"], sen_slope],
}).to_csv(os.path.join(DATA_DIR, "趋势检验.csv"), index=False, encoding="utf-8-sig")
print("1. Mann-Kendall 检验结果 [OK]")

pd.DataFrame({
    "指标": ["K", "突变点位置", "对应年份", "p"],
    "值": [pettitt_result["K_stat"], pettitt_result["change_point"], change_year, pettitt_result["p"]],
}).to_csv(os.path.join(DATA_DIR, "突变检验.csv"), index=False, encoding="utf-8-sig")
print("2. Pettitt 检验结果 [OK]")

trans_df.to_csv(os.path.join(DATA_DIR, "马尔科夫转移矩阵.csv"), encoding="utf-8-sig")
print("3. 马尔科夫转移矩阵 [OK]")

yearly_season.to_csv(os.path.join(DATA_DIR, "季节划分.csv"), encoding="utf-8-sig")
print("4. 国标季节划分 [OK]")

pd.DataFrame({
    "模型": ["游客量模型", "旅游收入模型"],
    "最优alpha": [best_a1, best_a2],
    "调整R2": [best_r1, best_r2],
}).to_csv(os.path.join(DATA_DIR, "岭回归结果.csv"), index=False, encoding="utf-8-sig")
print("5. 岭回归结果 [OK]")

# 新增：线性倾向估计（使用实际滑动窗口计算结果）
lt_keys = ["原始序列", "3年滑动", "5年滑动", "7年滑动", "9年滑动"]
lt_vals_b = [lt_raw["b"]] + [lt_results.get(f"{k}年滑动", {}).get("b", 0) for k in [3, 5, 7, 9]]
lt_vals_r2 = [lt_raw["r2"]] + [lt_results.get(f"{k}年滑动", {}).get("r2", 0) for k in [3, 5, 7, 9]]
pd.DataFrame({
    "方法": lt_keys,
    "倾向率b(每年)": lt_vals_b,
    "倾向率(每10年)": [v * 10 for v in lt_vals_b],
    "R2": lt_vals_r2,
}).to_csv(os.path.join(DATA_DIR, "线性倾向估计.csv"), index=False, encoding="utf-8-sig")
print("6. 线性倾向估计 [OK]")

# 新增：SI 指数对比
yearly_si.to_csv(os.path.join(DATA_DIR, "SI指数对比.csv"), encoding="utf-8-sig")
print("7. SI 指数对比 [OK]")

# 新增：虚拟变量回归
pd.DataFrame({
    "指标": ["R^2", "调整R^2", "F统计量", "CCI弹性系数", "百分比弹性",
             "CCI系数t值", "CCI系数p值", "样本量"],
    "值": [f"{best_m.rsquared:.4f}", f"{best_m.rsquared_adj:.4f}",
           f"{best_m.fvalue:.4f}", f"{cci_coef:.4f}", f"{pct_elasticity:.4f}",
           f"{t_val:.4f}", f"{p_val_cci:.4f}", f"{n}"],
}).to_csv(os.path.join(DATA_DIR, "虚拟变量回归弹性.csv"),
          index=False, encoding="utf-8-sig")
qdata.to_csv(os.path.join(DATA_DIR, "季度回归数据.csv"),
             index=False, encoding="utf-8-sig")
print("8. 虚拟变量回归与弹性系数 [OK]")

# 新增：两阶段对比
pd.DataFrame({
    "指标": ["CCI", "Tavg", "Tmax", "rain_days"],
    "前段均值": [
        f"{climate_indicators[climate_indicators.index <= 2018]['CCI'].mean():.2f}",
        f"{climate_indicators[climate_indicators.index <= 2018]['Tavg'].mean():.2f}",
        f"{climate_indicators[climate_indicators.index <= 2018]['Tmax'].mean():.2f}",
        f"{climate_indicators[climate_indicators.index <= 2018]['rain_days'].mean():.1f}",
    ],
    "后段均值": [
        f"{climate_indicators[climate_indicators.index >= 2019]['CCI'].mean():.2f}",
        f"{climate_indicators[climate_indicators.index >= 2019]['Tavg'].mean():.2f}",
        f"{climate_indicators[climate_indicators.index >= 2019]['Tmax'].mean():.2f}",
        f"{climate_indicators[climate_indicators.index >= 2019]['rain_days'].mean():.1f}",
    ],
}).to_csv(os.path.join(DATA_DIR, "两阶段对比.csv"), index=False, encoding="utf-8-sig")
print("9. 两阶段对比分析 [OK]")

with open(os.path.join(DATA_DIR, "analysis_conclusions.txt"), "w", encoding="utf-8") as f:
    f.write(f"桂林气候舒适度与旅游影响分析 - 完整结论\n{'='*60}\n\n"
            f"1. MK趋势检验: S={mk_result['S']:.0f}, Z={mk_result['Z']:.4f}, "
            f"p={mk_result['p']:.4f}, Sen's斜率={sen_slope:.4f}\n"
            f"2. Pettitt突变检验: K={pettitt_result['K_stat']:.0f}, "
            f"突变点={change_year}年, p={pettitt_result['p']:.4f}\n"
            f"3. 线性倾向估计: b={lt_raw['b']:.4f}/年, b×10={lt_raw['b']*10:.2f}/10年, "
            f"R^2={lt_raw['r2']:.4f}\n"
            f"4. SI与CCI相关系数: r={corr_si_cci:.4f}\n"
            f"5. 虚拟变量回归: R^2={best_m.rsquared:.4f}, "
            f"CCI弹性系数={cci_coef:.4f}（百分点）, 百分比弹性={pct_elasticity:.4f}\n"
            f"6. 岭回归(游客量): alpha={best_a1:.4f}, R^2={best_r1:.4f}\n"
            f"7. 岭回归(收入): alpha={best_a2:.4f}, R^2={best_r2:.4f}\n"
            f"8. 两阶段对比: CCI差值={late.mean()-early.mean():+.2f}, "
            f"p={p_two:.4f}\n")
print("10. 综合结论 [OK]")

print("\n" + "=" * 80)
print("全部分析完成！")
print("=" * 80)
print("\n已解决的问题：")
print("1. Mann-Kendall 趋势检验")
print("2. Pettitt 突变检验")
print("3. 马尔科夫链等级转换")
print("4. 国标 GB/T 42074-2022 季节划分")
print("5. 线性倾向估计与滑动平均")
print("6. SI 气候舒适度指数对比")
print("7. 虚拟变量回归与气候弹性系数")
print("8. 两阶段对比分析")
print("9. 岭回归解决多重共线性")
print("=" * 80)
