# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
from pyecharts import options as opts
from pyecharts.charts import HeatMap, Line, Bar, Scatter, Pie
from pyecharts.globals import ThemeType
from scipy import stats
from statsmodels.stats.stattools import durbin_watson
import statsmodels.api as sm

# ─── 导入公共模块 ─────────────────────────────────────────────
from cci_core import (
    PROJECT_ROOT, DATA_DIR, IMG_DIR,
    read_weather_data, add_comfort_scores, calc_cci,
    classify_percentile, weather_score, wind_score_simple,
    wind_score_owci, calc_si, wind_to_speed,
)

os.makedirs(IMG_DIR, exist_ok=True)

# ═════════════════════════════════════════════════════════════
# 数据准备
# ═════════════════════════════════════════════════════════════

df = read_weather_data()
df = add_comfort_scores(df)
df, weights = calc_cci(df)

# 百分位数法等级划分
p10, p30, p70, p90 = df["CCI"].quantile([0.1, 0.3, 0.7, 0.9])
conditions = [
    df["CCI"] <= p10,
    (df["CCI"] > p10) & (df["CCI"] <= p30),
    (df["CCI"] > p30) & (df["CCI"] <= p70),
    (df["CCI"] > p70) & (df["CCI"] <= p90),
    df["CCI"] > p90,
]
choices = ["最不舒适", "较不舒适", "正常", "较舒适", "最舒适"]
df["CCI_level"] = np.select(conditions, choices, default="正常")

# 读取真实旅游数据
yearly_tourism = pd.read_excel(
    os.path.join(DATA_DIR, "年度旅游气候数据.xlsx"),
    usecols=["Year", "tourists_total", "tourism_revenue"],
)

# 年度 CCI
yearly_cci = df.groupby("Year")["CCI"].mean().reset_index()
yearly_cci.columns = ["Year", "CCI_mean"]
yearly_data = pd.merge(yearly_tourism, yearly_cci, on="Year")

# 季度数据
quarterly_data = pd.read_excel(os.path.join(DATA_DIR, "季度旅游气候数据.xlsx"))


# ═════════════════════════════════════════════════════════════
# 图1: CCI 热力图
# ═════════════════════════════════════════════════════════════
def chart_cci_heatmap():
    heat = df.groupby(["Year", "Month"])["CCI"].mean().unstack()
    years = [str(y) for y in heat.index]
    months = [f"{m}月" for m in range(1, 13)]
    data = [[j, i, round(heat.values[i, j], 1)] for i in range(len(years)) for j in range(12)]

    c = (
        HeatMap(init_opts=opts.InitOpts(theme=ThemeType.MACARONS, width="900px", height="600px"))
        .add_xaxis(months)
        .add_yaxis("CCI", years, data, itemstyle_opts=opts.ItemStyleOpts(opacity=0.8))
        .set_global_opts(
            visualmap_opts=opts.VisualMapOpts(
                min_=40, max_=90,
                range_color=["#d94e5d", "#eac736", "#50a3ba", "#1e90ff"],
            ),
            xaxis_opts=opts.AxisOpts(name="月份"),
            yaxis_opts=opts.AxisOpts(name="年份", is_inverse=True),
        )
    )
    c.render(os.path.join(IMG_DIR, "1_CCI热力图.html"))
    print("[OK] 1_CCI热力图.html")


# ═════════════════════════════════════════════════════════════
# 图2: 月度 CCI 变化曲线
# ═════════════════════════════════════════════════════════════
def chart_monthly_cci():
    monthly = df.groupby("Month")["CCI"].mean()
    c = (
        Line(init_opts=opts.InitOpts(theme=ThemeType.MACARONS, width="900px", height="450px"))
        .add_xaxis([f"{m}月" for m in range(1, 13)])
        .add_yaxis("月均CCI", monthly.values.round(1).tolist(),
                    linestyle_opts=opts.LineStyleOpts(is_show=True, width=3, color="#5470c6"),
                    itemstyle_opts=opts.ItemStyleOpts(color="#5470c6"),
                    is_symbol_show=True, symbol_size=8,
                    label_opts=opts.LabelOpts(is_show=True, position="top",
                                              formatter="{@[1]}", font_size=9))
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(name="月份"),
            yaxis_opts=opts.AxisOpts(name="CCI", min_=20, max_=90),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
        )
    )
    c.render(os.path.join(IMG_DIR, "2_月度CCI变化.html"))
    print("[OK] 2_月度CCI变化.html")


# ═════════════════════════════════════════════════════════════
# 图5: 灰色关联分析
# ═════════════════════════════════════════════════════════════
def chart_gra():
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()

    yearly_climate = df.groupby("Year").agg(
        CCI_mean=("CCI", "mean"),
        Tmax_mean=("Tmax", "mean"),
        Tavg_mean=("Tavg", "mean"),
        rain_days=("day_weather", lambda x: sum("雨" in str(i) for i in x)),
        hot_days=("Tmax", lambda x: sum(x >= 35)),
        cold_days=("Tmin", lambda x: sum(x <= 5)),
    ).reset_index()

    yearly_merged = pd.merge(yearly_tourism, yearly_climate, on="Year")
    factors_cols = ["CCI_mean", "Tmax_mean", "Tavg_mean", "rain_days", "hot_days", "cold_days"]
    factors_names = ["CCI指数", "最高气温", "平均气温", "降雨天数", "高温天数", "低温天数"]

    gra_results = []
    ref_norm = scaler.fit_transform(yearly_merged["tourists_total"].values.reshape(-1, 1)).flatten()
    for col, name in zip(factors_cols, factors_names):
        fac_norm = scaler.fit_transform(yearly_merged[[col]].values)
        diff = np.abs(fac_norm - ref_norm.reshape(-1, 1))
        d_min, d_max = diff.min(), diff.max()
        coef = (d_min + 0.5 * d_max) / (diff + 0.5 * d_max)
        gra_results.append({"因素": name, "关联系数": coef.mean()})

    gra_df = pd.DataFrame(gra_results).sort_values("关联系数")

    c = (
        Bar(init_opts=opts.InitOpts(theme=ThemeType.MACARONS, width="750px", height="450px"))
        .add_xaxis(gra_df["因素"].tolist())
        .add_yaxis("关联系数", gra_df["关联系数"].round(3).tolist(),
                    itemstyle_opts=opts.ItemStyleOpts(color="#5470c6"),
                    label_opts=opts.LabelOpts(position="right", formatter="{c}"))
        .reversal_axis()
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(name="关联系数", max_=1),
            yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(position="left")),
        )
    )
    c.render(os.path.join(IMG_DIR, "5_灰色关联分析.html"))
    print("[OK] 5_灰色关联分析.html")


# ═════════════════════════════════════════════════════════════
# 图6: CCI 与游客量关系（双轴趋势对比）
# ═════════════════════════════════════════════════════════════
def chart_cci_scatter():
    years = [str(y) for y in yearly_data["Year"]]
    tourists = yearly_data["tourists_total"].tolist()
    cci_vals = yearly_data["CCI_mean"].round(1).tolist()

    bar = (
        Bar(init_opts=opts.InitOpts(theme=ThemeType.MACARONS, width="950px", height="550px"))
        .add_xaxis(years)
        .add_yaxis("年度游客量（万人次）", tourists,
                    itemstyle_opts=opts.ItemStyleOpts(color="#5470c6", opacity=0.7),
                    label_opts=opts.LabelOpts(is_show=True, position="top",
                                              font_size=9, rotate=0))
        .extend_axis(yaxis=opts.AxisOpts(
            name="CCI", position="right",
            min_=45, max_=58,
            axislabel_opts=opts.LabelOpts(formatter="{value}"),
        ))
    )

    line = (
        Line()
        .add_xaxis(years)
        .add_yaxis("年均CCI", cci_vals, yaxis_index=1,
                    linestyle_opts=opts.LineStyleOpts(is_show=True, width=3, color="#E94F37"),
                    itemstyle_opts=opts.ItemStyleOpts(color="#E94F37"),
                    is_symbol_show=True, symbol_size=12, symbol="circle",
                    label_opts=opts.LabelOpts(is_show=True, position="bottom",
                                              formatter="{c}", font_size=10))
    )
    bar.overlap(line)
    bar.set_global_opts(
        xaxis_opts=opts.AxisOpts(name="年份"),
        yaxis_opts=opts.AxisOpts(name="年度游客量（万人次）"),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        legend_opts=opts.LegendOpts(pos_top="3%"),
    )
    bar.render(os.path.join(IMG_DIR, "6_CCI与游客量关系.html"))
    print("[OK] 6_CCI与游客量关系.html")


# ═════════════════════════════════════════════════════════════
# 图7: 年度 CCI 趋势
# ═════════════════════════════════════════════════════════════
def chart_yearly_cci():
    yearly = df.groupby("Year")["CCI"].mean()
    xn = np.arange(len(yearly))
    yv = yearly.values
    slope, intercept, rv, pv, se = stats.linregress(xn, yv)
    trend_y = slope * xn + intercept

    c = (
        Line(init_opts=opts.InitOpts(theme=ThemeType.MACARONS, width="950px", height="500px"))
        .add_xaxis([str(y) for y in yearly.index])
        .add_yaxis("年均CCI", yearly.values.round(1).tolist(),
                    linestyle_opts=opts.LineStyleOpts(is_show=True, width=3, color="#E94F37"),
                    itemstyle_opts=opts.ItemStyleOpts(color="#E94F37"),
                    is_symbol_show=True, symbol_size=10,
                    label_opts=opts.LabelOpts(is_show=True, position="top",
                                              formatter="{@[1]}", font_size=10),
                    markpoint_opts=opts.MarkPointOpts(data=[
                        opts.MarkPointItem(type_="max", name="最高"),
                        opts.MarkPointItem(type_="min", name="最低"),
                    ]))
        .add_yaxis("趋势线", trend_y.round(2).tolist(),
                    linestyle_opts=opts.LineStyleOpts(is_show=True, width=2, color="#999999", type_="dashed"),
                    itemstyle_opts=opts.ItemStyleOpts(opacity=0), is_symbol_show=False)
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(name="年份"),
            yaxis_opts=opts.AxisOpts(name="CCI", min_=48, max_=58),
        )
    )
    c.render(os.path.join(IMG_DIR, "7_年度CCI趋势.html"))
    print("[OK] 7_年度CCI趋势.html")


# ═════════════════════════════════════════════════════════════
# 图8: 等级分布饼图
# ═════════════════════════════════════════════════════════════
def chart_level_pie():
    level_counts = df["CCI_level"].value_counts()
    data = [[level, int(count)] for level, count in zip(level_counts.index, level_counts.values)]

    c = (
        Pie(init_opts=opts.InitOpts(theme=ThemeType.MACARONS, width="750px", height="550px"))
        .add("", data, radius=["35%", "70%"], rosetype="radius",
             label_opts=opts.LabelOpts(formatter="{b}: {c}天\n({d}%)", font_size=12))
        .set_global_opts(
            legend_opts=opts.LegendOpts(pos_left="left", orient="vertical"),
        )
        .set_colors(["#d94e5d", "#eac736", "#50a3ba", "#1e90ff", "#228B22"])
    )
    c.render(os.path.join(IMG_DIR, "8_等级分布饼图.html"))
    print("[OK] 8_等级分布饼图.html")


# ═════════════════════════════════════════════════════════════
# 图11: 线性倾向估计与滑动平均
# ═════════════════════════════════════════════════════════════
def chart_linear_tendency():
    yearly = df.groupby("Year")["CCI"].mean()
    years = yearly.index.tolist()
    values = yearly.values
    n = len(values)

    # 9年滑动平均
    ma9 = pd.Series(values).rolling(window=9, center=True).mean()
    ma9 = ma9.ffill().bfill().values

    # 原始序列和滑动平均的线性倾向（统一用 0..n-1 作为时间轴）
    t_raw = np.arange(n)
    slope_raw, intercept_raw, r_raw, p_raw, _ = stats.linregress(t_raw, values)
    trend_raw = slope_raw * t_raw + intercept_raw
    slope_ma, intercept_ma, r_ma, p_ma, _ = stats.linregress(t_raw, ma9)
    trend_ma = slope_ma * t_raw + intercept_ma

    c = (
        Line(init_opts=opts.InitOpts(theme=ThemeType.MACARONS, width="1000px", height="550px"))
        .add_xaxis([str(y) for y in years])
        .add_yaxis("年均CCI（原始）", values.round(1).tolist(),
                    linestyle_opts=opts.LineStyleOpts(is_show=True, width=2, color="#aaa"),
                    itemstyle_opts=opts.ItemStyleOpts(color="#aaa"),
                    is_symbol_show=True, symbol_size=8, symbol="circle")
        .add_yaxis("9年滑动平均", ma9.round(1).tolist(),
                    linestyle_opts=opts.LineStyleOpts(is_show=True, width=3, color="#E94F37"),
                    itemstyle_opts=opts.ItemStyleOpts(color="#E94F37"),
                    is_symbol_show=True, symbol_size=10, symbol="diamond",
                    label_opts=opts.LabelOpts(is_show=True, position="bottom",
                                              formatter="{@[1]}", font_size=9))
        .add_yaxis("滑动平均趋势线", trend_ma.round(1).tolist(),
                    linestyle_opts=opts.LineStyleOpts(is_show=True, width=2, color="#999", type_="dashed"),
                    itemstyle_opts=opts.ItemStyleOpts(opacity=0), is_symbol_show=False)
        .add_yaxis("原始序列趋势线", trend_raw.round(1).tolist(),
                    linestyle_opts=opts.LineStyleOpts(is_show=True, width=2, color="#ccc", type_="dotted"),
                    itemstyle_opts=opts.ItemStyleOpts(opacity=0), is_symbol_show=False)
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(name="年份"),
            yaxis_opts=opts.AxisOpts(name="CCI", min_=48, max_=58),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
        )
    )
    c.render(os.path.join(IMG_DIR, "11_线性倾向估计.html"))
    print("[OK] 11_线性倾向估计.html")


# ═════════════════════════════════════════════════════════════
# 图12: SI指数与CCI年度对比
# ═════════════════════════════════════════════════════════════
def chart_si_vs_cci():
    df_si = calc_si(df.copy())
    yearly = df_si.groupby("Year").agg(
        CCI=("CCI", "mean"), SI_score=("SI_score", "mean"),
    ).round(1)
    years = [str(y) for y in yearly.index]

    bar = (
        Bar(init_opts=opts.InitOpts(theme=ThemeType.MACARONS, width="950px", height="500px"))
        .add_xaxis(years)
        .add_yaxis("CCI", yearly["CCI"].tolist(),
                    itemstyle_opts=opts.ItemStyleOpts(color="#5470c6", opacity=0.8),
                    label_opts=opts.LabelOpts(is_show=True, position="inside",
                                              formatter="{c}", font_size=10))
        .extend_axis(yaxis=opts.AxisOpts(name="SI_score", position="right", min_=60, max_=80))
    )

    line = (
        Line()
        .add_xaxis(years)
        .add_yaxis("SI_score", yearly["SI_score"].tolist(), yaxis_index=1,
                    linestyle_opts=opts.LineStyleOpts(is_show=True, width=3, color="#ee6666"),
                    itemstyle_opts=opts.ItemStyleOpts(color="#ee6666"),
                    is_symbol_show=True, symbol_size=12,
                    label_opts=opts.LabelOpts(is_show=True, position="top",
                                              formatter="{@[1]}", font_size=10))
    )
    bar.overlap(line)
    bar.set_global_opts(
        xaxis_opts=opts.AxisOpts(name="年份"),
        yaxis_opts=opts.AxisOpts(name="CCI", min_=45, max_=58),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        legend_opts=opts.LegendOpts(pos_top="5%"),
    )
    bar.render(os.path.join(IMG_DIR, "12_SI与CCI对比.html"))
    print("[OK] 12_SI与CCI对比.html")


# ═════════════════════════════════════════════════════════════
# 图13: 虚拟变量回归诊断 —— 实际 vs 预测
# ═════════════════════════════════════════════════════════════
def chart_dummy_regression():
    # 读取 季度回归数据.csv
    qdata = pd.read_csv(os.path.join(DATA_DIR, "季度回归数据.csv"))
    qdata["label"] = qdata.apply(
        lambda r: f"{int(r['Year'])}Q{int(r['Quarter'])}", axis=1)

    # 拟合模型3
    X = sm.add_constant(qdata[["CCI_mean", "Q_dummy", "spring_dummy", "pandemic"]])
    y = qdata["Q_index"]
    model = sm.OLS(y, X).fit()
    qdata["predicted"] = model.predict(X)

    labels = qdata["label"].tolist()
    actual = qdata["Q_index"].tolist()
    predicted = qdata["predicted"].tolist()

    c = (
        Line(init_opts=opts.InitOpts(theme=ThemeType.MACARONS, width="1000px", height="550px"))
        .add_xaxis(labels)
        .add_yaxis("实际Q_index(%)", actual,
                    linestyle_opts=opts.LineStyleOpts(is_show=True, width=2.5, color="#5470c6"),
                    itemstyle_opts=opts.ItemStyleOpts(color="#5470c6"),
                    is_symbol_show=True, symbol_size=8, symbol="circle",
                    label_opts=opts.LabelOpts(is_show=False))
        .add_yaxis("预测Q_index(%)", predicted,
                    linestyle_opts=opts.LineStyleOpts(is_show=True, width=2.5, color="#ee6666"),
                    itemstyle_opts=opts.ItemStyleOpts(color="#ee6666"),
                    is_symbol_show=True, symbol_size=8, symbol="diamond",
                    label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(name="季度", axislabel_opts=opts.LabelOpts(rotate=45, font_size=9)),
            yaxis_opts=opts.AxisOpts(name="季度游客量指数 (%)"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            legend_opts=opts.LegendOpts(pos_top="5%"),
        )
    )
    c.render(os.path.join(IMG_DIR, "13_虚拟变量回归诊断.html"))
    print("[OK] 13_虚拟变量回归诊断.html")


# ═════════════════════════════════════════════════════════════
# 图14: 两阶段气候指标对比
# ═════════════════════════════════════════════════════════════
def chart_two_stage():
    # 读取两阶段数据
    ts = pd.read_csv(os.path.join(DATA_DIR, "两阶段对比.csv"))
    indicators = ["CCI", "Tavg", "Tmax", "rain_days"]
    names = ["CCI", "年均气温(°C)", "年最高气温(°C)", "年降雨天数"]
    early_vals = [float(v) for v in ts["前段均值"]]
    late_vals = [float(v) for v in ts["后段均值"]]

    c = (
        Bar(init_opts=opts.InitOpts(theme=ThemeType.MACARONS, width="900px", height="500px"))
        .add_xaxis(names)
        .add_yaxis("前段 (2011-2018)", early_vals,
                    itemstyle_opts=opts.ItemStyleOpts(color="#5470c6", opacity=0.8),
                    label_opts=opts.LabelOpts(is_show=True, position="inside",
                                              formatter="{c}", font_size=11))
        .add_yaxis("后段 (2019-2025)", late_vals,
                    itemstyle_opts=opts.ItemStyleOpts(color="#ee6666", opacity=0.8),
                    label_opts=opts.LabelOpts(is_show=True, position="inside",
                                              formatter="{c}", font_size=11))
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(name="指标"),
            yaxis_opts=opts.AxisOpts(name="均值"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            legend_opts=opts.LegendOpts(pos_top="5%"),
        )
    )
    c.render(os.path.join(IMG_DIR, "14_两阶段对比.html"))
    print("[OK] 14_两阶段对比.html")


# ═════════════════════════════════════════════════════════════
# 主函数
# ═════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("开始生成优化版 ECharts 图表...\n")
    chart_cci_heatmap()
    chart_monthly_cci()
    chart_gra()
    chart_cci_scatter()
    chart_yearly_cci()
    chart_level_pie()
    chart_linear_tendency()
    chart_si_vs_cci()
    chart_dummy_regression()
    chart_two_stage()
    print(f"\n全部完成！图表保存在: {IMG_DIR}")
