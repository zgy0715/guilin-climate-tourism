# 桂林气候舒适度与旅游影响分析

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-1.5+-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?style=flat&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-1.10+-8CAAE6?style=flat&logo=scipy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![ECharts](https://img.shields.io/badge/pyecharts-1.9+-AA344D?style=flat&logo=apacheecharts&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)

</div>

## 项目概述

基于桂林2011-2025年每日天气数据和2013-2025年月度旅游数据，分析气候舒适度时空演变规律及其对旅游业的影响。

**项目状态：** 所有代码已测试通过，数据已核验，论文已更新（含新增月度旅游数据）。

**最后更新：** 2026年5月30日

---

## 一、问题与解决方案

### 问题一：气候舒适度指数构建与时序分析

| 问题 | 解决方法 | 状态 |
|------|---------|------|
| 缺少Mann-Kendall趋势检验 | MK检验+Sen斜率估计 | 已完成 |
| 缺少国标季节划分对照 | 严格按GB/T 42074-2022计算 | 已完成 |
| 缺少等级转换分析 | 马尔科夫转移概率矩阵 | 已完成 |

### 问题二：气候舒适度等级划分与变化趋势

| 问题 | 解决方法 | 状态 |
|------|---------|------|
| 缺少突变检验 | Pettitt检验 | 已完成 |
| 缺少等级转换概率矩阵 | 5级转移矩阵 | 已完成 |
| 缺少国标季节划分对比 | 国标法四季天数统计 | 已完成 |

### 问题三：气候对旅游产业影响分析

| 问题 | 解决方法 | 状态 |
|------|---------|------|
| VIF>2000多重共线性 | 岭回归（L2惩罚） | 已完成 |
| 气候边际效应量化 | 虚拟变量回归+弹性系数（曹伟宏方法） | 已完成 |
| 前后段气候变化检测 | 两阶段对比分析（荣培君方法） | 已完成 |

---

## 二、核心分析结果

### 2.1 Mann-Kendall趋势检验

| 指标 | 值 |
|------|-----|
| 检验统计量 S | -7 |
| Z 统计量 | -0.2969 |
| p 值 | 0.7665 |
| 趋势 | 下降 |
| 显著性 | 不显著 |
| Sen's斜率 | -0.0402 |

**结论：** 桂林CCI在2011-2025年间呈下降趋势，但统计上不显著。

### 2.2 Pettitt突变检验

| 指标 | 值 |
|------|-----|
| 突变统计量 K | 14 |
| 突变点位置 | 第5年（2015年） |
| p 值 | ≈1.0（近似公式边界效应，截断值） |

**结论：** CCI未发生显著突变（p≈1.0）。

### 2.3 马尔科夫链等级转换

转移概率矩阵显示，各等级最可能转移到"3-正常"等级（概率约40%）。

### 2.4 国标GB/T 42074-2022季节划分

| 季节 | 平均天数 |
|------|---------|
| 春季 | 74.9天 |
| 夏季 | 179.5天 |
| 秋季 | 70.4天 |
| 冬季 | 39.5天 |

### 2.5 岭回归（解决多重共线性）

| 模型 | 最优alpha | 调整R2 |
|------|----------|--------|
| 游客量模型 | 0.0010 | 0.6888 |
| 旅游收入模型 | 0.0010 | 0.6604 |

**岭回归方程（游客量）：**
游客量 = -34050.76 + (-537.94)×CCI + 3648.33×Tmax + (-79.25)×降雨天数 + (-112.81)×高温天数 + (-33.43)×低温天数 + (-1573.73)×疫情

### 2.6 线性倾向估计与滑动平均（荣培君方法）

| 窗口 | 倾向率b（/年） | 倾向率（/10年） | R^2 | p值 |
|------|-------------|---------------|------|------|
| 原始序列 | -0.0295 | -0.30 | 0.005 | 0.802 |
| 9年滑动 | -0.1384 | -1.38 | 0.897 | 0.001 |

**结论：** 与MK检验结论一致——CCI呈微弱下降趋势，原始序列统计不显著。

### 2.7 SI指数对比验证（荣培君方法）

- SI与CCI年度均值的Pearson相关系数 r = 0.78
- 两种方法对气候舒适度的评价高度一致，CCI模型可靠性得到验证

### 2.8 虚拟变量回归与气候弹性系数（曹伟宏方法）

| 模型 | R^2 | 调整R^2 | CCI弹性系数 | 百分比弹性 |
|------|------|--------|-----------|----------|
| 模型3（CCI+旺季+春节+疫情） | 0.521 | 0.420 | -0.283 | -0.590 |

**结论：** 季节效应（春节、暑假/国庆）是驱动季度旅游分布的主导因素；CCI偏效应不显著（p=0.216），气候弹性-0.59%远优于之前-310%的异常值。参考：丽江海外游客弹性1.31%，国内游客弹性0.56%。

### 2.9 两阶段对比分析（荣培君方法）

| 指标 | 前段(2011-2018) | 后段(2019-2025) | 差值 | p值 |
|------|----------------|----------------|------|------|
| CCI | 53.22 | 51.96 | -1.26 | 0.202 |
| 年均气温 | 20.18°C | 21.11°C | +0.93°C | **0.004** |
| 降雨天数 | 171.5天 | 111.9天 | -59.6天 | **<0.001** |

**结论：** 后段气温显著升高、降雨显著减少，但CCI变化不显著（温度上升与降雨减少的效应相互抵消）。

---

## 三、数据来源

### 3.1 年度旅游数据（2011-2024年官方统计 + 2025年估算）

- **来源：** 桂林市国民经济和社会发展统计公报
- **核验链接：** https://tjj.guilin.gov.cn/tjsj_2/tjgb/（统计局新路径）或 https://www.guilin.gov.cn/glsj/sjfb/tjgb/（市政府门户汇总）
- **备份查阅：** https://www.guilin.gov.cn/glsj/sjfb_2/tjsj/（统计数据分页列表）
- **用途：** 灰色关联分析、长期趋势分析、回归模型
- **说明：** 2011-2024 年为官方数据，2025 年为基于季度累计的估算值（年度尚未结束）

### 3.2 月度旅游数据（2013-2025年）

- **来源：** 桂林市统计局和文化广电和旅游局官方月度统计
- **核验链接：** https://www.guilin.gov.cn/glsj/sjfb_2/tjsj/index_13.shtml（桂林市政府官网月度/季度旅游统计）
- **用途：** 灰色关联分析、月度回归模型、季节性分析、两阶段对比分析
- **说明：** 2013-2019年逐月数据完整；2020-2024年国内游客仅季度数据，入境游客逐月公布；共75个有效月度数据点

### 3.3 季度旅游数据（2020-2025年）

- **来源：** 桂林市文化广电和旅游局官方季度通报
- **核验链接：** http://wglj.guilin.gov.cn/zfxxgk/fdzdgknr/sjfb/
- **用途：** 面板回归分析、季节性分析

### 3.4 节假日旅游数据（2019-2025年）

- **来源：** 桂林市文化广电和旅游局官方假日旅游通报
- **核验链接：** http://wglj.guilin.gov.cn/zwdt/mtjj/
- **用途：** 节假日案例分析

### 3.5 气象数据

- **来源：** 附件2：2011-2025桂林天气记录（每日）.xlsx
- **用途：** CCI计算、气候特征分析

### 3.6 气候季节划分标准

- **来源：** GB/T 42074-2022《气候季节划分》
- **附件：** 附件1：气候季节划分国家标准GBT+42074-2022.pdf

---

## 四、文件结构

```
数模/
├── B题/                          # 题目和附件
│   ├── B题.docx                  # 题目文档
│   ├── 附件1：气候季节划分国家标准GBT+42074-2022.pdf
│   └── 附件2：2011-2025桂林天气记录（每日）.xlsx
├── 代码/
│   ├── cci_core.py               # 公共模块（路径/CCI/评分函数，被各脚本引用）
│   ├── complete_analysis.py      # 完整问题解决方案（推荐）
│   ├── problem3_analysis.py      # 问题三基础分析
│   ├── final_analysis.py         # 综合分析
│   ├── charts_optimized.py       # ECharts图表生成
│   └── optimized_CCI.py          # 优化版CCI模型
├── 数据/
│   ├── yearly_data_with_climate.xlsx    # 年度综合数据（2011-2025）
│   ├── quarterly_data_with_climate.xlsx # 季度综合数据（2020-2025）
│   ├── holiday_data_with_climate.xlsx   # 节假日数据（2019-2025）
│   ├── 数据来源说明.txt                  # 数据来源和核验说明
│   ├── mann_kendall_test.xlsx           # MK检验结果
│   ├── pettitt_test.xlsx                # Pettitt检验结果
│   ├── markov_transition.xlsx           # 马尔科夫转移矩阵
│   ├── season_division_gb.xlsx          # 国标季节划分
│   ├── ridge_regression_results.xlsx    # 岭回归结果
│   ├── linear_tendency.xlsx             # 线性倾向估计
│   ├── si_index_comparison.xlsx         # SI指数对比
│   ├── dummy_regression_elasticity.xlsx # 虚拟变量回归弹性系数
│   ├── two_stage_comparison.xlsx        # 两阶段对比分析
├── 图片/
│   ├── 图1_CCI计算流程图.png
│   ├── 图2_四季划分与无春解释图.png
│   ├── 图3_气候对旅游产业影响机制图.png
│   ├── 1_CCI热力图.html          (论文图9)
│   ├── 2_月度CCI变化.html        (论文图4)
│   ├── 5_灰色关联分析.html       (论文图6)
│   ├── 6_CCI与游客量关系.html    (论文图7)
│   ├── 7_年度CCI趋势.html        (论文图2)
│   ├── 8_等级分布饼图.html       (论文图3)
│   ├── 9_三种模型对比.html
│   ├── 10_假期因素影响.html
│   ├── 11_线性倾向估计.html      (论文图10)
│   ├── 12_SI与CCI对比.html       (论文图11)
│   ├── 13_虚拟变量回归诊断.html  (论文图12)
│   └── 14_两阶段对比.html        (论文图13)
├── 论文/
│   ├── 写作模板.docx
│   ├── 承诺书.docx
│   ├── 桂林气候舒适度与旅游影响分析论文.md    # 论文正文
│   ├── 桂林气候舒适度与旅游影响分析论文.docx  # 论文Word版
│   ├── md2docx.py                              # Markdown转Word脚本
│   └── csv2xlsx.py                             # CSV转Excel脚本
├── 规则/                         # 比赛规则
├── 文献综述与解决方案.md
└── README.md                     # 项目说明
```

---

## 五、使用说明

### 5.0 环境准备

```bash
# 安装所有依赖
pip install -r requirements.txt
```

### 5.1 一键运行所有代码

```bash
# 1. 运行完整分析（生成所有数据文件）
python 代码/complete_analysis.py

# 2. 运行问题三分析
python 代码/problem3_analysis.py

# 3. 生成ECharts图表
python 代码/charts_optimized.py

# 4. CSV转Excel（可选）
python 论文/csv2xlsx.py

# 5. Markdown转Word（可选）
python 论文/md2docx.py
```

**依赖包：** pandas, numpy, scipy, statsmodels, scikit-learn, openpyxl, pyecharts（图表生成需要）

### 5.1 运行完整分析（推荐）

```bash
python 代码/complete_analysis.py
```

包含：MK趋势检验、Pettitt突变检验、马尔科夫链、岭回归（输出CSV格式，需运行csv2xlsx.py转换为Excel）

### 5.3 运行问题三分析

```bash
python 代码/problem3_analysis.py
```

包含：灰色关联分析、多元回归、面板回归、节假日案例分析

### 5.4 生成ECharts图表

```bash
python 代码/charts_optimized.py
```

生成10张交互式图表：
1. CCI热力图
2. 月度CCI变化曲线
3. 灰色关联分析
4. CCI与游客量关系
5. 年度CCI趋势
6. 等级分布饼图
7. 线性倾向估计与滑动平均（荣培君方法）
8. SI与CCI年度对比验证
9. 虚拟变量回归诊断（曹伟宏方法）
10. 两阶段气候指标对比（荣培君方法）

### 5.5 CSV转Excel

```bash
python 论文/csv2xlsx.py
```

将数据目录下所有.csv文件转换为.xlsx格式。

### 5.6 Markdown转Word

```bash
python 论文/md2docx.py
```

将论文md文件转换为Word文档，格式规范：
- 论文题目：三号黑体居中
- 摘要标题：四号黑体居中
- 正文：小四宋体，单倍行距
- 英文：小四Times New Roman

### 5.7 查看图表

在浏览器中打开 `图片/*.html` 文件即可查看交互式图表。

---

## 六、参考文献

### 6.1 趋势检验与突变分析

[1] Mann H B. Nonparametric tests against trend[J]. Econometrica, 1945, 13(3): 245-259.
[2] Kendall M G. Rank Correlation Methods[M]. 4th ed. London: Griffin, 1975.
[3] Yue S, Pilon P, Phinney B, et al. The influence of autocorrelation on the ability to detect trend in hydrological data[J]. Hydrological Processes, 2002, 16(9): 1807-1829.
[4] Hamed K H, Rao A R. A modified Mann-Kendall trend test for autocorrelated data[J]. Journal of Hydrology, 1998, 204(1-4): 182-196.
[5] Pettitt A N. A non-parametric approach to the change-point problem[J]. Applied Statistics, 1979, 28(2): 126-135.

### 6.2 气候季节划分

[6] 中国国家标准化管理委员会. GB/T 42074-2022 气候季节划分[S]. 北京: 中国标准出版社, 2022.

### 6.3 回归分析与变量选择

[7] Marquardt D W. Generalized inverses, ridge regression, biased linear estimation, and nonlinear estimation[J]. Technometrics, 1970, 12(3): 591-612.
[8] Hoerl A E, Kennard R W. Ridge regression: biased estimation for nonorthogonal problems[J]. Technometrics, 1970, 12(1): 55-67.
[9] Tibshirani R. Regression shrinkage and selection via the lasso[J]. Journal of the Royal Statistical Society: Series B, 1996, 58(1): 267-288.

### 6.4 灰色系统与数学方法

[10] 邓聚龙. 灰色系统理论教程[M]. 武汉: 华中理工大学出版社, 1990.
[11] 刘思峰, 党耀国, 方志耕, 等. 灰色系统理论及其应用[M]. 8版. 北京: 科学出版社, 2017.
[12] 徐建华. 现代地理学中的数学方法[M]. 2版. 北京: 高等教育出版社, 2002.

### 6.5 旅游气候舒适度研究

[13] 曹伟宏, 何元庆, 李宗省, 等. 丽江旅游气候舒适度与年内客流量变化相关性分析[J]. 地理科学, 2012, 32(12): 1459-1464.
[14] 谭凯炎, 闵庆文, 王培娟. 一种基于中国气候特征和人体舒适感受的气候舒适指数模型[J]. 气象, 2022, 48(7): 913-924.
[15] 荣培君, 张荣荣, 赵现红. 中国气候舒适度时空演变及其对旅游业的影响[J]. 世界地理研究, 2024, 33(7): 139-149.

---

## 七、更新日志

### 2026-05-30 代码重构（路径/模块/质量）
- [x] 修复所有硬编码路径：`D:/download/数模/` → `cci_core.PROJECT_ROOT`（基于 `__file__` 自动计算）
- [x] 提取公共模块 `代码/cci_core.py`：统一 CCI 计算、天气/风力评分、数据读取函数
- [x] 统一列索引：`charts_optimized.py` 中 `df[3]`/`df[4]`/`df[6]` → `df["day_weather"]`/`df["night_weather"]`/`df["wind_power"]`
- [x] 创建 `requirements.txt`：项目依赖统一管理
- [x] 移除 `final_analysis.py` 中的模拟数据退化分支（`np.random.seed(42)`）
- [x] 修复数据来源描述：明确区分 2011-2024 年官方数据与 2025 年估算数据
- [x] 论文转换脚本 `csv2xlsx.py`/`md2docx.py` 路径也改为相对路径
- [x] 运行测试验证：所有代码完成重构并通过运行
- [x] 论文格式规范化：三号黑体标题、四号黑体摘要、小四宋体正文、单倍行距
- [x] 论文内容扩充：问题背景、问题分析、数据预处理、模型诊断等章节大幅扩展
- [x] 参考文献标注：15篇文献全部添加正文引用标注
- [x] 数据文件统一为xlsx格式，删除多余CSV文件
- [x] 删除多余docx文件（_新版、~$临时文件）
- [x] 新增md2docx.py：Markdown转Word脚本
- [x] 新增csv2xlsx.py：CSV转Excel脚本
- [x] 图片占位标记嵌入论文
- [x] 修复md2docx.py图片路径解析bug（改为项目根目录）
- [x] 修复文献综述引用编号与论文不一致问题
- [x] 所有代码测试通过（complete_analysis.py、problem3_analysis.py、charts_optimized.py）
- [x] 旅游数据改为从Excel文件读取，不再硬编码在代码中

---

## 八、注意事项

1. **数据真实性**：所有旅游数据均来自官方统计公报，可在指定链接验证
2. **模型局限性**：年度回归样本量仅15年，季度样本量仅24条，部分系数不显著
3. **多重共线性**：CCI与最高气温存在高度共线性（VIF>2000），已使用岭回归处理
4. **气候弹性系数**：虚拟变量回归测算的弹性系数（-0.59%）统计不显著（p=0.216），待更长时序数据验证
5. **2025年数据**：气象数据截至2025年12月16日（不完整），旅游数据为官方公布值
6. **代码输出**：complete_analysis.py输出CSV格式，需运行csv2xlsx.py转换为Excel
