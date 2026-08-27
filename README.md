# 脊柱侧弯治疗效果数据分析 Demo

这是一个脊柱侧弯治疗效果 Python full-pipeline demo，覆盖数据清洗、探索性数据分析、统计推断、PSM 倾向评分匹配、敏感性分析、二分类疗效分析和治疗效果预测。

仓库默认使用 `data/sample_scoliosis_data.xlsx` 合成示例数据，便于公开展示和复现。本地若存在私有原始文件 `脊柱侧弯数据统计表.xlsx`，pipeline 会优先读取该文件；该原始文件已被 `.gitignore` 排除，不建议上传到公开仓库。

## 快速运行

```bash
/Users/wangzeyu/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m src.pipeline
```

运行后会生成：

- `outputs/cleaned_data.csv`
- `outputs/dashboard.html`
- `outputs/report.html`
- `outputs/tables/executive_summary.csv`
- `outputs/tables/data_quality_summary.csv`
- `outputs/tables/data_dictionary.csv`
- `outputs/tables/eda_summary.csv`
- `outputs/tables/inference_results.csv`
- `outputs/tables/binary_outcome_results.csv`
- `outputs/tables/psm_balance.csv`
- `outputs/tables/psm_effect.csv`
- `outputs/tables/psm_sensitivity.csv`
- `outputs/tables/subgroup_effects.csv`
- `outputs/tables/model_metrics.csv`
- `outputs/tables/model_coefficients.csv`
- `outputs/tables/model_prediction_by_group.csv`
- `outputs/tables/model_feature_ablation.csv`
- `outputs/figures/*.png`

## 扩展分析

- 静态可视化界面：`outputs/dashboard.html` 提供中文筛选、核心结论摘要、方法说明、SVG 图表、去标识化队列表格搜索/分页/导出和打印样式。
- HTML 报告：`outputs/report.html` 汇总核心结论、质量检查、PSM 和模型结果。
- 二分类疗效：`improved_5deg = Cobb改善量 >= 5`，输出风险差、OR 和协变量调整线性概率模型。
- PSM 敏感性：比较无 caliper 与多个 logit propensity caliper 下的匹配对数、ATT 和匹配后 SMD。
- 分层描述：按年龄段、基线 Cobb 角分层、干预周期和性别输出描述性组间差异。
- 模型解释：输出 Ridge 交叉验证指标、按治疗组预测误差和 feature ablation 表。

## Notebook

`notebooks/01_demo_pipeline.ipynb` 是展示版分析入口。核心逻辑放在 `src/`，notebook 只负责叙事和查看输出，方便复现。

## 重要说明

- 原始 Excel 文件只读，不会被修改。
- GitHub 版本默认只提交合成示例数据；真实患者级数据和生成的 `outputs/` 不进入版本控制。
- 当前样本量较小，本 demo 强调分析流程与可解释性，不把模型输出解释成临床结论。
- `每周训练时间（小时）` 只在“支具+训练组”记录；建模中另建 `training_exposure_hours` 表示训练暴露强度，纯支具组置 0 仅用于工程特征，不代表真实记录了 0 小时训练。
- 若本地缺少 `scipy/sklearn/statsmodels/matplotlib/seaborn`，项目仍可运行：统计检验、PSM、Ridge 回归和 PNG 图表都有轻量 fallback 实现。

## 测试

```bash
/Users/wangzeyu/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 tests/test_pipeline.py
```
