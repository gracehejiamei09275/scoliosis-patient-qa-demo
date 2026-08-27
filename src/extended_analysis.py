from __future__ import annotations

import numpy as np
import pandas as pd


def data_quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    checks = [
        {
            "check": "row_count",
            "value": len(df),
            "status": "pass" if len(df) == 102 else "review",
            "note": "Expected 102 rows from two source sheets.",
        },
        {
            "check": "outcome_complete_rows",
            "value": int(df["outcome_observed"].sum()),
            "status": "pass" if int(df["outcome_observed"].sum()) == 101 else "review",
            "note": "Rows with parsed pre/post Cobb angle and analyzable outcome.",
        },
        {
            "check": "missing_pre_cobb",
            "value": int(df["pre_cobb"].isna().sum()),
            "status": "pass" if int(df["pre_cobb"].isna().sum()) == 0 else "review",
            "note": "Parsed numeric baseline Cobb angle missingness.",
        },
        {
            "check": "missing_post_cobb",
            "value": int(df["post_cobb"].isna().sum()),
            "status": "pass" if int(df["post_cobb"].isna().sum()) == 1 else "review",
            "note": "One missing post-intervention Cobb angle is retained in cleaned data.",
        },
        {
            "check": "negative_cobb_change_rows",
            "value": int((df["cobb_change"] < 0).sum()),
            "status": "review",
            "note": "Negative values mean Cobb angle increased after intervention.",
        },
        {
            "check": "duration_gap_abs_gt_2_months",
            "value": int(df["duration_gap_months"].abs().gt(2).sum()),
            "status": "review",
            "note": "Compares recorded intervention months with date difference approximation.",
        },
    ]
    return pd.DataFrame(checks)


def variable_codebook() -> pd.DataFrame:
    rows = [
        ("source_sheet", "原始 sheet 名", "text", "审计追溯字段"),
        ("source_id", "原始序号", "identifier", "仅用于追溯，不作为模型特征"),
        ("treatment_group", "治疗组", "category", "brace_only 或 brace_training"),
        ("treatment_binary", "治疗组二元编码", "binary", "brace_training=1, brace_only=0"),
        ("sex", "性别", "category", "标准化为 female/male/unknown"),
        ("sex_female", "女性指示变量", "binary", "female=1"),
        ("age", "年龄", "years", "患者年龄"),
        ("bone_age_stage", "初查骨龄程度", "ordinal", "原始表中的骨龄程度编码"),
        ("brace_scan_date", "支具 3D 扫描时间", "date", "原始日期字段"),
        ("followup_date", "复查时间", "date", "原始日期字段"),
        ("intervention_months", "干预周期", "months", "原始记录的干预周期（月）"),
        ("date_interval_months", "日期推算随访月数", "months", "复查时间与扫描时间之差/30.4375"),
        ("pre_cobb", "干预前 Cobb 角", "degrees", "从原始 Cobb 字段解析出的数值"),
        ("post_cobb", "干预后 Cobb 角", "degrees", "从原始 Cobb 字段解析出的数值"),
        ("cobb_change", "Cobb 改善量", "degrees", "pre_cobb - post_cobb；越大表示改善越多"),
        ("cobb_change_pct", "Cobb 改善率", "ratio", "cobb_change / pre_cobb"),
        ("improved_5deg", "是否改善至少 5 度", "binary", "cobb_change >= 5"),
        ("weekly_training_hours", "每周训练时间", "hours", "仅训练组有原始记录"),
        ("training_exposure_hours", "训练暴露强度", "hours", "建模特征；纯支具组置 0"),
        ("outcome_observed", "疗效是否可分析", "binary", "post_cobb 非缺失时为 1"),
    ]
    return pd.DataFrame(rows, columns=["variable", "description", "unit_or_type", "notes"])


def _assign_subgroups(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["age_band"] = pd.cut(
        work["age"],
        bins=[-np.inf, 11, 14, np.inf],
        labels=["<=11", "12-14", ">=15"],
    ).astype(str)
    work["baseline_cobb_band"] = pd.cut(
        work["pre_cobb"],
        bins=[-np.inf, 19.999, 29.999, np.inf],
        labels=["<20", "20-29", ">=30"],
    ).astype(str)
    work["duration_band"] = pd.cut(
        work["intervention_months"],
        bins=[-np.inf, 5.999, 12, np.inf],
        labels=["<6", "6-12", ">12"],
    ).astype(str)
    return work


def subgroup_effects(df: pd.DataFrame) -> pd.DataFrame:
    work = _assign_subgroups(df[df["outcome_observed"].eq(1)].copy())
    subgroup_columns = ["age_band", "baseline_cobb_band", "duration_band", "sex"]
    rows: list[dict[str, float | int | str]] = []

    for subgroup in subgroup_columns:
        for level, level_df in work.groupby(subgroup, dropna=False):
            if str(level) == "nan" or level_df.empty:
                continue
            group_summary = (
                level_df.groupby("treatment_group")["cobb_change"]
                .agg(["count", "mean", "std"])
                .reset_index()
            )
            lookup = group_summary.set_index("treatment_group")
            brace_mean = lookup.loc["brace_only", "mean"] if "brace_only" in lookup.index else np.nan
            training_mean = (
                lookup.loc["brace_training", "mean"] if "brace_training" in lookup.index else np.nan
            )
            rows.append(
                {
                    "subgroup": subgroup,
                    "level": str(level),
                    "n_total": int(len(level_df)),
                    "n_brace_only": int(lookup.loc["brace_only", "count"])
                    if "brace_only" in lookup.index
                    else 0,
                    "n_brace_training": int(lookup.loc["brace_training", "count"])
                    if "brace_training" in lookup.index
                    else 0,
                    "mean_change_brace_only": brace_mean,
                    "mean_change_brace_training": training_mean,
                    "mean_diff_training_minus_brace": training_mean - brace_mean
                    if np.isfinite(brace_mean) and np.isfinite(training_mean)
                    else np.nan,
                    "note": "Descriptive subgroup effect; small cells should be interpreted cautiously.",
                }
            )
    return pd.DataFrame(rows)


def executive_summary_table(
    inference: pd.DataFrame,
    binary_outcome: pd.DataFrame,
    psm_effect: pd.DataFrame,
    model_metrics: pd.DataFrame,
) -> pd.DataFrame:
    raw = inference[inference["analysis"].eq("raw_mean_difference")].iloc[0]
    adjusted = inference[inference["analysis"].eq("covariate_adjusted_ols")].iloc[0]
    binary = binary_outcome[binary_outcome["analysis"].eq("risk_difference")].iloc[0]
    psm = psm_effect.iloc[0]
    best_model = model_metrics[model_metrics["model"].eq("ridge")].sort_values("rmse").iloc[0]

    rows = [
        {
            "section": "continuous_outcome",
            "metric": "Raw mean difference in Cobb change",
            "value": raw["estimate"],
            "detail": f"95% bootstrap CI [{raw['ci_lower']:.2f}, {raw['ci_upper']:.2f}], p={raw['p_value']:.3f}",
        },
        {
            "section": "continuous_outcome",
            "metric": "Adjusted treatment coefficient",
            "value": adjusted["estimate"],
            "detail": f"95% approx CI [{adjusted['ci_lower']:.2f}, {adjusted['ci_upper']:.2f}], p={adjusted['p_value']:.3f}",
        },
        {
            "section": "binary_outcome",
            "metric": "Risk difference for improved >=5deg",
            "value": binary["estimate"],
            "detail": f"95% bootstrap CI [{binary['ci_lower']:.2f}, {binary['ci_upper']:.2f}], p={binary['p_value']:.3f}",
        },
        {
            "section": "psm",
            "metric": "PSM matched ATT",
            "value": psm["estimate"],
            "detail": f"{int(psm['matched_pairs'])} matched outcome-complete pairs.",
        },
        {
            "section": "modeling",
            "metric": "Best Ridge CV RMSE",
            "value": best_model["rmse"],
            "detail": f"alpha={best_model['alpha']}, MAE={best_model['mae']:.2f}, R2={best_model['r2']:.3f}",
        },
    ]
    return pd.DataFrame(rows)
