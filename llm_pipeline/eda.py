from __future__ import annotations

import numpy as np
import pandas as pd

from .config import GROUP_LABELS


EDA_VARIABLES = [
    "age",
    "bone_age_stage",
    "pre_cobb",
    "post_cobb",
    "intervention_months",
    "weekly_training_hours",
    "training_exposure_hours",
    "cobb_change",
    "cobb_change_pct",
]


def data_quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    flagged = df["quality_flags"].fillna("").ne("")
    checks = [
        ("row_count", len(df), "pass" if len(df) == 102 else "review", "Expected 102 rows."),
        (
            "outcome_complete_rows",
            int(df["outcome_observed"].sum()),
            "pass" if int(df["outcome_observed"].sum()) >= 101 else "review",
            "Rows with parsed post-treatment Cobb angle.",
        ),
        (
            "missing_pre_cobb",
            int(df["pre_cobb"].isna().sum()),
            "pass" if int(df["pre_cobb"].isna().sum()) == 0 else "review",
            "Baseline Cobb angle parsed from mixed Excel values.",
        ),
        (
            "missing_post_cobb",
            int(df["post_cobb"].isna().sum()),
            "pass" if int(df["post_cobb"].isna().sum()) <= 1 else "review",
            "Post-intervention Cobb angle missingness.",
        ),
        (
            "negative_cobb_change_rows",
            int(df["cobb_change"].lt(0).sum()),
            "review",
            "Negative values mean Cobb angle increased after intervention.",
        ),
        (
            "rows_with_quality_flags",
            int(flagged.sum()),
            "review" if int(flagged.sum()) else "pass",
            "Rule-based audit flags for dates, missingness, and ranges.",
        ),
    ]
    return pd.DataFrame(checks, columns=["check", "value", "status", "note"])


def eda_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group, group_df in df.groupby("treatment_group", dropna=False):
        for variable in EDA_VARIABLES:
            values = pd.to_numeric(group_df[variable], errors="coerce")
            observed = values.dropna()
            summary = {
                "mean": observed.mean() if len(observed) else np.nan,
                "std": observed.std(ddof=1) if len(observed) > 1 else np.nan,
                "min": observed.min() if len(observed) else np.nan,
                "q25": observed.quantile(0.25) if len(observed) else np.nan,
                "median": observed.median() if len(observed) else np.nan,
                "q75": observed.quantile(0.75) if len(observed) else np.nan,
                "max": observed.max() if len(observed) else np.nan,
            }
            rows.append(
                {
                    "treatment_group": group,
                    "treatment_label": GROUP_LABELS.get(group, group),
                    "variable": variable,
                    "count": int(len(observed)),
                    "missing": int(values.isna().sum()),
                    **summary,
                }
            )
    sex_counts = df.groupby(["treatment_group", "sex"], dropna=False).size()
    for (group, sex), count in sex_counts.items():
        rows.append(
            {
                "treatment_group": group,
                "treatment_label": GROUP_LABELS.get(group, group),
                "variable": f"sex={sex}",
                "count": int(count),
                "missing": 0,
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "q25": np.nan,
                "median": np.nan,
                "q75": np.nan,
                "max": np.nan,
            }
        )
    return pd.DataFrame(rows)


def treatment_effect_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    complete = df[df["outcome_observed"].eq(1)].copy()
    rows: list[dict[str, object]] = []
    for group, group_df in complete.groupby("treatment_group"):
        rows.append(
            {
                "treatment_group": group,
                "treatment_label": GROUP_LABELS.get(group, group),
                "n": int(len(group_df)),
                "mean_cobb_change": float(group_df["cobb_change"].mean()),
                "median_cobb_change": float(group_df["cobb_change"].median()),
                "improved_5deg_rate": float(group_df["improved_5deg"].mean()),
                "mean_intervention_months": float(group_df["intervention_months"].mean()),
                "mean_pre_cobb": float(group_df["pre_cobb"].mean()),
            }
        )
    return pd.DataFrame(rows)
