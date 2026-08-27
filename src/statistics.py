from __future__ import annotations

from dataclasses import dataclass
from math import erfc, sqrt

import numpy as np
import pandas as pd

from .config import GROUP_LABELS, RANDOM_SEED


BALANCE_COVARIATES = [
    ("age", "continuous"),
    ("sex_female", "binary"),
    ("bone_age_stage", "continuous"),
    ("pre_cobb", "continuous"),
    ("intervention_months", "continuous"),
]


@dataclass
class OLSResult:
    coefficients: pd.DataFrame
    r2: float
    n_obs: int
    residual_df: int


def _as_numeric(values: pd.Series) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)


def standardized_mean_difference(
    control: np.ndarray,
    treated: np.ndarray,
    variable_type: str = "continuous",
) -> float:
    control = control[~np.isnan(control)]
    treated = treated[~np.isnan(treated)]
    if len(control) == 0 or len(treated) == 0:
        return np.nan

    if variable_type == "binary":
        p_control = float(np.mean(control))
        p_treated = float(np.mean(treated))
        pooled = (p_control * (1 - p_control) + p_treated * (1 - p_treated)) / 2
        denom = sqrt(pooled) if pooled > 0 else np.nan
    else:
        denom = sqrt((np.var(control, ddof=1) + np.var(treated, ddof=1)) / 2)

    if not np.isfinite(denom) or denom == 0:
        return np.nan
    return float((np.mean(treated) - np.mean(control)) / denom)


def balance_table(
    df: pd.DataFrame,
    covariates: list[tuple[str, str]] | None = None,
    stage: str = "before_matching",
) -> pd.DataFrame:
    covariates = covariates or BALANCE_COVARIATES
    rows: list[dict[str, float | str | int]] = []
    for column, variable_type in covariates:
        control = _as_numeric(df.loc[df["treatment_binary"].eq(0), column])
        treated = _as_numeric(df.loc[df["treatment_binary"].eq(1), column])
        smd = standardized_mean_difference(control, treated, variable_type)
        rows.append(
            {
                "stage": stage,
                "variable": column,
                "type": variable_type,
                "n_brace_only": len(control),
                "n_brace_training": len(treated),
                "mean_brace_only": np.mean(control) if len(control) else np.nan,
                "mean_brace_training": np.mean(treated) if len(treated) else np.nan,
                "smd": smd,
                "abs_smd": abs(smd) if np.isfinite(smd) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def eda_summary(df: pd.DataFrame) -> pd.DataFrame:
    variables = [
        "age",
        "bone_age_stage",
        "pre_cobb",
        "post_cobb",
        "intervention_months",
        "weekly_training_hours",
        "cobb_change",
        "cobb_change_pct",
    ]
    rows: list[dict[str, float | str | int]] = []
    for group, group_df in df.groupby("treatment_group"):
        for variable in variables:
            values = pd.to_numeric(group_df[variable], errors="coerce")
            observed = values.dropna()
            if len(observed) == 0:
                stats = {
                    "mean": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "q25": np.nan,
                    "median": np.nan,
                    "q75": np.nan,
                    "max": np.nan,
                }
            else:
                stats = {
                    "mean": observed.mean(),
                    "std": observed.std(ddof=1),
                    "min": observed.min(),
                    "q25": observed.quantile(0.25),
                    "median": observed.median(),
                    "q75": observed.quantile(0.75),
                    "max": observed.max(),
                }
            rows.append(
                {
                    "treatment_group": group,
                    "treatment_label": GROUP_LABELS.get(group, group),
                    "variable": variable,
                    "count": int(len(observed)),
                    "missing": int(values.isna().sum()),
                    **stats,
                }
            )

    sex_counts = (
        df.groupby(["treatment_group", "sex"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    for _, row in sex_counts.iterrows():
        rows.append(
            {
                "treatment_group": row["treatment_group"],
                "treatment_label": GROUP_LABELS.get(row["treatment_group"], row["treatment_group"]),
                "variable": f"sex={row['sex']}",
                "count": int(row["count"]),
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


def permutation_mean_diff(
    df: pd.DataFrame,
    outcome: str,
    treatment: str = "treatment_binary",
    n_permutations: int = 5000,
    seed: int = RANDOM_SEED,
) -> dict[str, float | int]:
    work = df[[outcome, treatment]].dropna()
    y = work[outcome].to_numpy(dtype=float)
    g = work[treatment].to_numpy(dtype=int)
    observed = float(y[g == 1].mean() - y[g == 0].mean())

    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_permutations):
        shuffled = rng.permutation(g)
        diff = float(y[shuffled == 1].mean() - y[shuffled == 0].mean())
        if abs(diff) >= abs(observed):
            count += 1

    p_value = (count + 1) / (n_permutations + 1)
    return {
        "observed_mean_diff_training_minus_brace": observed,
        "p_value_two_sided": p_value,
        "n_permutations": n_permutations,
        "n_obs": len(work),
    }


def bootstrap_mean_diff_ci(
    df: pd.DataFrame,
    outcome: str,
    treatment: str = "treatment_binary",
    n_bootstrap: int = 5000,
    seed: int = RANDOM_SEED,
) -> dict[str, float | int]:
    work = df[[outcome, treatment]].dropna()
    control = work.loc[work[treatment].eq(0), outcome].to_numpy(dtype=float)
    treated = work.loc[work[treatment].eq(1), outcome].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        c = rng.choice(control, size=len(control), replace=True)
        t = rng.choice(treated, size=len(treated), replace=True)
        diffs[i] = t.mean() - c.mean()

    return {
        "ci_lower_2.5pct": float(np.quantile(diffs, 0.025)),
        "ci_upper_97.5pct": float(np.quantile(diffs, 0.975)),
        "n_bootstrap": n_bootstrap,
    }


def bootstrap_risk_difference_ci(
    df: pd.DataFrame,
    outcome: str,
    treatment: str = "treatment_binary",
    n_bootstrap: int = 5000,
    seed: int = RANDOM_SEED,
) -> dict[str, float | int]:
    work = df[[outcome, treatment]].dropna()
    control = work.loc[work[treatment].eq(0), outcome].to_numpy(dtype=float)
    treated = work.loc[work[treatment].eq(1), outcome].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        c = rng.choice(control, size=len(control), replace=True)
        t = rng.choice(treated, size=len(treated), replace=True)
        diffs[i] = t.mean() - c.mean()

    return {
        "ci_lower_2.5pct": float(np.quantile(diffs, 0.025)),
        "ci_upper_97.5pct": float(np.quantile(diffs, 0.975)),
        "n_bootstrap": n_bootstrap,
    }


def odds_ratio_with_ci(
    df: pd.DataFrame,
    outcome: str,
    treatment: str = "treatment_binary",
) -> dict[str, float]:
    work = df[[outcome, treatment]].dropna()
    treated_success = float(((work[treatment] == 1) & (work[outcome] == 1)).sum())
    treated_failure = float(((work[treatment] == 1) & (work[outcome] == 0)).sum())
    control_success = float(((work[treatment] == 0) & (work[outcome] == 1)).sum())
    control_failure = float(((work[treatment] == 0) & (work[outcome] == 0)).sum())

    # Haldane-Anscombe correction keeps the demo stable if a cell is zero.
    a = treated_success + 0.5
    b = treated_failure + 0.5
    c = control_success + 0.5
    d = control_failure + 0.5
    log_or = np.log((a * d) / (b * c))
    se = sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return {
        "odds_ratio": float(np.exp(log_or)),
        "ci_lower": float(np.exp(log_or - 1.96 * se)),
        "ci_upper": float(np.exp(log_or + 1.96 * se)),
    }


def fit_ols(y: np.ndarray, x: np.ndarray, feature_names: list[str]) -> OLSResult:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    x_aug = np.column_stack([np.ones(len(x)), x])
    names = ["intercept", *feature_names]

    xtx_inv = np.linalg.pinv(x_aug.T @ x_aug)
    beta = xtx_inv @ x_aug.T @ y
    fitted = x_aug @ beta
    residuals = y - fitted
    n_obs, n_params = x_aug.shape
    residual_df = max(n_obs - n_params, 1)
    sigma2 = float((residuals.T @ residuals) / residual_df)
    se = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma2, 0))
    z = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    p_values = np.array([erfc(abs(value) / sqrt(2)) if np.isfinite(value) else np.nan for value in z])

    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    coefficients = pd.DataFrame(
        {
            "term": names,
            "coefficient": beta,
            "std_error": se,
            "z_or_t_approx": z,
            "p_value_normal_approx": p_values,
        }
    )
    return OLSResult(coefficients=coefficients, r2=r2, n_obs=n_obs, residual_df=residual_df)


def adjusted_treatment_model(df: pd.DataFrame) -> OLSResult:
    features = [
        "treatment_binary",
        "age",
        "sex_female",
        "bone_age_stage",
        "pre_cobb",
        "intervention_months",
    ]
    work = df[["cobb_change", *features]].dropna()
    return fit_ols(
        y=work["cobb_change"].to_numpy(dtype=float),
        x=work[features].to_numpy(dtype=float),
        feature_names=features,
    )


def run_inference(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    complete = df[df["outcome_observed"].eq(1)].copy()
    perm = permutation_mean_diff(complete, "cobb_change")
    boot = bootstrap_mean_diff_ci(complete, "cobb_change")
    ols = adjusted_treatment_model(complete)
    treatment_row = ols.coefficients[
        ols.coefficients["term"].eq("treatment_binary")
    ].iloc[0]

    raw_means = (
        complete.groupby("treatment_group")["cobb_change"]
        .agg(["count", "mean", "std"])
        .reset_index()
    )

    rows = [
        {
            "analysis": "raw_mean_difference",
            "estimand": "Mean Cobb change: brace_training - brace_only",
            "estimate": perm["observed_mean_diff_training_minus_brace"],
            "std_error": np.nan,
            "p_value": perm["p_value_two_sided"],
            "ci_lower": boot["ci_lower_2.5pct"],
            "ci_upper": boot["ci_upper_97.5pct"],
            "n_obs": perm["n_obs"],
            "method_note": "Two-sided permutation p-value; bootstrap CI.",
        },
        {
            "analysis": "covariate_adjusted_ols",
            "estimand": "Treatment coefficient adjusted for baseline covariates",
            "estimate": treatment_row["coefficient"],
            "std_error": treatment_row["std_error"],
            "p_value": treatment_row["p_value_normal_approx"],
            "ci_lower": treatment_row["coefficient"] - 1.96 * treatment_row["std_error"],
            "ci_upper": treatment_row["coefficient"] + 1.96 * treatment_row["std_error"],
            "n_obs": ols.n_obs,
            "method_note": f"OLS normal-approx p-value; model R2={ols.r2:.3f}.",
        },
    ]

    group_rows = []
    for _, row in raw_means.iterrows():
        group_rows.append(
            {
                "analysis": "group_summary",
                "estimand": f"Mean Cobb change in {row['treatment_group']}",
                "estimate": row["mean"],
                "std_error": row["std"] / sqrt(row["count"]) if row["count"] > 0 else np.nan,
                "p_value": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "n_obs": int(row["count"]),
                "method_note": "Descriptive group mean.",
            }
        )

    inference = pd.DataFrame(group_rows + rows)
    return inference, ols.coefficients


def run_binary_outcome_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    complete = df[df["outcome_observed"].eq(1)].copy()
    complete["improved_5deg"] = pd.to_numeric(complete["improved_5deg"], errors="coerce")
    perm = permutation_mean_diff(complete, "improved_5deg")
    boot = bootstrap_risk_difference_ci(complete, "improved_5deg")
    odds = odds_ratio_with_ci(complete, "improved_5deg")

    raw_rates = (
        complete.groupby("treatment_group")["improved_5deg"]
        .agg(["count", "mean", "sum"])
        .reset_index()
    )

    adjusted_features = [
        "treatment_binary",
        "age",
        "sex_female",
        "bone_age_stage",
        "pre_cobb",
        "intervention_months",
    ]
    work = complete[["improved_5deg", *adjusted_features]].dropna()
    adjusted = fit_ols(
        y=work["improved_5deg"].to_numpy(dtype=float),
        x=work[adjusted_features].to_numpy(dtype=float),
        feature_names=adjusted_features,
    )
    treatment_row = adjusted.coefficients[
        adjusted.coefficients["term"].eq("treatment_binary")
    ].iloc[0]

    rows: list[dict[str, float | str | int]] = []
    for _, row in raw_rates.iterrows():
        rows.append(
            {
                "analysis": "group_rate",
                "estimand": f"Improved >=5deg rate in {row['treatment_group']}",
                "estimate": row["mean"],
                "std_error": np.nan,
                "p_value": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "n_obs": int(row["count"]),
                "events": int(row["sum"]),
                "method_note": "Descriptive event rate.",
            }
        )

    rows.extend(
        [
            {
                "analysis": "risk_difference",
                "estimand": "Pr(improved>=5deg): brace_training - brace_only",
                "estimate": perm["observed_mean_diff_training_minus_brace"],
                "std_error": np.nan,
                "p_value": perm["p_value_two_sided"],
                "ci_lower": boot["ci_lower_2.5pct"],
                "ci_upper": boot["ci_upper_97.5pct"],
                "n_obs": perm["n_obs"],
                "events": int(complete["improved_5deg"].sum()),
                "method_note": "Two-sided permutation p-value; bootstrap CI.",
            },
            {
                "analysis": "odds_ratio",
                "estimand": "Odds ratio for improved>=5deg: training vs brace_only",
                "estimate": odds["odds_ratio"],
                "std_error": np.nan,
                "p_value": np.nan,
                "ci_lower": odds["ci_lower"],
                "ci_upper": odds["ci_upper"],
                "n_obs": len(complete),
                "events": int(complete["improved_5deg"].sum()),
                "method_note": "Haldane-Anscombe corrected odds ratio.",
            },
            {
                "analysis": "adjusted_linear_probability",
                "estimand": "Adjusted treatment effect on Pr(improved>=5deg)",
                "estimate": treatment_row["coefficient"],
                "std_error": treatment_row["std_error"],
                "p_value": treatment_row["p_value_normal_approx"],
                "ci_lower": treatment_row["coefficient"] - 1.96 * treatment_row["std_error"],
                "ci_upper": treatment_row["coefficient"] + 1.96 * treatment_row["std_error"],
                "n_obs": adjusted.n_obs,
                "events": int(work["improved_5deg"].sum()),
                "method_note": f"Linear probability model; normal-approx p-value; R2={adjusted.r2:.3f}.",
            },
        ]
    )
    return pd.DataFrame(rows), adjusted.coefficients
