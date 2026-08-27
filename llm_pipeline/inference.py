from __future__ import annotations

from dataclasses import dataclass
from math import erfc, sqrt

import numpy as np
import pandas as pd

from .config import BASELINE_COVARIATES, RANDOM_SEED


@dataclass(frozen=True)
class LinearModelResult:
    coefficients: pd.DataFrame
    r2: float
    n_obs: int


def _group_arrays(df: pd.DataFrame, outcome: str) -> tuple[np.ndarray, np.ndarray]:
    work = df[[outcome, "treatment_binary"]].dropna()
    control = work.loc[work["treatment_binary"].eq(0), outcome].to_numpy(dtype=float)
    treated = work.loc[work["treatment_binary"].eq(1), outcome].to_numpy(dtype=float)
    return control, treated


def permutation_mean_difference(
    df: pd.DataFrame,
    outcome: str,
    n_permutations: int = 5000,
    seed: int = RANDOM_SEED,
) -> dict[str, float | int]:
    work = df[[outcome, "treatment_binary"]].dropna()
    y = work[outcome].to_numpy(dtype=float)
    g = work["treatment_binary"].to_numpy(dtype=int)
    observed = float(y[g == 1].mean() - y[g == 0].mean())
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n_permutations):
        shuffled = rng.permutation(g)
        diff = float(y[shuffled == 1].mean() - y[shuffled == 0].mean())
        if abs(diff) >= abs(observed):
            exceed += 1
    return {
        "estimate": observed,
        "p_value_two_sided": (exceed + 1) / (n_permutations + 1),
        "n_permutations": n_permutations,
        "n_obs": int(len(work)),
    }


def bootstrap_mean_difference_ci(
    df: pd.DataFrame,
    outcome: str,
    n_bootstrap: int = 5000,
    seed: int = RANDOM_SEED,
) -> dict[str, float | int]:
    control, treated = _group_arrays(df, outcome)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        c = rng.choice(control, size=len(control), replace=True)
        t = rng.choice(treated, size=len(treated), replace=True)
        diffs[i] = t.mean() - c.mean()
    return {
        "ci_lower": float(np.quantile(diffs, 0.025)),
        "ci_upper": float(np.quantile(diffs, 0.975)),
        "n_bootstrap": n_bootstrap,
    }


def fit_linear_model(y: np.ndarray, x: np.ndarray, feature_names: list[str]) -> LinearModelResult:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    x_aug = np.column_stack([np.ones(len(x)), x])
    xtx_inv = np.linalg.pinv(x_aug.T @ x_aug)
    beta = xtx_inv @ x_aug.T @ y
    fitted = x_aug @ beta
    residuals = y - fitted
    n_obs, n_params = x_aug.shape
    residual_df = max(n_obs - n_params, 1)
    sigma2 = float((residuals.T @ residuals) / residual_df)
    se = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma2, 0))
    z_values = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    p_values = [erfc(abs(z) / sqrt(2)) if np.isfinite(z) else np.nan for z in z_values]
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    coefficients = pd.DataFrame(
        {
            "term": ["intercept", *feature_names],
            "coefficient": beta,
            "std_error": se,
            "z_or_t_approx": z_values,
            "p_value_normal_approx": p_values,
        }
    )
    return LinearModelResult(coefficients=coefficients, r2=float(r2), n_obs=int(n_obs))


def adjusted_treatment_model(df: pd.DataFrame, outcome: str = "cobb_change") -> LinearModelResult:
    feature_names = ["treatment_binary", *[name for name, _ in BASELINE_COVARIATES]]
    feature_names = list(dict.fromkeys(feature_names))
    work = df[[outcome, *feature_names]].dropna()
    return fit_linear_model(
        y=work[outcome].to_numpy(dtype=float),
        x=work[feature_names].to_numpy(dtype=float),
        feature_names=feature_names,
    )


def odds_ratio_with_ci(df: pd.DataFrame, outcome: str) -> dict[str, float]:
    work = df[[outcome, "treatment_binary"]].dropna()
    a = float(((work["treatment_binary"] == 1) & (work[outcome] == 1)).sum()) + 0.5
    b = float(((work["treatment_binary"] == 1) & (work[outcome] == 0)).sum()) + 0.5
    c = float(((work["treatment_binary"] == 0) & (work[outcome] == 1)).sum()) + 0.5
    d = float(((work["treatment_binary"] == 0) & (work[outcome] == 0)).sum()) + 0.5
    log_or = np.log((a * d) / (b * c))
    se = sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return {
        "odds_ratio": float(np.exp(log_or)),
        "ci_lower": float(np.exp(log_or - 1.96 * se)),
        "ci_upper": float(np.exp(log_or + 1.96 * se)),
    }


def run_inference(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    complete = df[df["outcome_observed"].eq(1)].copy()
    perm = permutation_mean_difference(complete, "cobb_change")
    boot = bootstrap_mean_difference_ci(complete, "cobb_change")
    adjusted = adjusted_treatment_model(complete)
    treatment_row = adjusted.coefficients[
        adjusted.coefficients["term"].eq("treatment_binary")
    ].iloc[0]

    rows: list[dict[str, object]] = []
    group_summary = complete.groupby("treatment_group")["cobb_change"].agg(["count", "mean", "std"])
    for group, row in group_summary.reset_index().iterrows():
        rows.append(
            {
                "analysis": "group_mean",
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
    rows.extend(
        [
            {
                "analysis": "raw_mean_difference",
                "estimand": "Mean Cobb change: brace_training - brace_only",
                "estimate": perm["estimate"],
                "std_error": np.nan,
                "p_value": perm["p_value_two_sided"],
                "ci_lower": boot["ci_lower"],
                "ci_upper": boot["ci_upper"],
                "n_obs": perm["n_obs"],
                "method_note": "Two-sided permutation p-value with bootstrap confidence interval.",
            },
            {
                "analysis": "covariate_adjusted_ols",
                "estimand": "Treatment coefficient adjusted for baseline covariates",
                "estimate": treatment_row["coefficient"],
                "std_error": treatment_row["std_error"],
                "p_value": treatment_row["p_value_normal_approx"],
                "ci_lower": treatment_row["coefficient"] - 1.96 * treatment_row["std_error"],
                "ci_upper": treatment_row["coefficient"] + 1.96 * treatment_row["std_error"],
                "n_obs": adjusted.n_obs,
                "method_note": f"OLS normal approximation; model R2={adjusted.r2:.3f}.",
            },
        ]
    )
    return pd.DataFrame(rows), adjusted.coefficients


def run_binary_outcome_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    complete = df[df["outcome_observed"].eq(1)].copy()
    complete["improved_5deg"] = pd.to_numeric(complete["improved_5deg"], errors="coerce")
    perm = permutation_mean_difference(complete, "improved_5deg")
    boot = bootstrap_mean_difference_ci(complete, "improved_5deg")
    odds = odds_ratio_with_ci(complete, "improved_5deg")
    adjusted = adjusted_treatment_model(complete, outcome="improved_5deg")
    treatment_row = adjusted.coefficients[
        adjusted.coefficients["term"].eq("treatment_binary")
    ].iloc[0]

    rows: list[dict[str, object]] = []
    raw_rates = complete.groupby("treatment_group")["improved_5deg"].agg(["count", "mean", "sum"])
    for _, row in raw_rates.reset_index().iterrows():
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
                "estimate": perm["estimate"],
                "std_error": np.nan,
                "p_value": perm["p_value_two_sided"],
                "ci_lower": boot["ci_lower"],
                "ci_upper": boot["ci_upper"],
                "n_obs": perm["n_obs"],
                "events": int(complete["improved_5deg"].sum()),
                "method_note": "Two-sided permutation p-value with bootstrap confidence interval.",
            },
            {
                "analysis": "odds_ratio",
                "estimand": "Odds ratio for improved>=5deg: training vs brace_only",
                "estimate": odds["odds_ratio"],
                "std_error": np.nan,
                "p_value": np.nan,
                "ci_lower": odds["ci_lower"],
                "ci_upper": odds["ci_upper"],
                "n_obs": int(len(complete)),
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
                "events": int(complete["improved_5deg"].sum()),
                "method_note": f"Linear probability model; model R2={adjusted.r2:.3f}.",
            },
        ]
    )
    return pd.DataFrame(rows), adjusted.coefficients
