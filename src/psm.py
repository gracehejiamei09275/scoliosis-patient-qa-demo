from __future__ import annotations

from math import erfc, log, sqrt

import numpy as np
import pandas as pd

from .config import RANDOM_SEED
from .statistics import BALANCE_COVARIATES, balance_table

PSM_FEATURES = ["age", "sex_female", "bone_age_stage", "pre_cobb", "intervention_months"]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35, 35)
    return 1 / (1 + np.exp(-z))


def _standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0, ddof=0)
    std = np.where(std == 0, 1, std)
    return (x - mean) / std, mean, std


def fit_logistic_regression(
    x: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.15,
    ridge_alpha: float = 0.01,
    max_iter: int = 6000,
) -> tuple[np.ndarray, np.ndarray]:
    x_std, mean, std = _standardize(x)
    x_aug = np.column_stack([np.ones(len(x_std)), x_std])
    beta = np.zeros(x_aug.shape[1])

    for _ in range(max_iter):
        p = _sigmoid(x_aug @ beta)
        gradient = x_aug.T @ (p - y) / len(y)
        penalty = ridge_alpha * beta / len(y)
        penalty[0] = 0
        beta -= learning_rate * (gradient + penalty)

    scale = np.concatenate([[1.0], std])
    center = np.concatenate([[0.0], mean])
    return beta, np.vstack([center, scale])


def predict_propensity(x: np.ndarray, beta: np.ndarray, center_scale: np.ndarray) -> np.ndarray:
    center = center_scale[0, 1:]
    scale = center_scale[1, 1:]
    x_std = (x - center) / scale
    x_aug = np.column_stack([np.ones(len(x_std)), x_std])
    return np.clip(_sigmoid(x_aug @ beta), 1e-6, 1 - 1e-6)


def estimate_propensity_scores(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df[["treatment_binary", *PSM_FEATURES]].dropna().copy()
    x = work[PSM_FEATURES].to_numpy(dtype=float)
    y = work["treatment_binary"].to_numpy(dtype=float)
    beta, center_scale = fit_logistic_regression(x, y)
    propensity = predict_propensity(x, beta, center_scale)

    scored = df.copy()
    scored["propensity_score"] = np.nan
    scored.loc[work.index, "propensity_score"] = propensity
    scored["propensity_logit"] = np.log(
        scored["propensity_score"] / (1 - scored["propensity_score"])
    )

    coef = pd.DataFrame(
        {
            "term": ["intercept", *PSM_FEATURES],
            "coefficient_standardized_logit": beta,
        }
    )
    return scored, coef


def nearest_neighbor_match(
    scored: pd.DataFrame,
    caliper: float | None = None,
) -> pd.DataFrame:
    candidates = scored.dropna(subset=["propensity_logit"]).copy()
    treated = candidates[candidates["treatment_binary"].eq(1)].copy()
    controls = candidates[candidates["treatment_binary"].eq(0)].copy()

    available_controls = set(controls.index.tolist())
    rows: list[dict[str, float | int]] = []
    treated = treated.assign(
        nearest_control_distance=treated["propensity_logit"].map(
            lambda v: float(np.min(np.abs(controls["propensity_logit"].to_numpy() - v)))
        )
    ).sort_values("nearest_control_distance", ascending=False)

    for pair_id, (treated_index, treated_row) in enumerate(treated.iterrows(), start=1):
        if not available_controls:
            break
        control_indices = np.array(sorted(available_controls))
        control_logits = controls.loc[control_indices, "propensity_logit"].to_numpy(dtype=float)
        distances = np.abs(control_logits - float(treated_row["propensity_logit"]))
        best_position = int(np.argmin(distances))
        best_control_index = int(control_indices[best_position])
        best_distance = float(distances[best_position])
        if caliper is not None and best_distance > caliper:
            continue
        available_controls.remove(best_control_index)
        control_row = controls.loc[best_control_index]
        rows.append(
            {
                "pair_id": pair_id,
                "treated_index": int(treated_index),
                "control_index": int(best_control_index),
                "treated_propensity": float(treated_row["propensity_score"]),
                "control_propensity": float(control_row["propensity_score"]),
                "logit_distance": best_distance,
                "treated_cobb_change": float(treated_row["cobb_change"])
                if pd.notna(treated_row["cobb_change"])
                else np.nan,
                "control_cobb_change": float(control_row["cobb_change"])
                if pd.notna(control_row["cobb_change"])
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def matched_sample(scored: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    if matches.empty:
        columns = list(scored.columns) + ["pair_id", "matched_role"]
        return pd.DataFrame(columns=columns)

    rows = []
    for _, match in matches.iterrows():
        treated = scored.loc[int(match["treated_index"])].copy()
        control = scored.loc[int(match["control_index"])].copy()
        treated["pair_id"] = int(match["pair_id"])
        treated["matched_role"] = "treated"
        control["pair_id"] = int(match["pair_id"])
        control["matched_role"] = "control"
        rows.extend([treated, control])
    return pd.DataFrame(rows)


def psm_sensitivity_analysis(
    scored: pd.DataFrame,
    caliper_multipliers: list[float | None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    caliper_multipliers = caliper_multipliers or [None, 0.50, 0.25, 0.20, 0.10]
    logit_sd = float(scored["propensity_logit"].dropna().std(ddof=1))
    summary_rows: list[dict[str, float | int | str | None]] = []
    balance_frames: list[pd.DataFrame] = []

    for multiplier in caliper_multipliers:
        caliper = None if multiplier is None else multiplier * logit_sd
        label = "no_caliper" if multiplier is None else f"{multiplier:.2f}_sd_logit"
        matches = nearest_neighbor_match(scored, caliper=caliper)
        matched = matched_sample(scored, matches)

        if matched.empty:
            mean_abs_smd = np.nan
            max_abs_smd = np.nan
        else:
            after = balance_table(
                matched,
                BALANCE_COVARIATES,
                stage=f"sensitivity_{label}",
            )
            mean_abs_smd = float(after["abs_smd"].mean())
            max_abs_smd = float(after["abs_smd"].max())
            balance_frames.append(after)

        effect = matched_effect(matches).iloc[0]
        summary_rows.append(
            {
                "caliper_label": label,
                "caliper_multiplier_sd_logit": multiplier,
                "caliper_logit_distance": caliper,
                "matched_pairs_total": int(len(matches)),
                "matched_pairs_outcome_complete": int(effect["matched_pairs"]),
                "att_cobb_change": effect["estimate"],
                "att_std_error": effect["std_error"],
                "att_ci_lower": effect["ci_lower"],
                "att_ci_upper": effect["ci_upper"],
                "mean_abs_smd_after": mean_abs_smd,
                "max_abs_smd_after": max_abs_smd,
            }
        )

    balances = pd.concat(balance_frames, ignore_index=True) if balance_frames else pd.DataFrame()
    return pd.DataFrame(summary_rows), balances


def matched_effect(matches: pd.DataFrame) -> pd.DataFrame:
    complete = matches.dropna(subset=["treated_cobb_change", "control_cobb_change"]).copy()
    complete["pair_difference"] = complete["treated_cobb_change"] - complete["control_cobb_change"]
    n_pairs = len(complete)
    if n_pairs == 0:
        return pd.DataFrame(
            [
                {
                    "analysis": "psm_att",
                    "matched_pairs": 0,
                    "estimate": np.nan,
                    "std_error": np.nan,
                    "p_value_normal_approx": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "method_note": "No outcome-complete matched pairs.",
                }
            ]
        )

    diffs = complete["pair_difference"].to_numpy(dtype=float)
    estimate = float(np.mean(diffs))
    std_error = float(np.std(diffs, ddof=1) / sqrt(n_pairs)) if n_pairs > 1 else np.nan
    z = estimate / std_error if std_error and np.isfinite(std_error) else np.nan
    p_value = erfc(abs(z) / sqrt(2)) if np.isfinite(z) else np.nan
    ci_lower = estimate - 1.96 * std_error if np.isfinite(std_error) else np.nan
    ci_upper = estimate + 1.96 * std_error if np.isfinite(std_error) else np.nan
    return pd.DataFrame(
        [
            {
                "analysis": "psm_att",
                "matched_pairs": n_pairs,
                "estimate": estimate,
                "std_error": std_error,
                "p_value_normal_approx": p_value,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "method_note": "1:1 nearest-neighbor ATT on Cobb change; treated minus matched control.",
            }
        ]
    )


def run_psm(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    scored, propensity_coefficients = estimate_propensity_scores(df)
    matches = nearest_neighbor_match(scored)
    matched = matched_sample(scored, matches)

    before = balance_table(scored.dropna(subset=PSM_FEATURES), BALANCE_COVARIATES, "before_matching")
    after = balance_table(matched, BALANCE_COVARIATES, "after_matching")
    balance = pd.concat([before, after], ignore_index=True)
    effect = matched_effect(matches)
    sensitivity_summary, sensitivity_balance = psm_sensitivity_analysis(scored)

    support = pd.DataFrame(
        [
            {
                "metric": "n_treated_available",
                "value": int(scored["treatment_binary"].eq(1).sum()),
            },
            {
                "metric": "n_control_available",
                "value": int(scored["treatment_binary"].eq(0).sum()),
            },
            {"metric": "n_matched_pairs", "value": int(len(matches))},
            {
                "metric": "treated_propensity_min",
                "value": float(scored.loc[scored["treatment_binary"].eq(1), "propensity_score"].min()),
            },
            {
                "metric": "treated_propensity_max",
                "value": float(scored.loc[scored["treatment_binary"].eq(1), "propensity_score"].max()),
            },
            {
                "metric": "control_propensity_min",
                "value": float(scored.loc[scored["treatment_binary"].eq(0), "propensity_score"].min()),
            },
            {
                "metric": "control_propensity_max",
                "value": float(scored.loc[scored["treatment_binary"].eq(0), "propensity_score"].max()),
            },
        ]
    )

    return {
        "scored": scored,
        "propensity_coefficients": propensity_coefficients,
        "matches": matches,
        "matched_sample": matched,
        "balance": balance,
        "effect": effect,
        "support": support,
        "sensitivity": sensitivity_summary,
        "sensitivity_balance": sensitivity_balance,
    }
