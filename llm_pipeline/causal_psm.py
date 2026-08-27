from __future__ import annotations

from math import erfc, sqrt

import numpy as np
import pandas as pd

from .config import BASELINE_COVARIATES, PSM_FEATURES


def standardized_mean_difference(
    control: np.ndarray,
    treated: np.ndarray,
    variable_type: str = "continuous",
) -> float:
    control = np.asarray(control, dtype=float)
    treated = np.asarray(treated, dtype=float)
    control = control[~np.isnan(control)]
    treated = treated[~np.isnan(treated)]
    if len(control) == 0 or len(treated) == 0:
        return np.nan
    if variable_type == "binary":
        pc = float(control.mean())
        pt = float(treated.mean())
        pooled = (pc * (1 - pc) + pt * (1 - pt)) / 2
        denom = sqrt(pooled) if pooled > 0 else np.nan
    else:
        denom = sqrt((np.var(control, ddof=1) + np.var(treated, ddof=1)) / 2)
    if not np.isfinite(denom) or denom == 0:
        return np.nan
    return float((treated.mean() - control.mean()) / denom)


def balance_table(
    df: pd.DataFrame,
    covariates: list[tuple[str, str]] | None = None,
    stage: str = "before_matching",
) -> pd.DataFrame:
    covariates = covariates or BASELINE_COVARIATES
    rows: list[dict[str, object]] = []
    for column, variable_type in covariates:
        control = pd.to_numeric(
            df.loc[df["treatment_binary"].eq(0), column], errors="coerce"
        ).dropna()
        treated = pd.to_numeric(
            df.loc[df["treatment_binary"].eq(1), column], errors="coerce"
        ).dropna()
        smd = standardized_mean_difference(
            control.to_numpy(dtype=float),
            treated.to_numpy(dtype=float),
            variable_type,
        )
        rows.append(
            {
                "stage": stage,
                "variable": column,
                "type": variable_type,
                "n_brace_only": int(len(control)),
                "n_brace_training": int(len(treated)),
                "mean_brace_only": float(control.mean()) if len(control) else np.nan,
                "mean_brace_training": float(treated.mean()) if len(treated) else np.nan,
                "smd": smd,
                "abs_smd": abs(smd) if np.isfinite(smd) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(z, -35, 35)))


def _standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0, ddof=0)
    std = np.where(std == 0, 1, std)
    return (x - mean) / std, mean, std


def fit_logistic_propensity(
    x: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.15,
    ridge_alpha: float = 0.01,
    max_iter: int = 6000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_std, mean, std = _standardize(x)
    x_aug = np.column_stack([np.ones(len(x_std)), x_std])
    beta = np.zeros(x_aug.shape[1], dtype=float)
    for _ in range(max_iter):
        p = _sigmoid(x_aug @ beta)
        gradient = x_aug.T @ (p - y) / len(y)
        penalty = ridge_alpha * beta / len(y)
        penalty[0] = 0
        beta -= learning_rate * (gradient + penalty)
    return beta, mean, std


def predict_propensity(x: np.ndarray, beta: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    x_std = (x - mean) / std
    x_aug = np.column_stack([np.ones(len(x_std)), x_std])
    return np.clip(_sigmoid(x_aug @ beta), 1e-6, 1 - 1e-6)


def estimate_propensity_scores(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df[["treatment_binary", *PSM_FEATURES]].dropna().copy()
    x = work[PSM_FEATURES].to_numpy(dtype=float)
    y = work["treatment_binary"].to_numpy(dtype=float)
    beta, mean, std = fit_logistic_propensity(x, y)
    scored = df.copy()
    scored["propensity_score"] = np.nan
    scored.loc[work.index, "propensity_score"] = predict_propensity(x, beta, mean, std)
    scored["propensity_logit"] = np.log(scored["propensity_score"] / (1 - scored["propensity_score"]))
    coefficients = pd.DataFrame(
        {
            "term": ["intercept", *PSM_FEATURES],
            "coefficient_standardized_logit": beta,
            "feature_mean": [np.nan, *mean],
            "feature_std": [np.nan, *std],
        }
    )
    return scored, coefficients


def common_support_summary(scored: pd.DataFrame) -> pd.DataFrame:
    scored = scored.dropna(subset=["propensity_score"]).copy()
    treated = scored.loc[scored["treatment_binary"].eq(1), "propensity_score"]
    control = scored.loc[scored["treatment_binary"].eq(0), "propensity_score"]
    lower = max(float(treated.min()), float(control.min()))
    upper = min(float(treated.max()), float(control.max()))
    inside = scored["propensity_score"].between(lower, upper, inclusive="both")
    return pd.DataFrame(
        [
            {"metric": "support_lower", "value": lower},
            {"metric": "support_upper", "value": upper},
            {"metric": "rows_inside_common_support", "value": int(inside.sum())},
            {"metric": "rows_outside_common_support", "value": int((~inside).sum())},
            {"metric": "treated_available", "value": int(scored["treatment_binary"].eq(1).sum())},
            {"metric": "control_available", "value": int(scored["treatment_binary"].eq(0).sum())},
        ]
    )


def nearest_neighbor_match(scored: pd.DataFrame, caliper: float | None = None) -> pd.DataFrame:
    candidates = scored.dropna(subset=["propensity_logit"]).copy()
    treated = candidates[candidates["treatment_binary"].eq(1)].copy()
    controls = candidates[candidates["treatment_binary"].eq(0)].copy()
    available_controls = set(controls.index.tolist())
    rows: list[dict[str, object]] = []
    control_logits = controls["propensity_logit"].to_numpy(dtype=float)

    treated = treated.assign(
        nearest_control_distance=treated["propensity_logit"].map(
            lambda value: float(np.min(np.abs(control_logits - value)))
        )
    ).sort_values("nearest_control_distance", ascending=False)

    for pair_id, (treated_index, treated_row) in enumerate(treated.iterrows(), start=1):
        if not available_controls:
            break
        control_indices = np.array(sorted(available_controls), dtype=int)
        distances = np.abs(
            controls.loc[control_indices, "propensity_logit"].to_numpy(dtype=float)
            - float(treated_row["propensity_logit"])
        )
        best_pos = int(np.argmin(distances))
        best_index = int(control_indices[best_pos])
        best_distance = float(distances[best_pos])
        if caliper is not None and best_distance > caliper:
            continue
        available_controls.remove(best_index)
        control_row = controls.loc[best_index]
        rows.append(
            {
                "pair_id": int(pair_id),
                "treated_row": int(treated_index),
                "control_row": int(best_index),
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
        return pd.DataFrame(columns=list(scored.columns) + ["pair_id", "matched_role"])
    rows = []
    for _, match in matches.iterrows():
        treated = scored.loc[int(match["treated_row"])].copy()
        control = scored.loc[int(match["control_row"])].copy()
        treated["pair_id"] = int(match["pair_id"])
        treated["matched_role"] = "treated"
        control["pair_id"] = int(match["pair_id"])
        control["matched_role"] = "control"
        rows.extend([treated, control])
    return pd.DataFrame(rows)


def matched_effect(matches: pd.DataFrame) -> pd.DataFrame:
    complete = matches.dropna(subset=["treated_cobb_change", "control_cobb_change"]).copy()
    if complete.empty:
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
    diffs = complete["treated_cobb_change"] - complete["control_cobb_change"]
    estimate = float(diffs.mean())
    std_error = float(diffs.std(ddof=1) / sqrt(len(diffs))) if len(diffs) > 1 else np.nan
    z_value = estimate / std_error if np.isfinite(std_error) and std_error else np.nan
    p_value = erfc(abs(z_value) / sqrt(2)) if np.isfinite(z_value) else np.nan
    return pd.DataFrame(
        [
            {
                "analysis": "psm_att",
                "matched_pairs": int(len(diffs)),
                "estimate": estimate,
                "std_error": std_error,
                "p_value_normal_approx": p_value,
                "ci_lower": estimate - 1.96 * std_error if np.isfinite(std_error) else np.nan,
                "ci_upper": estimate + 1.96 * std_error if np.isfinite(std_error) else np.nan,
                "method_note": "1:1 nearest-neighbor ATT on Cobb change; treated minus matched control.",
            }
        ]
    )


def psm_sensitivity(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    logit_sd = float(scored["propensity_logit"].dropna().std(ddof=1))
    multipliers: list[float | None] = [None, 0.50, 0.25, 0.20, 0.10]
    rows: list[dict[str, object]] = []
    balance_frames: list[pd.DataFrame] = []
    for multiplier in multipliers:
        caliper = None if multiplier is None else multiplier * logit_sd
        label = "no_caliper" if multiplier is None else f"{multiplier:.2f}_sd_logit"
        matches = nearest_neighbor_match(scored, caliper=caliper)
        matched = matched_sample(scored, matches)
        if matched.empty:
            mean_abs_smd = np.nan
            max_abs_smd = np.nan
        else:
            balance = balance_table(matched, stage=f"sensitivity_{label}")
            mean_abs_smd = float(balance["abs_smd"].mean())
            max_abs_smd = float(balance["abs_smd"].max())
            balance_frames.append(balance)
        effect = matched_effect(matches).iloc[0]
        rows.append(
            {
                "caliper_label": label,
                "caliper_multiplier_sd_logit": multiplier,
                "caliper_logit_distance": caliper,
                "matched_pairs_total": int(len(matches)),
                "matched_pairs_outcome_complete": int(effect["matched_pairs"]),
                "att_cobb_change": effect["estimate"],
                "att_ci_lower": effect["ci_lower"],
                "att_ci_upper": effect["ci_upper"],
                "mean_abs_smd_after": mean_abs_smd,
                "max_abs_smd_after": max_abs_smd,
            }
        )
    sensitivity_balance = (
        pd.concat(balance_frames, ignore_index=True) if balance_frames else pd.DataFrame()
    )
    return pd.DataFrame(rows), sensitivity_balance


def run_psm(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    scored, propensity_coefficients = estimate_propensity_scores(df)
    matches = nearest_neighbor_match(scored)
    matched = matched_sample(scored, matches)
    before = balance_table(scored.dropna(subset=PSM_FEATURES), stage="before_matching")
    after = balance_table(matched, stage="after_matching")
    balance = pd.concat([before, after], ignore_index=True)
    sensitivity, sensitivity_balance = psm_sensitivity(scored)
    support = common_support_summary(scored)
    support = pd.concat(
        [
            support,
            pd.DataFrame(
                [{"metric": "matched_pairs_no_caliper", "value": int(len(matches))}]
            ),
        ],
        ignore_index=True,
    )
    return {
        "scored": scored,
        "propensity_coefficients": propensity_coefficients,
        "matches": matches,
        "matched_sample": matched,
        "balance": balance,
        "effect": matched_effect(matches),
        "support": support,
        "sensitivity": sensitivity,
        "sensitivity_balance": sensitivity_balance,
    }
