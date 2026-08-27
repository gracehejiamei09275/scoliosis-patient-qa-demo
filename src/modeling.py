from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RANDOM_SEED

MODEL_FEATURES = [
    "age",
    "sex_female",
    "bone_age_stage",
    "pre_cobb",
    "intervention_months",
    "treatment_binary",
    "training_exposure_hours",
    "treatment_x_pre_cobb",
    "duration_x_pre_cobb",
]


def make_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    work = df[df["outcome_observed"].eq(1)].copy()
    work["treatment_x_pre_cobb"] = work["treatment_binary"] * work["pre_cobb"]
    work["duration_x_pre_cobb"] = work["intervention_months"] * work["pre_cobb"]
    return work[["cobb_change", *MODEL_FEATURES]].dropna().copy()


def train_test_standardize(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0, ddof=0)
    std = np.where(std == 0, 1, std)
    return (x_train - mean) / std, (x_test - mean) / std, mean, std


def fit_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float,
) -> np.ndarray:
    x_aug = np.column_stack([np.ones(len(x_train)), x_train])
    penalty = np.eye(x_aug.shape[1]) * alpha
    penalty[0, 0] = 0
    return np.linalg.pinv(x_aug.T @ x_aug + penalty) @ x_aug.T @ y_train


def predict_ridge(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    x_aug = np.column_stack([np.ones(len(x)), x])
    return x_aug @ beta


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residuals = y_true - y_pred
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals**2)))
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"mae": mae, "rmse": rmse, "r2": r2}


def kfold_indices(n: int, n_splits: int = 5, seed: int = RANDOM_SEED) -> list[np.ndarray]:
    n_splits = min(n_splits, n)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    return [fold for fold in np.array_split(indices, n_splits) if len(fold) > 0]


def cross_validate_ridge(
    model_df: pd.DataFrame,
    feature_names: list[str] | None = None,
    alphas: list[float] | None = None,
    n_splits: int = 5,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_names = feature_names or MODEL_FEATURES
    alphas = alphas or [0.0, 0.1, 1.0, 10.0, 30.0]
    x = model_df[feature_names].to_numpy(dtype=float)
    y = model_df["cobb_change"].to_numpy(dtype=float)
    folds = kfold_indices(len(model_df), n_splits=n_splits, seed=seed)

    rows: list[dict[str, float | int | str]] = []
    prediction_frames: list[pd.DataFrame] = []

    baseline_predictions = np.full_like(y, y.mean(), dtype=float)
    baseline = regression_metrics(y, baseline_predictions)
    rows.append({"model": "mean_baseline", "alpha": np.nan, "n_splits": len(folds), **baseline})

    for alpha in alphas:
        fold_predictions = np.empty(len(y), dtype=float)
        for fold_id, test_idx in enumerate(folds, start=1):
            train_idx = np.setdiff1d(np.arange(len(y)), test_idx)
            x_train, x_test = x[train_idx], x[test_idx]
            y_train = y[train_idx]
            x_train_std, x_test_std, _, _ = train_test_standardize(x_train, x_test)
            beta = fit_ridge(x_train_std, y_train, alpha)
            fold_predictions[test_idx] = predict_ridge(x_test_std, beta)

            prediction_frames.append(
                pd.DataFrame(
                    {
                        "model": "ridge",
                        "alpha": alpha,
                        "fold": fold_id,
                        "row_position": test_idx,
                        "source_index": model_df.iloc[test_idx].index.to_numpy(),
                        "treatment_binary": model_df.iloc[test_idx]["treatment_binary"].to_numpy(dtype=int),
                        "actual_cobb_change": y[test_idx],
                        "predicted_cobb_change": fold_predictions[test_idx],
                    }
                )
            )

        metrics = regression_metrics(y, fold_predictions)
        rows.append({"model": "ridge", "alpha": alpha, "n_splits": len(folds), **metrics})

    metrics_df = pd.DataFrame(rows).sort_values(["model", "rmse"]).reset_index(drop=True)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    return metrics_df, predictions_df


def fit_final_ridge(model_df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    x = model_df[MODEL_FEATURES].to_numpy(dtype=float)
    y = model_df["cobb_change"].to_numpy(dtype=float)
    x_std, _, mean, std = train_test_standardize(x, x)
    beta = fit_ridge(x_std, y, alpha)
    return pd.DataFrame(
        {
            "term": ["intercept", *MODEL_FEATURES],
            "coefficient_standardized": beta,
            "feature_mean": [np.nan, *mean],
            "feature_std": [np.nan, *std],
            "alpha": alpha,
        }
    )


def prediction_metrics_by_treatment(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for treatment_binary, group_df in predictions.groupby("treatment_binary"):
        metrics = regression_metrics(
            group_df["actual_cobb_change"].to_numpy(dtype=float),
            group_df["predicted_cobb_change"].to_numpy(dtype=float),
        )
        label = "brace_training" if int(treatment_binary) == 1 else "brace_only"
        rows.append(
            {
                "treatment_group": label,
                "n_obs": len(group_df),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def feature_ablation(model_df: pd.DataFrame, best_alpha: float) -> pd.DataFrame:
    full_metrics, _ = cross_validate_ridge(
        model_df,
        feature_names=MODEL_FEATURES,
        alphas=[best_alpha],
    )
    full_rmse = float(full_metrics[full_metrics["model"].eq("ridge")].iloc[0]["rmse"])
    rows: list[dict[str, float | str]] = [
        {
            "removed_feature": "none_full_model",
            "rmse": full_rmse,
            "delta_rmse_vs_full": 0.0,
            "interpretation": "Reference model.",
        }
    ]

    for feature in MODEL_FEATURES:
        reduced_features = [name for name in MODEL_FEATURES if name != feature]
        metrics, _ = cross_validate_ridge(
            model_df,
            feature_names=reduced_features,
            alphas=[best_alpha],
        )
        rmse = float(metrics[metrics["model"].eq("ridge")].iloc[0]["rmse"])
        rows.append(
            {
                "removed_feature": feature,
                "rmse": rmse,
                "delta_rmse_vs_full": rmse - full_rmse,
                "interpretation": "Positive delta means removing this feature worsened CV RMSE.",
            }
        )
    return pd.DataFrame(rows).sort_values("delta_rmse_vs_full", ascending=False)


def run_modeling(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    model_df = make_model_frame(df)
    metrics, predictions = cross_validate_ridge(model_df)
    ridge_metrics = metrics[metrics["model"].eq("ridge")].copy()
    best_alpha = float(ridge_metrics.sort_values("rmse").iloc[0]["alpha"])
    best_predictions = predictions[predictions["alpha"].eq(best_alpha)].copy()
    coefficients = fit_final_ridge(model_df, alpha=best_alpha)
    group_metrics = prediction_metrics_by_treatment(best_predictions)
    ablation = feature_ablation(model_df, best_alpha=best_alpha)
    metadata = pd.DataFrame(
        [
            {
                "metric": "n_model_rows",
                "value": len(model_df),
            },
            {
                "metric": "best_ridge_alpha",
                "value": best_alpha,
            },
        ]
    )
    return {
        "model_frame": model_df,
        "metrics": metrics,
        "predictions": best_predictions,
        "coefficients": coefficients,
        "prediction_by_group": group_metrics,
        "feature_ablation": ablation,
        "metadata": metadata,
    }
