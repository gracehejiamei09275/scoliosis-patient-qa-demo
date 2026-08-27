from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MODEL_FEATURES, RANDOM_SEED


def make_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    work = df[df["outcome_observed"].eq(1)].copy()
    work["treatment_x_pre_cobb"] = work["treatment_binary"] * work["pre_cobb"]
    work["duration_x_pre_cobb"] = work["intervention_months"] * work["pre_cobb"]
    model_columns = ["cobb_change", "treatment_group", *MODEL_FEATURES]
    return work[model_columns].dropna().copy()


def _standardize_train_test(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0, ddof=0)
    std = np.where(std == 0, 1, std)
    return (x_train - mean) / std, (x_test - mean) / std, mean, std


def _fit_ridge(x_train: np.ndarray, y_train: np.ndarray, alpha: float) -> np.ndarray:
    x_aug = np.column_stack([np.ones(len(x_train)), x_train])
    penalty = np.eye(x_aug.shape[1]) * alpha
    penalty[0, 0] = 0
    return np.linalg.pinv(x_aug.T @ x_aug + penalty) @ x_aug.T @ y_train


def _predict_ridge(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x]) @ beta


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residuals = y_true - y_pred
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals**2)))
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"mae": mae, "rmse": rmse, "r2": float(r2)}


def kfold_indices(n: int, n_splits: int = 5, seed: int = RANDOM_SEED) -> list[np.ndarray]:
    n_splits = min(max(n_splits, 2), n)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    return [fold for fold in np.array_split(indices, n_splits) if len(fold)]


def cross_validate_ridge(
    model_df: pd.DataFrame,
    alphas: list[float] | None = None,
    n_splits: int = 5,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    alphas = alphas or [0.0, 0.1, 1.0, 10.0, 30.0]
    x = model_df[MODEL_FEATURES].to_numpy(dtype=float)
    y = model_df["cobb_change"].to_numpy(dtype=float)
    folds = kfold_indices(len(model_df), n_splits=n_splits, seed=seed)
    rows: list[dict[str, object]] = [
        {
            "model": "mean_baseline",
            "alpha": np.nan,
            "n_splits": len(folds),
            **regression_metrics(y, np.full_like(y, y.mean(), dtype=float)),
        }
    ]
    prediction_frames: list[pd.DataFrame] = []

    for alpha in alphas:
        predictions = np.empty(len(y), dtype=float)
        fold_ids = np.empty(len(y), dtype=int)
        for fold_id, test_idx in enumerate(folds, start=1):
            train_idx = np.setdiff1d(np.arange(len(y)), test_idx)
            x_train, x_test = x[train_idx], x[test_idx]
            y_train = y[train_idx]
            x_train_std, x_test_std, _, _ = _standardize_train_test(x_train, x_test)
            beta = _fit_ridge(x_train_std, y_train, alpha)
            predictions[test_idx] = _predict_ridge(x_test_std, beta)
            fold_ids[test_idx] = fold_id
        rows.append(
            {
                "model": "ridge",
                "alpha": alpha,
                "n_splits": len(folds),
                **regression_metrics(y, predictions),
            }
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "model": "ridge",
                    "alpha": alpha,
                    "fold": fold_ids,
                    "prediction_id": np.arange(1, len(y) + 1),
                    "treatment_group": model_df["treatment_group"].to_numpy(),
                    "actual_cobb_change": y,
                    "predicted_cobb_change": predictions,
                    "absolute_error": np.abs(y - predictions),
                }
            )
        )

    metrics = pd.DataFrame(rows).sort_values(["model", "rmse"]).reset_index(drop=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    return metrics, predictions


def fit_final_ridge(model_df: pd.DataFrame, alpha: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = model_df[MODEL_FEATURES].to_numpy(dtype=float)
    y = model_df["cobb_change"].to_numpy(dtype=float)
    x_std, _, mean, std = _standardize_train_test(x, x)
    beta = _fit_ridge(x_std, y, alpha)
    coefficients = pd.DataFrame(
        {
            "term": ["intercept", *MODEL_FEATURES],
            "coefficient_standardized": beta,
            "feature_mean": [np.nan, *mean],
            "feature_std": [np.nan, *std],
            "alpha": alpha,
        }
    )
    feature_importance = coefficients[coefficients["term"].ne("intercept")].copy()
    feature_importance["importance"] = feature_importance["coefficient_standardized"].abs()
    feature_importance = feature_importance.sort_values("importance", ascending=False)
    return coefficients, feature_importance[["term", "importance", "coefficient_standardized"]]


def prediction_metrics_by_group(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group, group_df in predictions.groupby("treatment_group"):
        rows.append(
            {
                "treatment_group": group,
                "n_obs": int(len(group_df)),
                **regression_metrics(
                    group_df["actual_cobb_change"].to_numpy(dtype=float),
                    group_df["predicted_cobb_change"].to_numpy(dtype=float),
                ),
            }
        )
    return pd.DataFrame(rows)


def run_modeling(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    model_df = make_model_frame(df)
    metrics, predictions = cross_validate_ridge(model_df)
    ridge_metrics = metrics[metrics["model"].eq("ridge")].copy()
    best_alpha = float(ridge_metrics.sort_values("rmse").iloc[0]["alpha"])
    best_predictions = predictions[predictions["alpha"].eq(best_alpha)].copy()
    coefficients, feature_importance = fit_final_ridge(model_df, alpha=best_alpha)
    group_metrics = prediction_metrics_by_group(best_predictions)
    metadata = pd.DataFrame(
        [
            {"metric": "n_model_rows", "value": int(len(model_df))},
            {"metric": "best_ridge_alpha", "value": best_alpha},
            {"metric": "leakage_guard", "value": "No source_id/date/raw angle columns in predictions."},
        ]
    )
    return {
        "model_frame": model_df,
        "metrics": metrics,
        "predictions": best_predictions,
        "coefficients": coefficients,
        "feature_importance": feature_importance,
        "prediction_by_group": group_metrics,
        "metadata": metadata,
    }
