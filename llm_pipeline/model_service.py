from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cleaning import load_clean_data, standardize_sex, validate_clean_data
from .config import DEFAULT_DATA_FILE, DEFAULT_OUTPUT_DIR, GROUP_LABELS, MODEL_FEATURES
from .modeling import fit_final_ridge, make_model_frame, run_modeling


ARTIFACT_FILENAME = "model_artifact.json"
SAFETY_NOTE = (
    "This exploratory model is for education and portfolio demonstration only; "
    "it is not clinical diagnosis, treatment advice, or a replacement for clinician judgment."
)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _as_float(value: object, field: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Field '{field}' must be numeric.") from exc
    if not np.isfinite(converted):
        raise ValueError(f"Field '{field}' must be finite.")
    return converted


def normalize_treatment_group(value: object) -> str:
    text = str(value).strip().lower()
    aliases = {
        "brace_only": "brace_only",
        "pure_brace": "brace_only",
        "bracing": "brace_only",
        "纯支具": "brace_only",
        "支具": "brace_only",
        "brace_training": "brace_training",
        "brace+training": "brace_training",
        "brace_training_group": "brace_training",
        "支具+训练": "brace_training",
        "支具加训练": "brace_training",
        "训练组": "brace_training",
    }
    if text in aliases:
        return aliases[text]
    raise ValueError("Field 'treatment_group' must be brace_only or brace_training.")


def build_patient_features(patient: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
    """Validate patient JSON and build the model feature vector."""
    required = [
        "age",
        "sex",
        "bone_age_stage",
        "pre_cobb",
        "intervention_months",
        "treatment_group",
    ]
    missing = [field for field in required if field not in patient]
    if missing:
        raise ValueError(f"Missing required patient field(s): {missing}")

    treatment_group = normalize_treatment_group(patient["treatment_group"])
    sex = standardize_sex(patient["sex"])
    if sex not in {"female", "male"}:
        raise ValueError("Field 'sex' must be female/male or 女/男.")

    weekly_training_hours = patient.get("weekly_training_hours")
    if treatment_group == "brace_training" and weekly_training_hours is None:
        raise ValueError("Field 'weekly_training_hours' is required for brace_training.")

    age = _as_float(patient["age"], "age")
    bone_age_stage = _as_float(patient["bone_age_stage"], "bone_age_stage")
    pre_cobb = _as_float(patient["pre_cobb"], "pre_cobb")
    intervention_months = _as_float(patient["intervention_months"], "intervention_months")
    weekly_training = 0.0 if treatment_group == "brace_only" else _as_float(weekly_training_hours, "weekly_training_hours")

    if not 0 < age < 30:
        raise ValueError("Field 'age' should be between 0 and 30 for this pediatric scoliosis demo.")
    if not 0 <= bone_age_stage <= 10:
        raise ValueError("Field 'bone_age_stage' should be between 0 and 10 for this dataset.")
    if not 0 < pre_cobb <= 90:
        raise ValueError("Field 'pre_cobb' should be between 0 and 90 degrees.")
    if not 0 < intervention_months <= 60:
        raise ValueError("Field 'intervention_months' should be between 0 and 60 months.")
    if weekly_training < 0 or weekly_training > 40:
        raise ValueError("Field 'weekly_training_hours' should be between 0 and 40.")

    post_cobb = patient.get("post_cobb")
    actual_cobb_change = None
    actual_improved_5deg = None
    if post_cobb is not None:
        post_cobb_value = _as_float(post_cobb, "post_cobb")
        if not 0 <= post_cobb_value <= 90:
            raise ValueError("Field 'post_cobb' should be between 0 and 90 degrees.")
        actual_cobb_change = pre_cobb - post_cobb_value
        actual_improved_5deg = actual_cobb_change >= 5
    else:
        post_cobb_value = None

    treatment_binary = 1.0 if treatment_group == "brace_training" else 0.0
    features = {
        "age": age,
        "sex_female": 1.0 if sex == "female" else 0.0,
        "bone_age_stage": bone_age_stage,
        "pre_cobb": pre_cobb,
        "intervention_months": intervention_months,
        "treatment_binary": treatment_binary,
        "training_exposure_hours": weekly_training if treatment_group == "brace_training" else 0.0,
        "treatment_x_pre_cobb": treatment_binary * pre_cobb,
        "duration_x_pre_cobb": intervention_months * pre_cobb,
    }
    normalized_patient = {
        "age": age,
        "sex": sex,
        "bone_age_stage": bone_age_stage,
        "pre_cobb": pre_cobb,
        "post_cobb": post_cobb_value,
        "intervention_months": intervention_months,
        "treatment_group": treatment_group,
        "treatment_label": GROUP_LABELS[treatment_group],
        "weekly_training_hours": weekly_training if treatment_group == "brace_training" else 0.0,
        "actual_cobb_change": actual_cobb_change,
        "actual_improved_5deg": actual_improved_5deg,
    }
    return normalized_patient, features


def artifact_path(output_dir: Path | str) -> Path:
    return Path(output_dir) / ARTIFACT_FILENAME


def train_model_artifact(data_file: Path | str = DEFAULT_DATA_FILE) -> dict[str, Any]:
    cleaned = load_clean_data(data_file)
    errors = validate_clean_data(cleaned)
    if errors:
        raise ValueError("Cleaned data validation failed: " + " | ".join(errors))

    model_df = make_model_frame(cleaned)
    modeling = run_modeling(cleaned)
    ridge_metrics = modeling["metrics"][modeling["metrics"]["model"].eq("ridge")].copy()
    best_row = ridge_metrics.sort_values("rmse").iloc[0]
    best_alpha = float(best_row["alpha"])
    coefficients, feature_importance = fit_final_ridge(model_df, best_alpha)
    coefficient_lookup = coefficients.set_index("term")

    artifact = {
        "schema_version": 1,
        "model_type": "ridge_standardized",
        "outcome": "cobb_change",
        "feature_names": MODEL_FEATURES,
        "alpha": best_alpha,
        "intercept": float(coefficient_lookup.loc["intercept", "coefficient_standardized"]),
        "coefficients": {
            feature: float(coefficient_lookup.loc[feature, "coefficient_standardized"])
            for feature in MODEL_FEATURES
        },
        "feature_mean": {
            feature: float(coefficient_lookup.loc[feature, "feature_mean"])
            for feature in MODEL_FEATURES
        },
        "feature_std": {
            feature: float(coefficient_lookup.loc[feature, "feature_std"])
            for feature in MODEL_FEATURES
        },
        "model_metrics": {
            "model": str(best_row["model"]),
            "alpha": best_alpha,
            "n_splits": int(best_row["n_splits"]),
            "mae": float(best_row["mae"]),
            "rmse": float(best_row["rmse"]),
            "r2": float(best_row["r2"]),
        },
        "feature_importance": json.loads(
            feature_importance.to_json(orient="records", force_ascii=False)
        ),
        "training_summary": {
            "data_file": str(Path(data_file).resolve()),
            "n_model_rows": int(len(model_df)),
            "group_counts": {
                str(k): int(v) for k, v in cleaned["treatment_group"].value_counts().items()
            },
        },
        "privacy_policy": {
            "patient_row_level_training_records_in_artifact": False,
            "artifact_contains": "coefficients, feature statistics, and aggregate model metrics only",
        },
        "safety_note": SAFETY_NOTE,
    }
    return artifact


def save_model_artifact(artifact: dict[str, Any], output_dir: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    path = artifact_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return path


def load_model_artifact(output_dir: Path | str = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    path = artifact_path(output_dir)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_model_artifact(
    data_file: Path | str = DEFAULT_DATA_FILE,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    path = artifact_path(output_dir)
    if path.exists():
        return load_model_artifact(output_dir)
    artifact = train_model_artifact(data_file)
    save_model_artifact(artifact, output_dir)
    return artifact


def predict_patient(patient: dict[str, Any], artifact: dict[str, Any]) -> dict[str, Any]:
    normalized_patient, features = build_patient_features(patient)
    prediction = float(artifact["intercept"])
    contributions: list[dict[str, Any]] = []
    for feature in artifact["feature_names"]:
        value = float(features[feature])
        mean = float(artifact["feature_mean"][feature])
        std = float(artifact["feature_std"][feature]) or 1.0
        coefficient = float(artifact["coefficients"][feature])
        standardized_value = (value - mean) / std
        contribution = coefficient * standardized_value
        prediction += contribution
        contributions.append(
            {
                "feature": feature,
                "value": value,
                "standardized_value": standardized_value,
                "coefficient_standardized": coefficient,
                "contribution": contribution,
                "direction": "increase" if contribution >= 0 else "decrease",
            }
        )

    contributions = sorted(contributions, key=lambda item: abs(item["contribution"]), reverse=True)
    rmse = float(artifact["model_metrics"]["rmse"])
    lower = prediction - rmse
    upper = prediction + rmse
    return {
        "normalized_patient": normalized_patient,
        "features": features,
        "predicted_cobb_change": prediction,
        "predicted_improved_5deg": prediction >= 5,
        "prediction_interval_note": (
            f"粗略误差范围约为 {lower:.2f} 到 {upper:.2f} 度，"
            f"即预测值 ± {rmse:.2f} 度；这是交叉验证 RMSE 的工程化解释，"
            "不是医学置信区间。"
        ),
        "model_metrics": artifact["model_metrics"],
        "feature_contributions": contributions,
        "safety_note": SAFETY_NOTE,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the patient QA Ridge model artifact.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_FILE), help="Input Excel workbook path.")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    artifact = train_model_artifact(args.data)
    path = save_model_artifact(artifact, args.out)
    print(json.dumps({"model_artifact": str(path.resolve()), "model_metrics": artifact["model_metrics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
