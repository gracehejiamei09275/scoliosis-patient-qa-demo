from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import DATA_FILE, SHEET_TO_GROUP


RAW_TO_STANDARD = {
    "序号": "source_id",
    "性别": "sex",
    "年龄": "age",
    "初查骨龄程度": "bone_age_stage",
    "支具3D扫描时间": "brace_scan_date",
    "cobb角(干预前)": "pre_cobb_raw",
    "干预周期（月）": "intervention_months",
    "复查时间": "followup_date",
    "cobb角（干预后）": "post_cobb_raw",
    "每周训练时间（小时）": "weekly_training_hours",
}

CANONICAL_COLUMNS = [
    "source_sheet",
    "source_id",
    "treatment_group",
    "treatment_binary",
    "sex",
    "sex_female",
    "age",
    "bone_age_stage",
    "brace_scan_date",
    "followup_date",
    "intervention_months",
    "date_interval_days",
    "date_interval_months",
    "duration_gap_months",
    "pre_cobb_raw",
    "post_cobb_raw",
    "pre_cobb",
    "post_cobb",
    "cobb_change",
    "cobb_change_pct",
    "improved_5deg",
    "weekly_training_hours",
    "training_exposure_hours",
    "training_hours_recorded",
    "outcome_observed",
]


def parse_numeric_angle(value: object) -> float:
    """Extract the first numeric Cobb angle from mixed Excel values."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else np.nan


def standardize_sex(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    text = str(value).strip().lower()
    female_values = {"女", "female", "f", "woman", "girl"}
    male_values = {"男", "male", "m", "man", "boy"}
    if text in female_values:
        return "female"
    if text in male_values:
        return "male"
    return text or "unknown"


def read_raw_workbook(path: Path | str = DATA_FILE) -> pd.DataFrame:
    """Read both treatment-group sheets and stack them without changing source data."""
    path = Path(path)
    frames: list[pd.DataFrame] = []
    for sheet_name, treatment_group in SHEET_TO_GROUP.items():
        sheet_df = pd.read_excel(path, sheet_name=sheet_name)
        sheet_df = sheet_df.drop(columns=["序号.1"], errors="ignore")
        sheet_df["source_sheet"] = sheet_name
        sheet_df["treatment_group"] = treatment_group
        frames.append(sheet_df)

    raw = pd.concat(frames, ignore_index=True, sort=False)
    return raw


def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Create an analysis-ready patient-level table."""
    df = raw.rename(columns=RAW_TO_STANDARD).copy()

    for column in RAW_TO_STANDARD.values():
        if column not in df.columns:
            df[column] = np.nan

    df["sex"] = df["sex"].map(standardize_sex)
    df["sex_female"] = (df["sex"] == "female").astype(int)
    df["treatment_binary"] = (df["treatment_group"] == "brace_training").astype(int)

    numeric_columns = [
        "age",
        "bone_age_stage",
        "intervention_months",
        "weekly_training_hours",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["brace_scan_date"] = pd.to_datetime(df["brace_scan_date"], errors="coerce")
    df["followup_date"] = pd.to_datetime(df["followup_date"], errors="coerce")
    df["date_interval_days"] = (
        df["followup_date"] - df["brace_scan_date"]
    ).dt.days
    df["date_interval_months"] = df["date_interval_days"] / 30.4375
    df["duration_gap_months"] = df["intervention_months"] - df["date_interval_months"]

    df["pre_cobb"] = df["pre_cobb_raw"].map(parse_numeric_angle)
    df["post_cobb"] = df["post_cobb_raw"].map(parse_numeric_angle)
    df["cobb_change"] = df["pre_cobb"] - df["post_cobb"]
    df["cobb_change_pct"] = df["cobb_change"] / df["pre_cobb"]
    df["improved_5deg"] = np.where(
        df["cobb_change"].notna(), (df["cobb_change"] >= 5).astype(int), np.nan
    )
    df["outcome_observed"] = df["post_cobb"].notna().astype(int)

    df["training_hours_recorded"] = df["weekly_training_hours"].notna().astype(int)
    df["training_exposure_hours"] = np.where(
        df["treatment_binary"].eq(1),
        df["weekly_training_hours"].fillna(0),
        0,
    )

    for column in CANONICAL_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan

    return df[CANONICAL_COLUMNS].copy()


def load_clean_data(path: Path | str = DATA_FILE) -> pd.DataFrame:
    return clean_data(read_raw_workbook(path))


def outcome_complete(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["outcome_observed"].eq(1)].copy()


def validate_clean_data(df: pd.DataFrame, expected_rows: int = 102) -> list[str]:
    """Return validation messages; empty means all checks passed."""
    errors: list[str] = []
    if len(df) != expected_rows:
        errors.append(f"Expected {expected_rows} rows, found {len(df)}.")

    counts = df["treatment_group"].value_counts().to_dict()
    if counts.get("brace_only", 0) != 73:
        errors.append(f"Expected 73 brace_only rows, found {counts.get('brace_only', 0)}.")
    if counts.get("brace_training", 0) != 29:
        errors.append(
            f"Expected 29 brace_training rows, found {counts.get('brace_training', 0)}."
        )

    if int(df["pre_cobb"].isna().sum()) != 0:
        errors.append("Expected no missing parsed pre_cobb values.")
    if int(df["post_cobb"].isna().sum()) != 1:
        errors.append("Expected exactly one missing parsed post_cobb value.")

    complete = df["outcome_observed"].eq(1)
    recomputed = df.loc[complete, "pre_cobb"] - df.loc[complete, "post_cobb"]
    if not np.allclose(recomputed, df.loc[complete, "cobb_change"], equal_nan=True):
        errors.append("cobb_change does not equal pre_cobb - post_cobb.")

    return errors


def ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
