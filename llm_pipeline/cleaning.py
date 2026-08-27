from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import CANONICAL_COLUMNS, GROUP_LABELS, RAW_TO_STANDARD
from .data_ingestion import read_raw_workbook


def parse_numeric_angle(value: object) -> float:
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
    if text in {"女", "female", "f", "woman", "girl"}:
        return "female"
    if text in {"男", "male", "m", "man", "boy"}:
        return "male"
    return text or "unknown"


def _quality_flags(row: pd.Series) -> str:
    flags: list[str] = []
    if pd.isna(row.get("pre_cobb")):
        flags.append("missing_pre_cobb")
    if pd.isna(row.get("post_cobb")):
        flags.append("missing_post_cobb")
    if pd.notna(row.get("pre_cobb")) and not 0 <= float(row["pre_cobb"]) <= 90:
        flags.append("pre_cobb_out_of_range")
    if pd.notna(row.get("post_cobb")) and not 0 <= float(row["post_cobb"]) <= 90:
        flags.append("post_cobb_out_of_range")
    if pd.notna(row.get("cobb_change")) and abs(float(row["cobb_change"])) > 45:
        flags.append("large_cobb_change")
    if pd.notna(row.get("duration_gap_months")) and abs(float(row["duration_gap_months"])) > 2:
        flags.append("duration_date_gap_gt_2m")
    if pd.notna(row.get("weekly_training_hours")) and float(row["weekly_training_hours"]) < 0:
        flags.append("negative_training_hours")
    return ";".join(flags)


def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Create an analysis-ready table while preserving audit fields."""
    df = raw.rename(columns=RAW_TO_STANDARD).copy()

    for column in RAW_TO_STANDARD.values():
        if column not in df.columns:
            df[column] = np.nan

    df["treatment_binary"] = df["treatment_group"].eq("brace_training").astype(int)
    df["treatment_label"] = df["treatment_group"].map(GROUP_LABELS).fillna(df["treatment_group"])
    df["sex"] = df["sex"].map(standardize_sex)
    df["sex_female"] = df["sex"].eq("female").astype(int)

    for column in ["age", "bone_age_stage", "intervention_months", "weekly_training_hours"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["brace_scan_date"] = pd.to_datetime(df["brace_scan_date"], errors="coerce")
    df["followup_date"] = pd.to_datetime(df["followup_date"], errors="coerce")
    df["date_interval_days"] = (df["followup_date"] - df["brace_scan_date"]).dt.days
    df["date_interval_months"] = df["date_interval_days"] / 30.4375
    df["duration_gap_months"] = df["intervention_months"] - df["date_interval_months"]

    df["pre_cobb"] = df["pre_cobb_raw"].map(parse_numeric_angle)
    df["post_cobb"] = df["post_cobb_raw"].map(parse_numeric_angle)
    df["cobb_change"] = df["pre_cobb"] - df["post_cobb"]
    df["cobb_change_pct"] = df["cobb_change"] / df["pre_cobb"]
    df["improved_5deg"] = np.where(
        df["cobb_change"].notna(), df["cobb_change"].ge(5).astype(int), np.nan
    )
    df["outcome_observed"] = df["post_cobb"].notna().astype(int)

    df["training_hours_recorded"] = df["weekly_training_hours"].notna().astype(int)
    df["training_exposure_hours"] = np.where(
        df["treatment_binary"].eq(1),
        df["weekly_training_hours"].fillna(0),
        0,
    )
    df["quality_flags"] = df.apply(_quality_flags, axis=1)

    for column in CANONICAL_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan
    return df[CANONICAL_COLUMNS].copy()


def load_clean_data(path: Path | str) -> pd.DataFrame:
    return clean_data(read_raw_workbook(path))


def outcome_complete(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["outcome_observed"].eq(1)].copy()


def validate_clean_data(df: pd.DataFrame, expected_rows: int = 102) -> list[str]:
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
    if int(df["post_cobb"].isna().sum()) > 1:
        errors.append("Expected at most one missing parsed post_cobb value.")
    complete = df["outcome_observed"].eq(1)
    recomputed = df.loc[complete, "pre_cobb"] - df.loc[complete, "post_cobb"]
    if not np.allclose(recomputed, df.loc[complete, "cobb_change"], equal_nan=True):
        errors.append("cobb_change does not equal pre_cobb - post_cobb.")
    observed = df.loc[complete, "cobb_change"].ge(5).astype(int)
    if not observed.equals(df.loc[complete, "improved_5deg"].astype(int)):
        errors.append("improved_5deg does not match cobb_change >= 5.")
    return errors


def ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
