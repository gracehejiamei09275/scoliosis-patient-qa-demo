from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_DATA_FILE = PROJECT_ROOT / "脊柱侧弯数据统计表.xlsx"
SAMPLE_DATA_FILE = PROJECT_ROOT / "data" / "sample_scoliosis_data.xlsx"
DEFAULT_DATA_FILE = PRIVATE_DATA_FILE if PRIVATE_DATA_FILE.exists() else SAMPLE_DATA_FILE
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs_llm"

SHEET_TO_GROUP = {
    "纯支具组": "brace_only",
    "支具+训练组": "brace_training",
}

GROUP_LABELS = {
    "brace_only": "纯支具",
    "brace_training": "支具+训练",
}

RANDOM_SEED = 20260704

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
    "treatment_label",
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
    "quality_flags",
]

BASELINE_COVARIATES = [
    ("age", "continuous"),
    ("sex_female", "binary"),
    ("bone_age_stage", "continuous"),
    ("pre_cobb", "continuous"),
    ("intervention_months", "continuous"),
]

PSM_FEATURES = [name for name, _ in BASELINE_COVARIATES]

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
