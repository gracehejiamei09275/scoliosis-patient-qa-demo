from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_DATA_FILE = PROJECT_ROOT / "脊柱侧弯数据统计表.xlsx"
SAMPLE_DATA_FILE = PROJECT_ROOT / "data" / "sample_scoliosis_data.xlsx"
DATA_FILE = PRIVATE_DATA_FILE if PRIVATE_DATA_FILE.exists() else SAMPLE_DATA_FILE
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"

SHEET_TO_GROUP = {
    "纯支具组": "brace_only",
    "支具+训练组": "brace_training",
}

GROUP_LABELS = {
    "brace_only": "Brace only",
    "brace_training": "Brace + training",
}

RANDOM_SEED = 20260703
