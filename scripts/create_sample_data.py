from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = PROJECT_ROOT / "data" / "sample_scoliosis_data.xlsx"
RANDOM_SEED = 20260703


def _make_group(
    n_rows: int,
    treatment_group: str,
    start_id: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    is_training = treatment_group == "brace_training"
    rows = []
    base_date = pd.Timestamp("2024-01-15")

    for offset in range(n_rows):
        age = int(np.clip(rng.normal(13.3 + (0.7 if is_training else 0.0), 2.0), 8, 18))
        sex = "女" if rng.random() < 0.74 else "男"
        bone_age = int(np.clip(np.rint((age - 8) / 2 + rng.normal(0, 0.8)), 0, 5))
        intervention_months = int(np.clip(np.rint(rng.normal(9.6 + (0.8 if is_training else 0.0), 3.1)), 4, 18))
        pre_cobb = float(np.clip(rng.normal(22.0 + (2.6 if is_training else 0.0), 6.0), 10, 42))
        expected_change = (
            3.8
            + 0.16 * intervention_months
            + 0.07 * (pre_cobb - 22)
            + (0.5 if is_training else 0.0)
            + rng.normal(0, 4.1)
        )
        post_cobb = float(np.clip(pre_cobb - expected_change, 3, 45))
        scan_date = base_date + pd.Timedelta(days=int(rng.integers(0, 120)))
        followup_date = scan_date + pd.Timedelta(
            days=int(intervention_months * 30.4375 + rng.normal(0, 18))
        )

        row = {
            "序号": start_id + offset,
            "性别": sex,
            "年龄": age,
            "初查骨龄程度": bone_age,
            "支具3D扫描时间": scan_date,
            "cobb角(干预前)": f"{pre_cobb:.1f}°",
            "干预周期（月）": intervention_months,
            "复查时间": followup_date,
            "cobb角（干预后）": f"{post_cobb:.1f}°",
        }
        if is_training:
            row["序号.1"] = start_id + offset
            row["每周训练时间（小时）"] = round(float(np.clip(rng.normal(4.0, 1.5), 1, 8)), 1)
        rows.append(row)

    df = pd.DataFrame(rows)
    if is_training:
        df.loc[df.index[-1], "cobb角（干预后）"] = np.nan
    return df


def create_sample_data(output_file: Path = OUTPUT_FILE) -> Path:
    rng = np.random.default_rng(RANDOM_SEED)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    brace_only = _make_group(73, "brace_only", 1, rng)
    brace_training = _make_group(29, "brace_training", 1, rng)

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        brace_only.to_excel(writer, sheet_name="纯支具组", index=False)
        brace_training.to_excel(writer, sheet_name="支具+训练组", index=False)
    return output_file


def main() -> None:
    path = create_sample_data()
    print(f"Wrote synthetic sample workbook: {path}")


if __name__ == "__main__":
    main()
