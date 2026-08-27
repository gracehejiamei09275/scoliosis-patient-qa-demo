from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_FILE, SAMPLE_DATA_FILE
from src.data_cleaning import load_clean_data, validate_clean_data
from src.pipeline import run_pipeline


def assert_nonempty_csv(path: Path) -> None:
    assert path.exists(), f"Missing output: {path}"
    df = pd.read_csv(path)
    assert not df.empty, f"Output is empty: {path}"


def test_cleaning_contract() -> None:
    cleaned = load_clean_data(DATA_FILE)
    assert len(cleaned) == 102
    assert cleaned["treatment_group"].value_counts().to_dict() == {
        "brace_only": 73,
        "brace_training": 29,
    }
    assert int(cleaned["pre_cobb"].isna().sum()) == 0
    assert int(cleaned["post_cobb"].isna().sum()) == 1
    assert int(cleaned["outcome_observed"].sum()) == 101
    assert validate_clean_data(cleaned) == []


def test_sample_data_contract() -> None:
    cleaned = load_clean_data(SAMPLE_DATA_FILE)
    assert len(cleaned) == 102
    assert cleaned["treatment_group"].value_counts().to_dict() == {
        "brace_only": 73,
        "brace_training": 29,
    }
    assert int(cleaned["pre_cobb"].isna().sum()) == 0
    assert int(cleaned["post_cobb"].isna().sum()) == 1
    assert validate_clean_data(cleaned) == []


def test_pipeline_outputs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "outputs"
        results = run_pipeline(DATA_FILE, output_dir=output_dir, make_figures=True)
        assert results["manifest"]["n_rows_cleaned"] == 102
        assert results["manifest"]["n_rows_outcome_complete"] == 101

        assert (output_dir / "cleaned_data.csv").exists()
        for name in [
            "data_quality_summary.csv",
            "data_dictionary.csv",
            "eda_summary.csv",
            "executive_summary.csv",
            "inference_results.csv",
            "binary_outcome_results.csv",
            "psm_balance.csv",
            "psm_effect.csv",
            "psm_sensitivity.csv",
            "model_metrics.csv",
            "model_coefficients.csv",
            "model_feature_ablation.csv",
            "model_prediction_by_group.csv",
            "subgroup_effects.csv",
        ]:
            assert_nonempty_csv(output_dir / "tables" / name)

        matches = pd.read_csv(output_dir / "tables" / "psm_matches.csv")
        assert len(matches) > 0

        metrics = pd.read_csv(output_dir / "tables" / "model_metrics.csv")
        assert "ridge" in set(metrics["model"])

        binary = pd.read_csv(output_dir / "tables" / "binary_outcome_results.csv")
        assert "risk_difference" in set(binary["analysis"])

        sensitivity = pd.read_csv(output_dir / "tables" / "psm_sensitivity.csv")
        assert len(sensitivity) >= 3

        report = output_dir / "report.html"
        assert report.exists()
        assert "脊柱侧弯治疗效果数据分析 Demo Report" in report.read_text(encoding="utf-8")

        dashboard = output_dir / "dashboard.html"
        assert dashboard.exists()
        dashboard_html = dashboard.read_text(encoding="utf-8")
        assert 'id="dashboard-root"' in dashboard_html
        for label in [
            "总览",
            "队列探索",
            "疗效对比",
            "PSM",
            "预测模型",
            "数据质量",
            "核心结论摘要",
            "方法说明",
            "搜索当前筛选队列",
            "导出 CSV",
        ]:
            assert label in dashboard_html
        for label in [
            "Overview",
            "Cohort Explorer",
            "Treatment Effects",
            "Prediction",
            "Data Quality",
        ]:
            assert f">{label}<" not in dashboard_html
        for fn_name in [
            "createFilterState",
            "renderFilterChips",
            "renderInsightStrip",
            "renderMetrics",
            "renderSvgCharts",
            "exportCohortCsv",
            "switchTab",
        ]:
            assert f"function {fn_name}" in dashboard_html
        assert "https://" not in dashboard_html
        assert "http://" not in dashboard_html

        match = re.search(
            r'<script id="dashboard-data" type="application/json">(.*?)</script>',
            dashboard_html,
            re.DOTALL,
        )
        assert match, "Dashboard embedded JSON was not found."
        dashboard_data = json.loads(match.group(1))
        assert len(dashboard_data["patients"]) == 102
        sensitive_fields = {
            "source_id",
            "brace_scan_date",
            "followup_date",
            "pre_cobb_raw",
            "post_cobb_raw",
        }
        assert all(
            not sensitive_fields.intersection(row.keys())
            for row in dashboard_data["patients"]
        )
        assert all(
            "source_index" not in row and "row_position" not in row
            for row in dashboard_data["model_predictions"]
        )
        assert results["manifest"]["dashboard"] == "dashboard.html"

        figures = list((output_dir / "figures").glob("*.png"))
        assert len(figures) >= 8
        assert all(path.stat().st_size > 0 for path in figures)


def main() -> None:
    test_cleaning_contract()
    test_sample_data_contract()
    test_pipeline_outputs()
    print("All pipeline smoke tests passed.")


if __name__ == "__main__":
    main()
