from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from llm_pipeline.cleaning import load_clean_data, validate_clean_data
from llm_pipeline.config import SAMPLE_DATA_FILE
from llm_pipeline.llm_insights import normalize_chat_completions_url
from llm_pipeline.run import run_pipeline


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_llm_cleaning_contract() -> None:
    cleaned = load_clean_data(SAMPLE_DATA_FILE)
    assert len(cleaned) == 102
    assert cleaned["treatment_group"].value_counts().to_dict() == {
        "brace_only": 73,
        "brace_training": 29,
    }
    assert int(cleaned["pre_cobb"].isna().sum()) == 0
    assert int(cleaned["post_cobb"].isna().sum()) <= 1
    assert validate_clean_data(cleaned) == []

    complete = cleaned[cleaned["outcome_observed"].eq(1)]
    recomputed = complete["pre_cobb"] - complete["post_cobb"]
    assert recomputed.equals(complete["cobb_change"])
    assert complete["improved_5deg"].astype(int).equals(complete["cobb_change"].ge(5).astype(int))


def test_http_llm_url_normalization() -> None:
    assert (
        normalize_chat_completions_url("https://dashscope.aliyuncs.com/compatible-mode/v1")
        == "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    )
    assert (
        normalize_chat_completions_url("https://example.com/v1/chat/completions")
        == "https://example.com/v1/chat/completions"
    )


def test_llm_pipeline_outputs_and_privacy() -> None:
    source_hash = _sha256(SAMPLE_DATA_FILE)
    original_src_files = sorted(path.relative_to(PROJECT_ROOT) for path in (PROJECT_ROOT / "src").glob("*.py"))

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "outputs_llm"
        results = run_pipeline(SAMPLE_DATA_FILE, output_dir=out, llm_provider="mock", make_figures=True)
        assert results["manifest"]["n_rows_cleaned"] == 102
        assert results["manifest"]["n_rows_outcome_complete"] >= 101
        assert (out / "cleaned_data.csv").exists()
        assert (out / "report.html").exists()
        assert (out / "run_manifest.json").exists()

        for name in [
            "data_quality.csv",
            "eda_summary.csv",
            "inference_results.csv",
            "psm_balance.csv",
            "psm_effect.csv",
            "psm_matches.csv",
            "psm_sensitivity.csv",
            "model_metrics.csv",
            "model_predictions.csv",
            "model_feature_importance.csv",
        ]:
            path = out / "tables" / name
            assert path.exists(), f"Missing output: {path}"
            assert not pd.read_csv(path).empty, f"Empty output: {path}"

        psm_balance = pd.read_csv(out / "tables" / "psm_balance.csv")
        assert {"before_matching", "after_matching"}.issubset(set(psm_balance["stage"]))
        psm_sensitivity = pd.read_csv(out / "tables" / "psm_sensitivity.csv")
        assert len(psm_sensitivity) >= 3
        assert int(pd.read_csv(out / "tables" / "psm_matches.csv").shape[0]) > 0

        metrics = pd.read_csv(out / "tables" / "model_metrics.csv")
        assert {"mae", "rmse", "r2"}.issubset(metrics.columns)
        assert "ridge" in set(metrics["model"])

        predictions = pd.read_csv(out / "tables" / "model_predictions.csv")
        leaked_columns = {
            "source_id",
            "brace_scan_date",
            "followup_date",
            "pre_cobb_raw",
            "post_cobb_raw",
            "source_index",
            "row_position",
        }
        assert not leaked_columns.intersection(predictions.columns)

        insights = json.loads((out / "tables" / "llm_insights.json").read_text(encoding="utf-8"))
        insight_text = json.dumps(insights, ensure_ascii=False)
        for phrase in ["探索性", "非临床诊断", "样本量限制"]:
            assert phrase in insight_text
        assert insights["input_policy"]["row_level_records_sent"] is False

        report_html = (out / "report.html").read_text(encoding="utf-8")
        assert "不构成临床诊断/建议" in report_html
        assert "样本量限制" in report_html
        assert len(list((out / "figures").glob("*.png"))) >= 5

    assert _sha256(SAMPLE_DATA_FILE) == source_hash
    current_src_files = sorted(path.relative_to(PROJECT_ROOT) for path in (PROJECT_ROOT / "src").glob("*.py"))
    assert current_src_files == original_src_files


def main() -> None:
    test_llm_cleaning_contract()
    test_http_llm_url_normalization()
    test_llm_pipeline_outputs_and_privacy()
    print("All LLM pipeline smoke tests passed.")


if __name__ == "__main__":
    main()
