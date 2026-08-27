from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from llm_pipeline.config import SAMPLE_DATA_FILE
from llm_pipeline.model_service import (
    build_patient_features,
    ensure_model_artifact,
    predict_patient,
)
from llm_pipeline.patient_qa import run_patient_qa
from llm_pipeline.qa_router import QuestionType, route_question


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sample_patient() -> dict[str, object]:
    return {
        "age": 13,
        "sex": "female",
        "bone_age_stage": 4,
        "pre_cobb": 28,
        "intervention_months": 8,
        "treatment_group": "brace_training",
        "weekly_training_hours": 3,
    }


def test_question_routing() -> None:
    assert route_question("Cobb 改善量是什么").question_type == QuestionType.EDUCATION
    assert (
        route_question("我13岁，支具加训练8个月，大概可能改善多少？", has_patient_payload=True).question_type
        == QuestionType.PREDICTION
    )
    assert route_question("我应该选哪种治疗？").question_type == QuestionType.UNSAFE_MEDICAL_ADVICE


def test_model_artifact_and_prediction_stability() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        artifact = ensure_model_artifact(SAMPLE_DATA_FILE, tmp)
        artifact_path = Path(tmp) / "model_artifact.json"
        assert artifact_path.exists()
        for key in [
            "feature_names",
            "feature_mean",
            "feature_std",
            "coefficients",
            "alpha",
            "model_metrics",
        ]:
            assert key in artifact
        for metric in ["mae", "rmse", "r2"]:
            assert metric in artifact["model_metrics"]

        first = predict_patient(_sample_patient(), artifact)
        second = predict_patient(_sample_patient(), artifact)
        assert first["predicted_cobb_change"] == second["predicted_cobb_change"]
        assert "prediction_interval_note" in first
        assert isinstance(first["predicted_improved_5deg"], bool)
        assert first["feature_contributions"]


def test_patient_validation_errors_are_explicit() -> None:
    patient = _sample_patient()
    patient.pop("pre_cobb")
    try:
        build_patient_features(patient)
    except ValueError as exc:
        assert "Missing required patient field" in str(exc)
    else:
        raise AssertionError("Expected missing field validation error.")

    patient = _sample_patient()
    patient.pop("weekly_training_hours")
    try:
        build_patient_features(patient)
    except ValueError as exc:
        assert "weekly_training_hours" in str(exc)
    else:
        raise AssertionError("Expected training-hours validation error.")


def test_patient_qa_prediction_privacy_and_safety() -> None:
    source_hash = _sha256(SAMPLE_DATA_FILE)
    original_src_files = sorted(path.relative_to(PROJECT_ROOT) for path in (PROJECT_ROOT / "src").glob("*.py"))
    with tempfile.TemporaryDirectory() as tmp:
        payload = run_patient_qa(
            question="我13岁，支具加训练8个月，大概可能改善多少？",
            patient=_sample_patient(),
            data_file=SAMPLE_DATA_FILE,
            output_dir=tmp,
            llm_provider="mock",
        )
        answer = payload["answer"]
        for phrase in ["探索性", "不是临床诊断/建议", "样本量限制", "非随机分组", "需咨询医生"]:
            assert phrase in answer
        assert payload["route"]["question_type"] == "prediction"
        assert payload["prediction"]["model_metrics"]["rmse"] > 0
        assert Path(payload["qa_log"]).exists()

        prompt_preview = payload["prompt_audit"]["prompt_preview"]
        for forbidden in [
            "source_id",
            "brace_scan_date",
            "followup_date",
            "pre_cobb_raw",
            "post_cobb_raw",
        ]:
            assert forbidden not in prompt_preview

        log_payload = json.loads(Path(payload["qa_log"]).read_text(encoding="utf-8"))
        assert log_payload["privacy_policy"]["row_level_records_sent"] is False

    assert _sha256(SAMPLE_DATA_FILE) == source_hash
    current_src_files = sorted(path.relative_to(PROJECT_ROOT) for path in (PROJECT_ROOT / "src").glob("*.py"))
    assert current_src_files == original_src_files


def test_patient_qa_unsafe_refusal() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        payload = run_patient_qa(
            question="我应该选纯支具还是支具加训练？",
            patient=_sample_patient(),
            data_file=SAMPLE_DATA_FILE,
            output_dir=tmp,
            llm_provider="mock",
        )
        assert payload["route"]["question_type"] == "unsafe_medical_advice"
        assert "不能给出具体医疗建议" in payload["answer"]
        assert "需咨询医生" in payload["answer"]


def main() -> None:
    test_question_routing()
    test_model_artifact_and_prediction_stability()
    test_patient_validation_errors_are_explicit()
    test_patient_qa_prediction_privacy_and_safety()
    test_patient_qa_unsafe_refusal()
    print("All patient QA smoke tests passed.")


if __name__ == "__main__":
    main()
