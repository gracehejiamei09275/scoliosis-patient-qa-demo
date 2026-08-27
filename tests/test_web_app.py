from __future__ import annotations

import json
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from llm_pipeline.config import SAMPLE_DATA_FILE
from llm_pipeline.web_app import build_server


def _post_json(url: str, payload: dict[str, object]) -> tuple[int, dict[str, object]]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _get_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=20) as response:
        return response.read().decode("utf-8")


class ServerFixture:
    def __init__(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.server = build_server(
            host="127.0.0.1",
            port=0,
            data_file=SAMPLE_DATA_FILE,
            output_dir=Path(self.tmp.name) / "outputs_llm",
            default_llm="mock",
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> "ServerFixture":
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.tmp.cleanup()

    @property
    def base_url(self) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}"


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


def test_index_page_contains_ui_and_no_secret_markers() -> None:
    with ServerFixture() as fixture:
        html = _get_text(f"{fixture.base_url}/")
        for text in ["脊柱侧弯疗效问答", "患者变量", "问题", "提交问答", "本系统仅用于探索性数据分析展示"]:
            assert text in html
        for forbidden in [".env", "LLM_API_KEY", "sk-", "API key"]:
            assert forbidden not in html


def test_health_endpoint() -> None:
    with ServerFixture() as fixture:
        data = json.loads(_get_text(f"{fixture.base_url}/api/health"))
        assert data["status"] == "ok"
        assert data["default_llm"] == "mock"
        assert data["model_artifact_exists"] is True


def test_qa_prediction_mock_response() -> None:
    with ServerFixture() as fixture:
        status, data = _post_json(
            f"{fixture.base_url}/api/qa",
            {
                "question": "我13岁，支具加训练8个月，大概可能改善多少？",
                "patient": _sample_patient(),
                "llm_provider": "mock",
            },
        )
        assert status == 200
        assert data["route"]["question_type"] == "prediction"
        assert "探索性" in data["answer"]
        assert "不是临床诊断/建议" in data["answer"]
        assert data["prediction"]["predicted_cobb_change"] is not None
        assert data["prediction"]["model_metrics"]["rmse"] > 0
        assert data["privacy_policy"]["row_level_records_sent"] is False


def test_brace_only_training_hours_are_zero() -> None:
    patient = _sample_patient()
    patient["treatment_group"] = "brace_only"
    patient["weekly_training_hours"] = 9
    with ServerFixture() as fixture:
        status, data = _post_json(
            f"{fixture.base_url}/api/qa",
            {
                "question": "我13岁，纯支具8个月，大概可能改善多少？",
                "patient": patient,
                "llm_provider": "mock",
            },
        )
        assert status == 200
        assert data["prediction"]["normalized_patient"]["weekly_training_hours"] == 0.0
        assert data["prediction"]["features"]["training_exposure_hours"] == 0.0


def test_missing_required_field_returns_400() -> None:
    patient = _sample_patient()
    patient.pop("pre_cobb")
    with ServerFixture() as fixture:
        status, data = _post_json(
            f"{fixture.base_url}/api/qa",
            {
                "question": "我13岁，支具加训练8个月，大概可能改善多少？",
                "patient": patient,
                "llm_provider": "mock",
            },
        )
        assert status == 400
        assert "Missing required patient field" in data["error"]


def test_unsafe_question_refuses_without_prediction() -> None:
    with ServerFixture() as fixture:
        status, data = _post_json(
            f"{fixture.base_url}/api/qa",
            {
                "question": "我应该选纯支具还是支具加训练？",
                "patient": _sample_patient(),
                "llm_provider": "mock",
            },
        )
        assert status == 200
        assert data["route"]["question_type"] == "unsafe_medical_advice"
        assert "不能给出具体医疗建议" in data["answer"]
        assert data["prediction"] is None


def main() -> None:
    test_index_page_contains_ui_and_no_secret_markers()
    test_health_endpoint()
    test_qa_prediction_mock_response()
    test_brace_only_training_hours_are_zero()
    test_missing_required_field_returns_400()
    test_unsafe_question_refuses_without_prediction()
    print("All web app smoke tests passed.")


if __name__ == "__main__":
    main()
