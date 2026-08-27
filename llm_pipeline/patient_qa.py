from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .answer_composer import compose_answer, load_project_context
from .config import DEFAULT_DATA_FILE, DEFAULT_OUTPUT_DIR
from .model_service import ensure_model_artifact
from .qa_router import route_question


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


def parse_patient_json(raw: str | None) -> dict[str, Any] | None:
    if raw is None or raw.strip() == "":
        return None
    try:
        patient = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--patient-json must be valid JSON: {exc}") from exc
    if not isinstance(patient, dict):
        raise ValueError("--patient-json must decode to a JSON object.")
    return patient


def save_qa_log(output_dir: Path | str, payload: dict[str, Any]) -> Path:
    log_dir = Path(output_dir) / "qa_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = log_dir / f"qa_{timestamp}.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return path


def run_patient_qa(
    *,
    question: str,
    patient: dict[str, Any] | None,
    data_file: Path | str = DEFAULT_DATA_FILE,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    llm_provider: str = "mock",
) -> dict[str, Any]:
    artifact = ensure_model_artifact(data_file=data_file, output_dir=output_dir)
    context = load_project_context(data_file=data_file, output_dir=output_dir)
    route = route_question(question, has_patient_payload=patient is not None)
    payload = compose_answer(
        question=question,
        route=route,
        context=context,
        artifact=artifact,
        patient=patient,
        llm_provider=llm_provider,
    )
    log_path = save_qa_log(output_dir, payload)
    payload["qa_log"] = str(log_path.resolve())
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run patient-friendly ML + LLM QA.")
    parser.add_argument("--question", required=True, help="Patient or demo question.")
    parser.add_argument("--patient-json", default=None, help="Structured patient variables as JSON.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_FILE), help="Input Excel workbook path.")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument(
        "--llm",
        default="mock",
        choices=["mock", "http"],
        help="LLM provider. Mock is deterministic and offline.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    patient = parse_patient_json(args.patient_json)
    payload = run_patient_qa(
        question=args.question,
        patient=patient,
        data_file=args.data,
        output_dir=args.out,
        llm_provider=args.llm,
    )
    print(payload["answer"])
    summary = {
        "route": payload["route"],
        "prediction": payload.get("prediction"),
        "qa_log": payload["qa_log"],
        "privacy_policy": payload["privacy_policy"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
