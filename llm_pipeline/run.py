from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .causal_psm import run_psm
from .cleaning import load_clean_data, validate_clean_data
from .config import DEFAULT_DATA_FILE, DEFAULT_OUTPUT_DIR
from .data_ingestion import inspect_workbook
from .eda import data_quality_summary, eda_summary, treatment_effect_snapshot
from .figures import generate_figures
from .inference import run_binary_outcome_analysis, run_inference
from .llm_insights import build_llm_payload, generate_llm_insights
from .modeling import run_modeling
from .report import build_html_report


def _write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


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


def _write_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return path


def run_pipeline(
    data_file: Path | str = DEFAULT_DATA_FILE,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    llm_provider: str = "mock",
    make_figures: bool = True,
) -> dict[str, Any]:
    data_file = Path(data_file)
    output_dir = Path(output_dir)
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    workbook = inspect_workbook(data_file)
    cleaned = load_clean_data(data_file)
    validation_errors = validate_clean_data(cleaned)
    if validation_errors:
        raise ValueError("Cleaned data validation failed: " + " | ".join(validation_errors))
    _write_csv(cleaned, output_dir / "cleaned_data.csv")

    quality = data_quality_summary(cleaned)
    eda = eda_summary(cleaned)
    snapshot = treatment_effect_snapshot(cleaned)
    _write_csv(quality, table_dir / "data_quality.csv")
    _write_csv(eda, table_dir / "eda_summary.csv")
    _write_csv(snapshot, table_dir / "treatment_effect_snapshot.csv")

    inference, adjusted_coefficients = run_inference(cleaned)
    binary_outcome, binary_coefficients = run_binary_outcome_analysis(cleaned)
    _write_csv(inference, table_dir / "inference_results.csv")
    _write_csv(adjusted_coefficients, table_dir / "adjusted_coefficients.csv")
    _write_csv(binary_outcome, table_dir / "binary_outcome_results.csv")
    _write_csv(binary_coefficients, table_dir / "binary_adjusted_coefficients.csv")

    psm = run_psm(cleaned)
    _write_csv(psm["scored"], table_dir / "psm_scored.csv")
    _write_csv(psm["propensity_coefficients"], table_dir / "propensity_coefficients.csv")
    _write_csv(psm["matches"], table_dir / "psm_matches.csv")
    _write_csv(psm["balance"], table_dir / "psm_balance.csv")
    _write_csv(psm["effect"], table_dir / "psm_effect.csv")
    _write_csv(psm["support"], table_dir / "psm_support.csv")
    _write_csv(psm["sensitivity"], table_dir / "psm_sensitivity.csv")
    _write_csv(psm["sensitivity_balance"], table_dir / "psm_sensitivity_balance.csv")

    modeling = run_modeling(cleaned)
    _write_csv(modeling["metrics"], table_dir / "model_metrics.csv")
    _write_csv(modeling["predictions"], table_dir / "model_predictions.csv")
    _write_csv(modeling["coefficients"], table_dir / "model_coefficients.csv")
    _write_csv(modeling["feature_importance"], table_dir / "model_feature_importance.csv")
    _write_csv(modeling["prediction_by_group"], table_dir / "model_prediction_by_group.csv")
    _write_csv(modeling["metadata"], table_dir / "model_metadata.csv")

    figure_paths: list[Path] = []
    if make_figures:
        figure_paths = generate_figures(
            cleaned=cleaned,
            psm_balance=psm["balance"],
            model_predictions=modeling["predictions"],
            figure_dir=figure_dir,
        )

    llm_payload = build_llm_payload(
        data_quality=quality,
        eda_summary=eda,
        treatment_snapshot=snapshot,
        inference=inference,
        binary_outcome=binary_outcome,
        psm_effect=psm["effect"],
        psm_balance=psm["balance"],
        psm_sensitivity=psm["sensitivity"],
        model_metrics=modeling["metrics"],
        model_feature_importance=modeling["feature_importance"],
        row_count=len(cleaned),
    )
    insights = generate_llm_insights(llm_payload, provider=llm_provider)
    _write_json(insights, table_dir / "llm_insights.json")

    manifest = {
        "pipeline": "llm_pipeline",
        "data_file": str(data_file.resolve()),
        "output_dir": str(output_dir.resolve()),
        "llm_provider": llm_provider,
        "privacy_policy": insights["input_policy"],
        "workbook": {
            "sheets": workbook.sheets,
            "rows_by_sheet": workbook.rows_by_sheet,
            "columns_by_sheet": workbook.columns_by_sheet,
        },
        "n_rows_cleaned": int(len(cleaned)),
        "n_rows_outcome_complete": int(cleaned["outcome_observed"].sum()),
        "group_counts": {
            str(k): int(v) for k, v in cleaned["treatment_group"].value_counts().items()
        },
        "tables": sorted(str(path.relative_to(output_dir)) for path in table_dir.glob("*.csv"))
        + ["tables/llm_insights.json"],
        "figures": sorted(str(path.relative_to(output_dir)) for path in figure_paths),
        "report": "report.html",
    }

    report_path = build_html_report(
        output_dir=output_dir,
        data_quality=quality,
        treatment_snapshot=snapshot,
        inference=inference,
        binary_outcome=binary_outcome,
        psm_effect=psm["effect"],
        psm_balance=psm["balance"],
        model_metrics=modeling["metrics"],
        model_feature_importance=modeling["feature_importance"],
        llm_insights=insights,
        figure_paths=figure_paths,
        manifest=manifest,
    )
    _write_json(manifest, output_dir / "run_manifest.json")

    return {
        "cleaned": cleaned,
        "data_quality": quality,
        "eda": eda,
        "treatment_snapshot": snapshot,
        "inference": inference,
        "binary_outcome": binary_outcome,
        "psm": psm,
        "modeling": modeling,
        "llm_insights": insights,
        "report_path": report_path,
        "manifest": manifest,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the LLM-enhanced scoliosis pipeline.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_FILE), help="Input Excel workbook path.")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument(
        "--llm",
        default="mock",
        choices=["mock", "http"],
        help="LLM provider. Use mock for offline deterministic output.",
    )
    parser.add_argument("--no-figures", action="store_true", help="Skip PNG figure generation.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    results = run_pipeline(
        data_file=args.data,
        output_dir=args.out,
        llm_provider=args.llm,
        make_figures=not args.no_figures,
    )
    print(json.dumps(results["manifest"], ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
