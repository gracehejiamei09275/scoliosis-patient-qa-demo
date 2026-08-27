from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import DATA_FILE, OUTPUT_DIR
from .dashboard import build_dashboard
from .data_cleaning import load_clean_data, outcome_complete, validate_clean_data
from .extended_analysis import (
    data_quality_summary,
    executive_summary_table,
    subgroup_effects,
    variable_codebook,
)
from .modeling import run_modeling
from .psm import run_psm
from .reporting import build_html_report
from .simple_plots import (
    save_binary_rate_plot,
    save_boxplot_by_group,
    save_histogram_by_group,
    save_prediction_scatter,
    save_psm_sensitivity_plot,
    save_scatter_by_group,
    save_smd_balance_plot,
)
from .statistics import balance_table, eda_summary, run_binary_outcome_analysis, run_inference


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def generate_figures(
    cleaned: pd.DataFrame,
    binary_outcome: pd.DataFrame,
    psm_balance: pd.DataFrame,
    psm_sensitivity: pd.DataFrame,
    predictions: pd.DataFrame,
    figure_dir: Path,
) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    complete = outcome_complete(cleaned)
    figure_paths = [
        figure_dir / "age_distribution.png",
        figure_dir / "pre_cobb_distribution.png",
        figure_dir / "training_hours_distribution.png",
        figure_dir / "cobb_change_by_group.png",
        figure_dir / "duration_vs_cobb_change.png",
        figure_dir / "improved_5deg_rate.png",
        figure_dir / "psm_balance_smd.png",
        figure_dir / "psm_sensitivity.png",
        figure_dir / "model_predictions.png",
    ]

    save_histogram_by_group(
        cleaned,
        "age",
        figure_paths[0],
        "Age distribution by treatment group",
        "Age",
    )
    save_histogram_by_group(
        cleaned,
        "pre_cobb",
        figure_paths[1],
        "Baseline Cobb angle distribution",
        "Baseline Cobb angle",
    )
    save_histogram_by_group(
        cleaned,
        "weekly_training_hours",
        figure_paths[2],
        "Weekly training hours distribution",
        "Weekly training hours",
        bins=6,
    )
    save_boxplot_by_group(
        complete,
        "cobb_change",
        figure_paths[3],
        "Cobb change by treatment group",
        "Cobb change",
    )
    save_scatter_by_group(
        complete,
        "intervention_months",
        "cobb_change",
        figure_paths[4],
        "Intervention duration vs Cobb change",
        "Intervention months",
        "Cobb change",
    )
    save_binary_rate_plot(binary_outcome, figure_paths[5])
    save_smd_balance_plot(psm_balance, figure_paths[6])
    save_psm_sensitivity_plot(psm_sensitivity, figure_paths[7])
    save_prediction_scatter(predictions, figure_paths[8])
    return figure_paths


def run_pipeline(
    data_file: Path | str = DATA_FILE,
    output_dir: Path | str = OUTPUT_DIR,
    make_figures: bool = True,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    cleaned = load_clean_data(Path(data_file))
    validation_errors = validate_clean_data(cleaned)
    if validation_errors:
        raise ValueError("Cleaned data validation failed: " + " | ".join(validation_errors))

    _write_csv(cleaned, output_dir / "cleaned_data.csv")

    quality = data_quality_summary(cleaned)
    codebook = variable_codebook()
    _write_csv(quality, table_dir / "data_quality_summary.csv")
    _write_csv(codebook, table_dir / "data_dictionary.csv")

    eda = eda_summary(cleaned)
    _write_csv(eda, table_dir / "eda_summary.csv")

    baseline_balance = balance_table(cleaned, stage="before_matching")
    _write_csv(baseline_balance, table_dir / "baseline_balance.csv")

    inference, ols_coefficients = run_inference(cleaned)
    _write_csv(inference, table_dir / "inference_results.csv")
    _write_csv(ols_coefficients, table_dir / "adjusted_ols_coefficients.csv")

    binary_outcome, binary_coefficients = run_binary_outcome_analysis(cleaned)
    _write_csv(binary_outcome, table_dir / "binary_outcome_results.csv")
    _write_csv(binary_coefficients, table_dir / "binary_adjusted_coefficients.csv")

    psm_results = run_psm(cleaned)
    _write_csv(psm_results["balance"], table_dir / "psm_balance.csv")
    _write_csv(psm_results["effect"], table_dir / "psm_effect.csv")
    _write_csv(psm_results["matches"], table_dir / "psm_matches.csv")
    _write_csv(psm_results["support"], table_dir / "psm_support.csv")
    _write_csv(psm_results["propensity_coefficients"], table_dir / "propensity_coefficients.csv")
    _write_csv(psm_results["sensitivity"], table_dir / "psm_sensitivity.csv")
    _write_csv(psm_results["sensitivity_balance"], table_dir / "psm_sensitivity_balance.csv")

    modeling_results = run_modeling(cleaned)
    _write_csv(modeling_results["metrics"], table_dir / "model_metrics.csv")
    _write_csv(modeling_results["coefficients"], table_dir / "model_coefficients.csv")
    _write_csv(modeling_results["predictions"], table_dir / "model_predictions.csv")
    _write_csv(modeling_results["prediction_by_group"], table_dir / "model_prediction_by_group.csv")
    _write_csv(modeling_results["feature_ablation"], table_dir / "model_feature_ablation.csv")
    _write_csv(modeling_results["metadata"], table_dir / "model_metadata.csv")

    subgroups = subgroup_effects(cleaned)
    _write_csv(subgroups, table_dir / "subgroup_effects.csv")

    summary = executive_summary_table(
        inference=inference,
        binary_outcome=binary_outcome,
        psm_effect=psm_results["effect"],
        model_metrics=modeling_results["metrics"],
    )
    _write_csv(summary, table_dir / "executive_summary.csv")

    figure_paths: list[Path] = []
    if make_figures:
        figure_paths = generate_figures(
            cleaned=cleaned,
            binary_outcome=binary_outcome,
            psm_balance=psm_results["balance"],
            psm_sensitivity=psm_results["sensitivity"],
            predictions=modeling_results["predictions"],
            figure_dir=figure_dir,
        )

    report_path = build_html_report(
        output_dir=output_dir,
        summary=summary,
        data_quality=quality,
        inference=inference,
        binary_outcome=binary_outcome,
        psm_effect=psm_results["effect"],
        psm_sensitivity=psm_results["sensitivity"],
        model_metrics=modeling_results["metrics"],
        prediction_by_group=modeling_results["prediction_by_group"],
        feature_ablation=modeling_results["feature_ablation"],
        subgroup_effects=subgroups,
    )
    dashboard_path = build_dashboard(output_dir)

    manifest = {
        "data_file": str(Path(data_file).resolve()),
        "output_dir": str(output_dir.resolve()),
        "n_rows_cleaned": int(len(cleaned)),
        "n_rows_outcome_complete": int(cleaned["outcome_observed"].sum()),
        "group_counts": {
            str(k): int(v) for k, v in cleaned["treatment_group"].value_counts().items()
        },
        "tables": sorted(str(path.relative_to(output_dir)) for path in table_dir.glob("*.csv")),
        "figures": sorted(str(path.relative_to(output_dir)) for path in figure_paths),
        "report": str(report_path.relative_to(output_dir)),
        "dashboard": str(dashboard_path.relative_to(output_dir)),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "cleaned": cleaned,
        "eda": eda,
        "inference": inference,
        "binary_outcome": binary_outcome,
        "psm": psm_results,
        "modeling": modeling_results,
        "summary": summary,
        "report_path": report_path,
        "dashboard_path": dashboard_path,
        "manifest": manifest,
    }


def main() -> None:
    results = run_pipeline()
    print(json.dumps(results["manifest"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
