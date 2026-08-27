from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd


def _format_number(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, float)):
        return f"{value:.3f}"
    return str(value)


def _table_html(df: pd.DataFrame, max_rows: int = 12) -> str:
    preview = df.head(max_rows).copy()
    for column in preview.columns:
        preview[column] = preview[column].map(_format_number)
    return preview.to_html(index=False, escape=True, border=0, classes="data-table")


def _figure_html(relative_path: str, caption: str) -> str:
    return (
        '<figure class="figure-card">'
        f'<img src="{escape(relative_path)}" alt="{escape(caption)}">'
        f"<figcaption>{escape(caption)}</figcaption>"
        "</figure>"
    )


def build_html_report(
    output_dir: Path | str,
    summary: pd.DataFrame,
    data_quality: pd.DataFrame,
    inference: pd.DataFrame,
    binary_outcome: pd.DataFrame,
    psm_effect: pd.DataFrame,
    psm_sensitivity: pd.DataFrame,
    model_metrics: pd.DataFrame,
    prediction_by_group: pd.DataFrame,
    feature_ablation: pd.DataFrame,
    subgroup_effects: pd.DataFrame,
) -> Path:
    output_dir = Path(output_dir)
    report_path = output_dir / "report.html"

    figures = [
        ("figures/cobb_change_by_group.png", "Cobb change by treatment group"),
        ("figures/improved_5deg_rate.png", "Improved >=5 degrees rate"),
        ("figures/psm_balance_smd.png", "Covariate balance before and after PSM"),
        ("figures/psm_sensitivity.png", "PSM sensitivity across calipers"),
        ("figures/model_predictions.png", "Cross-validated Ridge predictions"),
        ("figures/duration_vs_cobb_change.png", "Intervention duration vs Cobb change"),
    ]
    figure_grid = "\n".join(_figure_html(path, caption) for path, caption in figures if (output_dir / path).exists())

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>脊柱侧弯治疗效果数据分析 Demo Report</title>
  <style>
    body {{
      margin: 0;
      background: #f6f7f9;
      color: #1f2933;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
      line-height: 1.55;
    }}
    header {{
      background: #ffffff;
      border-bottom: 1px solid #d8dee7;
      padding: 28px 44px 22px;
    }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    h2 {{ margin: 34px 0 12px; font-size: 21px; }}
    p {{ margin: 8px 0; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 24px 34px 44px; }}
    .note {{
      background: #fff8e6;
      border: 1px solid #ecd69b;
      border-radius: 8px;
      padding: 12px 14px;
      margin-top: 14px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
      gap: 18px;
      align-items: start;
    }}
    .panel, .figure-card {{
      background: #fff;
      border: 1px solid #d8dee7;
      border-radius: 8px;
      padding: 16px;
      overflow: auto;
    }}
    .data-table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 13px;
    }}
    .data-table th, .data-table td {{
      border-bottom: 1px solid #e5eaf0;
      padding: 7px 8px;
      text-align: left;
      vertical-align: top;
    }}
    .data-table th {{
      background: #eef2f6;
      font-weight: 650;
    }}
    figure {{ margin: 0; }}
    img {{ width: 100%; height: auto; display: block; }}
    figcaption {{ margin-top: 8px; color: #52606d; font-size: 13px; }}
    footer {{
      color: #66788a;
      font-size: 13px;
      padding-top: 18px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>脊柱侧弯治疗效果数据分析 Demo Report</h1>
    <p>Full pipeline: cleaning, EDA, inference, PSM sensitivity, and predictive modeling.</p>
    <div class="note">当前样本量较小，本报告用于方法演示和探索性分析，不应直接作为临床决策结论。</div>
  </header>
  <main>
    <section>
      <h2>Executive Summary</h2>
      <div class="panel">{_table_html(summary, max_rows=10)}</div>
    </section>

    <section>
      <h2>Quality Checks</h2>
      <div class="panel">{_table_html(data_quality, max_rows=20)}</div>
    </section>

    <section>
      <h2>Key Figures</h2>
      <div class="grid">{figure_grid}</div>
    </section>

    <section>
      <h2>Continuous Outcome Inference</h2>
      <div class="panel">{_table_html(inference, max_rows=10)}</div>
    </section>

    <section>
      <h2>Binary Outcome: Improved >=5 Degrees</h2>
      <div class="panel">{_table_html(binary_outcome, max_rows=10)}</div>
    </section>

    <section>
      <h2>PSM Results</h2>
      <div class="grid">
        <div class="panel">{_table_html(psm_effect, max_rows=5)}</div>
        <div class="panel">{_table_html(psm_sensitivity, max_rows=10)}</div>
      </div>
    </section>

    <section>
      <h2>Predictive Modeling</h2>
      <div class="grid">
        <div class="panel">{_table_html(model_metrics, max_rows=10)}</div>
        <div class="panel">{_table_html(prediction_by_group, max_rows=10)}</div>
      </div>
    </section>

    <section>
      <h2>Feature Ablation and Subgroups</h2>
      <div class="grid">
        <div class="panel">{_table_html(feature_ablation, max_rows=12)}</div>
        <div class="panel">{_table_html(subgroup_effects, max_rows=18)}</div>
      </div>
    </section>

    <footer>
      Generated by the Python demo pipeline. Source Excel is read-only; derived outputs are stored under <code>outputs/</code>.
    </footer>
  </main>
</body>
</html>
"""
    report_path.write_text(html, encoding="utf-8")
    return report_path
