from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd


def _fmt(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return html.escape(str(value))


def dataframe_to_html(df: pd.DataFrame, max_rows: int = 12) -> str:
    if df.empty:
        return "<p class='muted'>No rows available.</p>"
    columns = list(df.columns)
    header = "".join(f"<th>{html.escape(str(column))}</th>" for column in columns)
    body_rows = []
    for _, row in df.head(max_rows).iterrows():
        cells = "".join(f"<td>{_fmt(row[column])}</td>" for column in columns)
        body_rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def _section_text(insights: dict[str, object], key: str) -> str:
    sections = insights.get("sections", {})
    if not isinstance(sections, dict):
        return ""
    return html.escape(str(sections.get(key, "")))


def build_html_report(
    *,
    output_dir: Path,
    data_quality: pd.DataFrame,
    treatment_snapshot: pd.DataFrame,
    inference: pd.DataFrame,
    binary_outcome: pd.DataFrame,
    psm_effect: pd.DataFrame,
    psm_balance: pd.DataFrame,
    model_metrics: pd.DataFrame,
    model_feature_importance: pd.DataFrame,
    llm_insights: dict[str, object],
    figure_paths: list[Path],
    manifest: dict[str, object] | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_html = []
    captions = (
        llm_insights.get("sections", {}).get("figure_captions", {})
        if isinstance(llm_insights.get("sections", {}), dict)
        else {}
    )
    for path in figure_paths:
        rel = path.relative_to(output_dir)
        caption = captions.get(path.name, "") if isinstance(captions, dict) else ""
        figure_html.append(
            "<figure>"
            f"<img src='{html.escape(str(rel))}' alt='{html.escape(path.stem)}'>"
            f"<figcaption>{html.escape(str(caption))}</figcaption>"
            "</figure>"
        )

    manifest_block = ""
    if manifest:
        manifest_block = (
            "<details><summary>Run manifest</summary>"
            f"<pre>{html.escape(json.dumps(manifest, ensure_ascii=False, indent=2))}</pre>"
            "</details>"
        )

    html_doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>脊柱侧弯 LLM 增强版数据科学报告</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #20242c; background: #f7f8fa; }}
    header {{ background: #ffffff; border-bottom: 1px solid #dde2ea; padding: 28px 42px; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px 24px 56px; }}
    section {{ margin: 0 0 26px; padding: 24px; background: #ffffff; border: 1px solid #dde2ea; border-radius: 8px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    h2 {{ margin: 0 0 16px; font-size: 21px; }}
    p {{ line-height: 1.68; }}
    .notice {{ color: #7b341e; background: #fff4ec; border: 1px solid #ffd6bd; padding: 12px 14px; border-radius: 6px; }}
    .muted {{ color: #637083; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e7ebf0; padding: 8px 9px; text-align: left; vertical-align: top; }}
    th {{ background: #f0f3f7; }}
    .figures {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 18px; }}
    figure {{ margin: 0; }}
    img {{ width: 100%; height: auto; border: 1px solid #d9dee7; border-radius: 6px; background: white; }}
    figcaption {{ margin-top: 8px; color: #596579; font-size: 13px; line-height: 1.5; }}
    pre {{ white-space: pre-wrap; font-size: 12px; background: #f4f6f8; padding: 14px; border-radius: 6px; }}
  </style>
</head>
<body>
  <header>
    <h1>脊柱侧弯 LLM 增强版 Full Pipeline 数据科学报告</h1>
    <p class="muted">清洗、EDA、统计推断、PSM、预测模型与 LLM 解释层的并行方案输出。</p>
  </header>
  <main>
    <section>
      <h2>医学与方法边界</h2>
      <p class="notice">本报告仅用于探索性数据分析与课程/作品集展示，不构成临床诊断/建议。样本量限制、非随机分组和未观测混杂都可能影响结论；PSM 不能证明因果确定性。</p>
      <p>{_section_text(llm_insights, "executive_summary")}</p>
    </section>
    <section>
      <h2>LLM 数据质控解释</h2>
      <p>{_section_text(llm_insights, "data_quality")}</p>
      {dataframe_to_html(data_quality)}
    </section>
    <section>
      <h2>EDA 与疗效快照</h2>
      {dataframe_to_html(treatment_snapshot)}
    </section>
    <section>
      <h2>统计推断</h2>
      <p>{_section_text(llm_insights, "statistical_interpretation")}</p>
      {dataframe_to_html(inference)}
      <h3>二分类疗效：改善至少 5 度</h3>
      {dataframe_to_html(binary_outcome)}
    </section>
    <section>
      <h2>PSM 倾向评分匹配</h2>
      <p>{_section_text(llm_insights, "psm_interpretation")}</p>
      {dataframe_to_html(psm_effect)}
      {dataframe_to_html(psm_balance)}
    </section>
    <section>
      <h2>预测模型</h2>
      <p>{_section_text(llm_insights, "modeling_interpretation")}</p>
      {dataframe_to_html(model_metrics)}
      <h3>特征重要性</h3>
      {dataframe_to_html(model_feature_importance)}
    </section>
    <section>
      <h2>图表</h2>
      <div class="figures">{''.join(figure_html)}</div>
    </section>
    {manifest_block}
  </main>
</body>
</html>
"""
    report_path = output_dir / "report.html"
    report_path.write_text(html_doc, encoding="utf-8")
    return report_path
