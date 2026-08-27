from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SAFE_PATIENT_COLUMNS = [
    "treatment_group",
    "sex",
    "age",
    "bone_age_stage",
    "intervention_months",
    "pre_cobb",
    "post_cobb",
    "cobb_change",
    "cobb_change_pct",
    "improved_5deg",
    "training_exposure_hours",
    "outcome_observed",
]


def _clean_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    safe = df.astype(object).where(pd.notna(df), None)
    return [
        {str(key): _clean_value(value) for key, value in row.items()}
        for row in safe.to_dict(orient="records")
    ]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def _range(values: pd.Series) -> dict[str, float | None]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {"min": None, "max": None}
    return {"min": float(numeric.min()), "max": float(numeric.max())}


def _build_payload(output_dir: Path) -> dict[str, Any]:
    table_dir = output_dir / "tables"
    cleaned = _read_csv(output_dir / "cleaned_data.csv")
    safe_patients = cleaned[SAFE_PATIENT_COLUMNS].copy()
    model_predictions = _read_csv(table_dir / "model_predictions.csv")
    safe_prediction_columns = [
        col
        for col in [
            "model",
            "alpha",
            "fold",
            "treatment_binary",
            "actual_cobb_change",
            "predicted_cobb_change",
        ]
        if col in model_predictions.columns
    ]

    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "patients": _records(safe_patients),
        "ranges": {
            "age": _range(safe_patients["age"]),
            "pre_cobb": _range(safe_patients["pre_cobb"]),
            "intervention_months": _range(safe_patients["intervention_months"]),
        },
        "executive_summary": _records(_read_csv(table_dir / "executive_summary.csv")),
        "inference": _records(_read_csv(table_dir / "inference_results.csv")),
        "binary_outcome": _records(_read_csv(table_dir / "binary_outcome_results.csv")),
        "psm_balance": _records(_read_csv(table_dir / "psm_balance.csv")),
        "psm_sensitivity": _records(_read_csv(table_dir / "psm_sensitivity.csv")),
        "model_metrics": _records(_read_csv(table_dir / "model_metrics.csv")),
        "model_predictions": _records(model_predictions[safe_prediction_columns]),
        "model_feature_ablation": _records(_read_csv(table_dir / "model_feature_ablation.csv")),
        "data_quality": _records(_read_csv(table_dir / "data_quality_summary.csv")),
    }
    return payload


def build_dashboard(output_dir: Path | str) -> Path:
    output_dir = Path(output_dir)
    payload = _build_payload(output_dir)
    data_json = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    data_json = data_json.replace("</", "<\\/")
    html = DASHBOARD_TEMPLATE.replace("__DASHBOARD_DATA__", data_json)
    dashboard_path = output_dir / "dashboard.html"
    dashboard_path.write_text(html, encoding="utf-8")
    return dashboard_path


DASHBOARD_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>脊柱侧弯治疗效果可视化分析</title>
  <style>
    :root {
      --bg: #f7f8fa;
      --surface: #ffffff;
      --line: #dce3ec;
      --line-soft: #edf1f5;
      --text: #1f2a37;
      --muted: #667483;
      --blue: #28679f;
      --red: #bf5652;
      --teal: #2a9d8f;
      --amber: #b7791f;
      --slate: #637381;
      --focus: #245c8d;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
      line-height: 1.45;
    }
    header {
      background: #ffffff;
      border-bottom: 1px solid var(--line);
      padding: 16px 26px 14px;
    }
    .topbar {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 18px;
      max-width: 1440px;
      margin: 0 auto;
    }
    h1 {
      margin: 0 0 6px;
      font-size: 23px;
      letter-spacing: 0;
      font-weight: 700;
    }
    .subtitle {
      color: var(--muted);
      font-size: 14px;
      margin: 0;
    }
    .notice {
      border: 1px solid #e6d6a6;
      background: #fffaf0;
      color: #6f5419;
      border-radius: 8px;
      padding: 9px 12px;
      max-width: 500px;
      font-size: 13px;
    }
    main {
      max-width: 1440px;
      margin: 0 auto;
      padding: 16px 22px 34px;
    }
    .layout {
      display: grid;
      grid-template-columns: 270px minmax(0, 1fr);
      gap: 16px;
      align-items: start;
    }
    aside {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 13px;
      position: sticky;
      top: 12px;
    }
    .filter-title {
      font-size: 15px;
      font-weight: 700;
      margin: 0 0 10px;
    }
    .filter-group {
      border-top: 1px solid var(--line-soft);
      padding: 12px 0;
    }
    .filter-group:first-of-type { border-top: 0; padding-top: 0; }
    label {
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
      margin-bottom: 6px;
    }
    select, input {
      width: 100%;
      height: 34px;
      border: 1px solid #c7d0dc;
      border-radius: 5px;
      background: #fff;
      color: var(--text);
      font-size: 14px;
      padding: 6px 8px;
    }
    select:focus, input:focus, button:focus {
      outline: 2px solid rgba(37, 92, 153, 0.25);
      border-color: var(--focus);
    }
    .range-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .reset-button {
      width: 100%;
      border: 1px solid #b7c2d0;
      background: #f0f4f8;
      color: #1f425f;
      height: 34px;
      border-radius: 5px;
      font-weight: 650;
      cursor: pointer;
    }
    .filter-chips {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 0 0 12px;
      min-height: 32px;
      align-items: center;
    }
    .chip {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border: 1px solid #c9d5e3;
      background: #ffffff;
      color: #33475b;
      border-radius: 999px;
      padding: 5px 10px;
      font-size: 12px;
      line-height: 1.2;
    }
    .chip.muted {
      color: var(--muted);
      background: #f5f7fa;
    }
    .chip button {
      border: 0;
      background: transparent;
      color: #245c8d;
      font-weight: 700;
      padding: 0;
      cursor: pointer;
    }
    .tabs {
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      margin-bottom: 12px;
    }
    .tab-button {
      border: 1px solid var(--line);
      background: #fff;
      color: #314255;
      border-radius: 5px;
      padding: 8px 11px;
      font-weight: 650;
      cursor: pointer;
    }
    .tab-button.active {
      background: #245c8d;
      color: #fff;
      border-color: #245c8d;
    }
    .tab-panel { display: none; }
    .tab-panel.active { display: block; }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(6, minmax(130px, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }
    .metric {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 11px 12px;
      min-height: 76px;
    }
    .metric span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 8px;
    }
    .metric strong {
      font-size: 22px;
      letter-spacing: 0;
    }
    .insight-strip {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }
    .insight-card {
      background: #ffffff;
      border: 1px solid var(--line);
      border-left: 4px solid #28679f;
      border-radius: 6px;
      padding: 11px 12px;
      min-height: 92px;
    }
    .insight-card:nth-child(2) { border-left-color: #2a9d8f; }
    .insight-card:nth-child(3) { border-left-color: #b7791f; }
    .insight-card:nth-child(4) { border-left-color: #637381; }
    .insight-card span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 5px;
    }
    .insight-card strong {
      display: block;
      font-size: 21px;
      margin-bottom: 5px;
    }
    .insight-card small {
      display: block;
      color: #4f5f70;
      font-size: 12px;
      line-height: 1.35;
    }
    .method-panel {
      background: #ffffff;
      border: 1px solid var(--line);
      border-radius: 6px;
      margin-bottom: 14px;
      padding: 0;
    }
    .method-panel summary {
      cursor: pointer;
      padding: 11px 13px;
      color: #263a4d;
      font-weight: 700;
    }
    .method-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      border-top: 1px solid var(--line-soft);
      padding: 12px 13px 13px;
    }
    .method-grid div {
      border: 1px solid var(--line-soft);
      border-radius: 5px;
      padding: 9px 10px;
      background: #fbfcfd;
      font-size: 13px;
    }
    .method-grid strong {
      display: block;
      margin-bottom: 4px;
      color: #2e4053;
    }
    .section-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 14px;
    }
    .section-grid.three {
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }
    .panel {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 14px;
      min-width: 0;
    }
    .panel h2 {
      font-size: 16px;
      margin: 0 0 4px;
    }
    .panel .caption {
      color: var(--muted);
      font-size: 12px;
      margin: 0 0 10px;
    }
    .chart {
      width: 100%;
      min-height: 330px;
    }
    .chart svg {
      width: 100%;
      height: 330px;
      display: block;
    }
    .analysis-note {
      background: #eef6f5;
      border: 1px solid #b8dcd8;
      color: #285955;
      border-radius: 6px;
      padding: 10px 12px;
      margin-bottom: 12px;
      font-size: 13px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      border-bottom: 1px solid var(--line-soft);
      padding: 8px 7px;
      text-align: left;
      vertical-align: top;
      white-space: nowrap;
    }
    th {
      background: #eef2f6;
      color: #34475a;
      font-weight: 700;
      cursor: pointer;
      user-select: none;
    }
    .table-wrap {
      overflow: auto;
      max-height: 440px;
      border: 1px solid var(--line-soft);
      border-radius: 6px;
    }
    .table-actions {
      display: grid;
      grid-template-columns: minmax(180px, 1fr) auto auto auto auto auto;
      gap: 8px;
      align-items: center;
      margin: 10px 0;
    }
    .table-actions button,
    .table-actions select {
      height: 34px;
      border: 1px solid #b7c2d0;
      border-radius: 5px;
      background: #ffffff;
      color: #263a4d;
      font-size: 13px;
      padding: 0 10px;
      cursor: pointer;
      white-space: nowrap;
    }
    .table-actions button.primary {
      background: #245c8d;
      color: #ffffff;
      border-color: #245c8d;
      font-weight: 650;
    }
    .table-actions button:disabled {
      color: #95a1ae;
      background: #f4f6f8;
      cursor: not-allowed;
    }
    .table-page-info {
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }
    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      color: var(--muted);
      font-size: 12px;
      margin-top: 8px;
    }
    .legend-item {
      display: inline-flex;
      align-items: center;
      gap: 5px;
    }
    .swatch {
      width: 12px;
      height: 12px;
      border-radius: 2px;
      display: inline-block;
    }
    .empty {
      color: var(--muted);
      padding: 32px 12px;
      text-align: center;
      border: 1px dashed var(--line);
      border-radius: 8px;
    }
    .summary-list {
      display: grid;
      gap: 8px;
    }
    .summary-row {
      display: grid;
      grid-template-columns: 230px 100px 1fr;
      gap: 12px;
      align-items: baseline;
      border-bottom: 1px solid var(--line-soft);
      padding: 8px 0;
    }
    .summary-row:last-child { border-bottom: 0; }
    .summary-row strong { font-size: 15px; }
    .summary-row small { color: var(--muted); }
    @media (max-width: 1050px) {
      .layout { grid-template-columns: 1fr; }
      aside { position: static; }
      .metric-grid { grid-template-columns: repeat(3, minmax(130px, 1fr)); }
      .insight-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .section-grid, .section-grid.three { grid-template-columns: 1fr; }
      .method-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 640px) {
      header { padding: 16px; }
      main { padding: 14px; }
      .topbar { display: block; }
      .notice { margin-top: 12px; }
      .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .insight-strip { grid-template-columns: 1fr; }
      .method-grid { grid-template-columns: 1fr; }
      .table-actions { grid-template-columns: 1fr 1fr; }
      .table-actions input,
      .table-actions select,
      .table-actions button,
      .table-page-info { width: 100%; }
      .tabs { overflow-x: auto; flex-wrap: nowrap; padding-bottom: 4px; }
      .tab-button { flex: 0 0 auto; }
      .summary-row { grid-template-columns: 1fr; gap: 2px; }
      .chart svg { height: 300px; }
    }
    @media print {
      body { background: #ffffff; }
      header, main { padding: 0; }
      aside, .tabs, .table-actions, .method-panel { display: none !important; }
      .layout { display: block; }
      .tab-panel { display: block !important; page-break-inside: avoid; }
      .metric-grid, .insight-strip, .section-grid { break-inside: avoid; }
      .panel, .metric, .insight-card { box-shadow: none; border-color: #ccd4dd; }
      .table-wrap { max-height: none; overflow: visible; }
    }
  </style>
</head>
<body>
  <script id="dashboard-data" type="application/json">__DASHBOARD_DATA__</script>
  <div id="dashboard-root">
    <header>
      <div class="topbar">
        <div>
          <h1>脊柱侧弯治疗效果可视化分析</h1>
          <p class="subtitle" id="dashboard-subtitle">完整数据科学流程的交互式展示</p>
        </div>
        <div class="notice">探索性数据分析界面；当前样本量较小，统计和模型结果不应直接作为临床决策结论。</div>
      </div>
    </header>
    <main>
      <div class="layout">
        <aside aria-label="筛选条件">
          <p class="filter-title">筛选条件</p>
          <div class="filter-group">
            <label for="filter-treatment">治疗方案</label>
            <select id="filter-treatment">
              <option value="all">全部方案</option>
              <option value="brace_only">纯支具</option>
              <option value="brace_training">支具+训练</option>
            </select>
          </div>
          <div class="filter-group">
            <label for="filter-sex">性别</label>
            <select id="filter-sex">
              <option value="all">全部</option>
              <option value="female">女</option>
              <option value="male">男</option>
            </select>
          </div>
          <div class="filter-group">
            <label>年龄范围</label>
            <div class="range-row">
              <input id="filter-age-min" type="number" step="1" aria-label="最小年龄">
              <input id="filter-age-max" type="number" step="1" aria-label="最大年龄">
            </div>
          </div>
          <div class="filter-group">
            <label>干预前 Cobb 角范围（度）</label>
            <div class="range-row">
              <input id="filter-cobb-min" type="number" step="0.5" aria-label="最小干预前 Cobb 角">
              <input id="filter-cobb-max" type="number" step="0.5" aria-label="最大干预前 Cobb 角">
            </div>
          </div>
          <div class="filter-group">
            <label>干预周期（月）</label>
            <div class="range-row">
              <input id="filter-duration-min" type="number" step="1" aria-label="最短干预周期">
              <input id="filter-duration-max" type="number" step="1" aria-label="最长干预周期">
            </div>
          </div>
          <div class="filter-group">
            <label for="filter-improved">改善是否达到 5 度</label>
            <select id="filter-improved">
              <option value="all">全部疗效</option>
              <option value="1">达到</option>
              <option value="0">未达到</option>
              <option value="missing">疗效缺失</option>
            </select>
          </div>
          <button class="reset-button" id="reset-filters" type="button">重置筛选</button>
        </aside>
        <section>
          <nav class="tabs" aria-label="可视化页面导航">
            <button class="tab-button active" data-tab="overview" type="button">总览</button>
            <button class="tab-button" data-tab="cohort" type="button">队列探索</button>
            <button class="tab-button" data-tab="effects" type="button">疗效对比</button>
            <button class="tab-button" data-tab="psm" type="button">PSM</button>
            <button class="tab-button" data-tab="prediction" type="button">预测模型</button>
            <button class="tab-button" data-tab="quality" type="button">数据质量</button>
          </nav>
          <div id="metric-grid" class="metric-grid"></div>
          <div id="filter-chips" class="filter-chips" aria-live="polite"></div>
          <div id="insight-strip" class="insight-strip" aria-label="核心结论摘要"></div>
          <details class="method-panel">
            <summary>方法说明</summary>
            <div class="method-grid">
              <div><strong>Cobb 改善量</strong>干预前 Cobb 角减去干预后 Cobb 角；正值表示角度减小。</div>
              <div><strong>改善≥5度</strong>将连续改善量转为二分类疗效指标，便于比较事件率。</div>
              <div><strong>SMD</strong>标准化均差，用于判断匹配前后基线变量是否平衡。</div>
              <div><strong>PSM</strong>用倾向评分匹配病情相似患者，减少治疗分配偏倚。</div>
              <div><strong>ATT</strong>匹配样本中支具+训练相对纯支具的平均治疗效果差异。</div>
              <div><strong>RMSE / R²</strong>交叉验证预测误差和解释度，仅用于流程展示。</div>
            </div>
          </details>
          <div id="tab-overview" class="tab-panel active">
            <div class="section-grid">
              <div class="panel">
                <h2>年龄分布</h2>
                <p class="caption">当前筛选队列，按年龄和治疗方案统计人数。</p>
                <div id="chart-age" class="chart"></div>
              </div>
              <div class="panel">
                <h2>干预前 Cobb 角分布</h2>
                <p class="caption">当前筛选队列，单位为度。</p>
                <div id="chart-pre-cobb" class="chart"></div>
              </div>
            </div>
            <div class="section-grid">
              <div class="panel">
                <h2>干预周期与 Cobb 改善量</h2>
                <p class="caption">仅包含疗效完整记录；正值表示 Cobb 角减小。</p>
                <div id="chart-duration-scatter" class="chart"></div>
              </div>
              <div class="panel">
                <h2>核心结果摘要</h2>
                <p class="caption">来自完整分析流程；筛选变化不会重新拟合统计模型。</p>
                <div id="summary-list" class="summary-list"></div>
              </div>
            </div>
          </div>
          <div id="tab-cohort" class="tab-panel">
            <div class="section-grid">
              <div class="panel">
                <h2>不同治疗方案的 Cobb 改善量</h2>
                <p class="caption">当前筛选且疗效完整队列，单位为度。</p>
                <div id="chart-treatment-box" class="chart"></div>
              </div>
              <div class="panel">
                <h2>改善达到 5 度比例</h2>
                <p class="caption">当前筛选队列的事件率。</p>
                <div id="chart-improved-rate" class="chart"></div>
              </div>
            </div>
            <div class="panel">
              <h2>队列表格</h2>
              <p class="caption">仅展示去标识化字段；可搜索、排序、分页和导出当前筛选结果。</p>
              <div class="table-actions" aria-label="队列表格工具">
                <input id="cohort-search" type="search" placeholder="搜索当前筛选队列" aria-label="搜索当前筛选队列">
                <select id="cohort-page-size" aria-label="每页显示行数">
                  <option value="30">每页 30 行</option>
                  <option value="50">每页 50 行</option>
                  <option value="all">显示全部</option>
                </select>
                <button id="cohort-prev" type="button">上一页</button>
                <button id="cohort-next" type="button">下一页</button>
                <button id="cohort-export" class="primary" type="button">导出 CSV</button>
                <span id="cohort-page-info" class="table-page-info"></span>
              </div>
              <div id="cohort-table" class="table-wrap"></div>
            </div>
          </div>
          <div id="tab-effects" class="tab-panel">
            <div class="analysis-note">统计推断结果来自完整 pipeline，不随当前筛选重新拟合。</div>
            <div class="section-grid">
              <div class="panel">
                <h2>连续疗效指标</h2>
                <p class="caption">Cobb 改善量 = 干预前 Cobb 角 - 干预后 Cobb 角。</p>
                <div id="inference-table" class="table-wrap"></div>
              </div>
              <div class="panel">
                <h2>二分类疗效指标</h2>
                <p class="caption">是否改善达到 5 度。</p>
                <div id="binary-table" class="table-wrap"></div>
              </div>
            </div>
          </div>
          <div id="tab-psm" class="tab-panel">
            <div class="analysis-note">PSM 结果来自完整 pipeline，不随当前筛选重新匹配。</div>
            <div class="section-grid">
              <div class="panel">
                <h2>协变量平衡</h2>
                <p class="caption">匹配前后标准化均差绝对值（SMD）。</p>
                <div id="chart-psm-balance" class="chart"></div>
              </div>
              <div class="panel">
                <h2>卡尺敏感性</h2>
                <p class="caption">不同匹配卡尺下的 ATT，单位为 Cobb 改善量度数。</p>
                <div id="chart-psm-sensitivity" class="chart"></div>
              </div>
            </div>
          </div>
          <div id="tab-prediction" class="tab-panel">
            <div class="analysis-note">模型结果来自完整 pipeline 的交叉验证，不随当前筛选重新训练。</div>
            <div class="section-grid">
              <div class="panel">
                <h2>实际值与预测值</h2>
                <p class="caption">Ridge 交叉验证预测结果，单位为 Cobb 改善量度数。</p>
                <div id="chart-prediction" class="chart"></div>
              </div>
              <div class="panel">
                <h2>特征消融</h2>
                <p class="caption">Delta RMSE 为正表示移除该特征后交叉验证误差变大。</p>
                <div id="chart-ablation" class="chart"></div>
              </div>
            </div>
            <div class="panel">
              <h2>模型指标</h2>
              <p class="caption">完整分析流程中的交叉验证结果。</p>
              <div id="model-table" class="table-wrap"></div>
            </div>
          </div>
          <div id="tab-quality" class="tab-panel">
            <div class="section-grid">
              <div class="panel">
                <h2>数据质量检查</h2>
                <p class="caption">分析流程中的校验结果和需复核项。</p>
                <div id="quality-table" class="table-wrap"></div>
              </div>
              <div class="panel">
                <h2>当前筛选摘要</h2>
                <p class="caption">当前筛选队列的描述性统计。</p>
                <div id="filter-summary-table" class="table-wrap"></div>
              </div>
            </div>
          </div>
        </section>
      </div>
    </main>
  </div>
  <script>
    const DASHBOARD_DATA = JSON.parse(document.getElementById("dashboard-data").textContent);
    const GROUP_LABEL = { brace_only: "纯支具", brace_training: "支具+训练" };
    const SEX_LABEL = { female: "女", male: "男", unknown: "未知" };
    const GROUP_COLOR = { brace_only: "#28679f", brace_training: "#bf5652" };
    const MUTED = "#5d6b7a";
    const FIELD_LABEL = {
      treatment_group: "治疗方案",
      sex: "性别",
      age: "年龄",
      bone_age_stage: "骨龄程度",
      intervention_months: "干预周期（月）",
      pre_cobb: "干预前 Cobb（度）",
      post_cobb: "干预后 Cobb（度）",
      cobb_change: "Cobb 改善量（度）",
      cobb_change_pct: "改善率",
      improved_5deg: "改善≥5度",
      training_exposure_hours: "训练暴露（小时/周）",
      analysis: "分析类型",
      estimand: "估计对象",
      estimate: "估计值",
      p_value: "P 值",
      ci_lower: "区间下限",
      ci_upper: "区间上限",
      n_obs: "样本量",
      events: "事件数",
      model: "模型",
      alpha: "alpha",
      n_splits: "折数",
      mae: "MAE",
      rmse: "RMSE",
      r2: "R²",
      check: "检查项",
      value: "值",
      status: "状态",
      note: "说明",
      metric: "指标",
      unit: "单位"
    };
    const VALUE_LABEL = {
      brace_only: "纯支具",
      brace_training: "支具+训练",
      female: "女",
      male: "男",
      group_summary: "分组均值",
      raw_mean_difference: "原始均值差",
      covariate_adjusted_ols: "协变量调整 OLS",
      group_rate: "分组比例",
      risk_difference: "风险差",
      odds_ratio: "优势比",
      adjusted_linear_probability: "调整线性概率模型",
      ridge: "Ridge 回归",
      mean_baseline: "均值基线",
      pass: "通过",
      review: "需复核",
      row_count: "总记录数",
      outcome_complete_rows: "疗效完整记录数",
      missing_pre_cobb: "干预前 Cobb 缺失数",
      missing_post_cobb: "干预后 Cobb 缺失数",
      negative_cobb_change_rows: "Cobb 改善量为负记录数",
      duration_gap_abs_gt_2_months: "记录周期与日期差超过 2 个月"
    };
    const FEATURE_LABEL = {
      pre_cobb: "干预前 Cobb",
      age: "年龄",
      intervention_months: "干预周期",
      training_exposure_hours: "训练暴露",
      duration_x_pre_cobb: "周期×干预前 Cobb",
      treatment_x_pre_cobb: "方案×干预前 Cobb",
      treatment_binary: "治疗方案",
      bone_age_stage: "骨龄程度",
      sex_female: "女性"
    };
    let filterState = null;
    let sortState = { key: "age", dir: "asc" };
    let cohortTableState = { page: 1, pageSize: 30, search: "" };
    const COHORT_COLUMNS = [
      "treatment_group", "sex", "age", "bone_age_stage", "intervention_months",
      "pre_cobb", "post_cobb", "cobb_change", "cobb_change_pct",
      "improved_5deg", "training_exposure_hours"
    ];

    function numberOrNull(value) {
      const n = Number(value);
      return Number.isFinite(n) ? n : null;
    }

    function finiteValues(rows, key) {
      return rows.map(row => numberOrNull(row[key])).filter(value => value !== null);
    }

    function mean(values) {
      const good = values.filter(value => Number.isFinite(value));
      if (!good.length) return null;
      return good.reduce((acc, value) => acc + value, 0) / good.length;
    }

    function fmt(value, digits = 1) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
      return Number(value).toFixed(digits);
    }

    function fmtPct(value, digits = 1) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
      return `${(Number(value) * 100).toFixed(digits)}%`;
    }

    function escapeHtml(value) {
      return String(value ?? "").replace(/[&<>"']/g, char => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }[char]));
    }

    function labelFor(key) {
      return FIELD_LABEL[key] || FEATURE_LABEL[key] || VALUE_LABEL[key] || key;
    }

    function translateText(value) {
      if (value === null || value === undefined) return "";
      let text = String(value);
      if (VALUE_LABEL[text]) return VALUE_LABEL[text];
      if (FEATURE_LABEL[text]) return FEATURE_LABEL[text];
      const replacements = [
        ["Raw mean difference in Cobb change", "Cobb 改善量原始均值差"],
        ["Adjusted treatment coefficient", "协变量调整治疗效应"],
        ["Risk difference for improved >=5deg", "改善≥5度风险差"],
        ["PSM matched ATT", "PSM 匹配 ATT"],
        ["Best Ridge CV RMSE", "最佳 Ridge 交叉验证 RMSE"],
        ["Mean Cobb change in brace_only", "纯支具组平均 Cobb 改善量"],
        ["Mean Cobb change in brace_training", "支具+训练组平均 Cobb 改善量"],
        ["Mean Cobb change: brace_training - brace_only", "Cobb 改善量均值差：支具+训练 - 纯支具"],
        ["Treatment coefficient adjusted for baseline covariates", "调整基线协变量后的治疗效应"],
        ["Improved >=5deg rate in brace_only", "纯支具组改善≥5度比例"],
        ["Improved >=5deg rate in brace_training", "支具+训练组改善≥5度比例"],
        ["Pr(improved>=5deg): brace_training - brace_only", "改善≥5度概率差：支具+训练 - 纯支具"],
        ["Odds ratio for improved>=5deg: training vs brace_only", "改善≥5度优势比：支具+训练 vs 纯支具"],
        ["Adjusted treatment effect on Pr(improved>=5deg)", "调整后改善≥5度治疗效应"],
        ["Full-pipeline results, not refit when filters change.", "完整流程结果，筛选变化不重新拟合。"],
        ["Descriptive group mean.", "描述性分组均值。"],
        ["Descriptive event rate.", "描述性事件率。"],
        ["Two-sided permutation p-value; bootstrap CI.", "双侧置换检验 P 值；Bootstrap 置信区间。"],
        ["Haldane-Anscombe corrected odds ratio.", "Haldane-Anscombe 校正优势比。"],
        ["Linear probability model; normal-approx p-value", "线性概率模型；正态近似 P 值"],
        ["Positive delta means removing this feature worsened CV RMSE.", "正值表示移除该特征后交叉验证 RMSE 变大。"],
        ["Reference model.", "完整模型基准。"],
        ["Expected 102 rows from two source sheets.", "来自两张原始表的预期记录数为 102。"],
        ["Rows with parsed pre/post Cobb angle and analyzable outcome.", "干预前后 Cobb 角可解析且疗效可分析的记录数。"],
        ["Parsed numeric baseline Cobb angle missingness.", "干预前 Cobb 角数值解析缺失情况。"],
        ["One missing post-intervention Cobb angle is retained in cleaned data.", "清洗数据中保留 1 条干预后 Cobb 角缺失记录。"],
        ["Negative values mean Cobb angle increased after intervention.", "负值表示干预后 Cobb 角增大。"],
        ["Compares recorded intervention months with date difference approximation.", "比较记录干预周期与日期差推算周期。"],
        ["patients", "例"],
        ["degrees", "度"],
        ["years", "岁"],
        ["months", "月"],
        ["Duration", "平均周期"],
        ["Generated", "生成时间"],
        ["rows", "条记录"],
        ["outcome-complete", "疗效完整"]
      ];
      replacements.forEach(([from, to]) => {
        text = text.split(from).join(to);
      });
      text = text.replace(/95% bootstrap CI/g, "95% Bootstrap 置信区间");
      text = text.replace(/95% approx CI/g, "95% 近似置信区间");
      text = text.replace(/, p=/g, "，P=");
      text = text.replace(/matched outcome-complete pairs/g, "对疗效完整匹配样本");
      text = text.replace(/alpha=/g, "alpha=");
      return text;
    }

    function displayValue(key, value) {
      if (value === null || value === undefined) return "";
      if (key === "treatment_group") return GROUP_LABEL[value] || value;
      if (key === "sex") return SEX_LABEL[value] || value;
      if (key === "improved_5deg") {
        if (Number(value) === 1) return "达到";
        if (Number(value) === 0) return "未达到";
      }
      if (key === "cobb_change_pct") return fmtPct(value, 1);
      if (["pre_cobb", "post_cobb", "cobb_change", "training_exposure_hours", "estimate", "p_value", "ci_lower", "ci_upper", "mae", "rmse", "r2", "value"].includes(key) && Number.isFinite(Number(value))) {
        return fmt(value, key === "p_value" || key === "r2" ? 3 : 2);
      }
      return translateText(value);
    }

    function formatCaliper(value) {
      const text = String(value ?? "");
      if (text === "no_caliper") return "无卡尺";
      return text.replace("_sd_logit", " 倍SD");
    }

    function executiveMetric(rawName) {
      return (DASHBOARD_DATA.executive_summary || []).find(row => row.metric === rawName) || null;
    }

    function shortDetail(text) {
      return translateText(text || "").replace(/\.$/, "。");
    }

    function valueWithUnit(value, unit, digits = 2) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
      return `${fmt(value, digits)} ${unit}`;
    }

    function getRange(key) {
      const range = DASHBOARD_DATA.ranges[key] || {};
      return { min: Number(range.min), max: Number(range.max) };
    }

    function createFilterState() {
      return {
        treatment: "all",
        sex: "all",
        improved: "all",
        ageMin: getRange("age").min,
        ageMax: getRange("age").max,
        cobbMin: getRange("pre_cobb").min,
        cobbMax: getRange("pre_cobb").max,
        durationMin: getRange("intervention_months").min,
        durationMax: getRange("intervention_months").max
      };
    }

    function setFilterInputsFromState() {
      document.getElementById("filter-treatment").value = filterState.treatment;
      document.getElementById("filter-sex").value = filterState.sex;
      document.getElementById("filter-improved").value = filterState.improved;
      document.getElementById("filter-age-min").value = filterState.ageMin;
      document.getElementById("filter-age-max").value = filterState.ageMax;
      document.getElementById("filter-cobb-min").value = filterState.cobbMin;
      document.getElementById("filter-cobb-max").value = filterState.cobbMax;
      document.getElementById("filter-duration-min").value = filterState.durationMin;
      document.getElementById("filter-duration-max").value = filterState.durationMax;
    }

    function readFilterStateFromInputs() {
      filterState = {
        treatment: document.getElementById("filter-treatment").value,
        sex: document.getElementById("filter-sex").value,
        improved: document.getElementById("filter-improved").value,
        ageMin: numberOrNull(document.getElementById("filter-age-min").value),
        ageMax: numberOrNull(document.getElementById("filter-age-max").value),
        cobbMin: numberOrNull(document.getElementById("filter-cobb-min").value),
        cobbMax: numberOrNull(document.getElementById("filter-cobb-max").value),
        durationMin: numberOrNull(document.getElementById("filter-duration-min").value),
        durationMax: numberOrNull(document.getElementById("filter-duration-max").value)
      };
    }

    function filteredPatients() {
      return DASHBOARD_DATA.patients.filter(row => {
        if (filterState.treatment !== "all" && row.treatment_group !== filterState.treatment) return false;
        if (filterState.sex !== "all" && row.sex !== filterState.sex) return false;
        const age = numberOrNull(row.age);
        const cobb = numberOrNull(row.pre_cobb);
        const duration = numberOrNull(row.intervention_months);
        if (age === null || age < filterState.ageMin || age > filterState.ageMax) return false;
        if (cobb === null || cobb < filterState.cobbMin || cobb > filterState.cobbMax) return false;
        if (duration === null || duration < filterState.durationMin || duration > filterState.durationMax) return false;
        if (filterState.improved === "missing") return row.outcome_observed !== 1;
        if (filterState.improved !== "all" && String(row.improved_5deg) !== filterState.improved) return false;
        return true;
      });
    }

    function outcomeComplete(rows) {
      return rows.filter(row => row.outcome_observed === 1 && numberOrNull(row.cobb_change) !== null);
    }

    function renderMetrics(rows) {
      const complete = outcomeComplete(rows);
      const braceOnly = rows.filter(row => row.treatment_group === "brace_only").length;
      const training = rows.filter(row => row.treatment_group === "brace_training").length;
      const avgChange = mean(finiteValues(complete, "cobb_change"));
      const improvedRows = complete.filter(row => Number(row.improved_5deg) === 1).length;
      const improvedRate = complete.length ? improvedRows / complete.length : null;
      const avgPre = mean(finiteValues(rows, "pre_cobb"));
      const avgDuration = mean(finiteValues(rows, "intervention_months"));
      const metrics = [
        ["筛选后记录", rows.length, "例"],
        ["纯支具", braceOnly, "例"],
        ["支具+训练", training, "例"],
        ["平均 Cobb 改善量", fmt(avgChange, 2), "度"],
        ["改善达到 5 度", fmtPct(improvedRate, 1), `${improvedRows}/${complete.length}`],
        ["平均干预前 Cobb", fmt(avgPre, 1), `平均周期 ${fmt(avgDuration, 1)} 月`]
      ];
      document.getElementById("metric-grid").innerHTML = metrics.map(item => `
        <div class="metric"><span>${escapeHtml(item[0])}</span><strong>${escapeHtml(item[1])}</strong><span>${escapeHtml(item[2])}</span></div>
      `).join("");
      document.getElementById("dashboard-subtitle").textContent =
        `${DASHBOARD_DATA.patients.length} 条记录，${DASHBOARD_DATA.patients.filter(row => row.outcome_observed === 1).length} 条疗效完整 | 生成时间 ${DASHBOARD_DATA.generated_at}`;
    }

    function renderFilterChips(rows) {
      const defaults = createFilterState();
      const chips = [];
      if (filterState.treatment !== "all") chips.push(`治疗方案：${GROUP_LABEL[filterState.treatment] || filterState.treatment}`);
      if (filterState.sex !== "all") chips.push(`性别：${SEX_LABEL[filterState.sex] || filterState.sex}`);
      if (filterState.improved !== "all") {
        const improvedLabel = filterState.improved === "missing" ? "疗效缺失" : Number(filterState.improved) === 1 ? "达到 5 度" : "未达到 5 度";
        chips.push(`疗效：${improvedLabel}`);
      }
      const rangeChip = (label, minKey, maxKey, defaultMin, defaultMax, unit, digits = 0) => {
        const min = filterState[minKey], max = filterState[maxKey];
        if (min !== defaultMin || max !== defaultMax) chips.push(`${label}：${fmt(min, digits)}-${fmt(max, digits)}${unit}`);
      };
      rangeChip("年龄", "ageMin", "ageMax", defaults.ageMin, defaults.ageMax, "岁", 0);
      rangeChip("干预前 Cobb", "cobbMin", "cobbMax", defaults.cobbMin, defaults.cobbMax, "度", 1);
      rangeChip("干预周期", "durationMin", "durationMax", defaults.durationMin, defaults.durationMax, "月", 0);

      const html = chips.length
        ? chips.map(text => `<span class="chip">${escapeHtml(text)}</span>`).join("") +
          `<span class="chip muted">当前 ${rows.length} 例</span><span class="chip"><button id="chip-clear-filters" type="button">清除全部</button></span>`
        : `<span class="chip muted">当前显示全部记录，共 ${rows.length} 例</span>`;
      document.getElementById("filter-chips").innerHTML = html;
      const clear = document.getElementById("chip-clear-filters");
      if (clear) {
        clear.addEventListener("click", () => {
          filterState = createFilterState();
          setFilterInputsFromState();
          cohortTableState.page = 1;
          renderDashboard();
        });
      }
    }

    function renderInsightStrip() {
      const items = [
        {
          label: "原始组间差异",
          row: executiveMetric("Raw mean difference in Cobb change"),
          unit: "度",
          note: "支具+训练 - 纯支具"
        },
        {
          label: "调整后差异",
          row: executiveMetric("Adjusted treatment coefficient"),
          unit: "度",
          note: "控制年龄、骨龄、基线 Cobb 等协变量"
        },
        {
          label: "PSM 匹配差异",
          row: executiveMetric("PSM matched ATT"),
          unit: "度",
          note: "病情相似样本的平均差异"
        },
        {
          label: "预测模型误差",
          row: executiveMetric("Best Ridge CV RMSE"),
          unit: "度",
          note: "交叉验证 RMSE，越低表示误差越小"
        }
      ];
      document.getElementById("insight-strip").innerHTML = items.map(item => {
        const value = item.row ? valueWithUnit(item.row.value, item.unit, 2) : "n/a";
        const detail = item.row ? shortDetail(item.row.detail) : item.note;
        return `
          <div class="insight-card">
            <span>${escapeHtml(item.label)}</span>
            <strong>${escapeHtml(value)}</strong>
            <small>${escapeHtml(item.note)}；${escapeHtml(detail)}</small>
          </div>
        `;
      }).join("");
    }

    function chartScales(width, height, margin, xMin, xMax, yMin, yMax) {
      const safeX = xMax === xMin ? [xMin - 1, xMax + 1] : [xMin, xMax];
      const safeY = yMax === yMin ? [yMin - 1, yMax + 1] : [yMin, yMax];
      return {
        x: value => margin.left + (value - safeX[0]) / (safeX[1] - safeX[0]) * (width - margin.left - margin.right),
        y: value => height - margin.bottom - (value - safeY[0]) / (safeY[1] - safeY[0]) * (height - margin.top - margin.bottom),
        xDomain: safeX,
        yDomain: safeY
      };
    }

    function axes(width, height, margin, xLabel, yLabel, xTicks, yTicks, xScale, yScale) {
      const grid = yTicks.map(t => `<line x1="${margin.left}" x2="${width - margin.right}" y1="${yScale(t)}" y2="${yScale(t)}" stroke="#e2e8f0"/>`).join("");
      const yLabels = yTicks.map(t => `<text x="${margin.left - 10}" y="${yScale(t) + 4}" text-anchor="end" fill="${MUTED}" font-size="12">${fmt(t, 1)}</text>`).join("");
      const xLabels = xTicks.map(t => `<text x="${xScale(t)}" y="${height - margin.bottom + 24}" text-anchor="middle" fill="${MUTED}" font-size="12">${fmt(t, 1)}</text>`).join("");
      return `
        ${grid}
        <line x1="${margin.left}" x2="${margin.left}" y1="${margin.top}" y2="${height - margin.bottom}" stroke="#34475a" stroke-width="1.5"/>
        <line x1="${margin.left}" x2="${width - margin.right}" y1="${height - margin.bottom}" y2="${height - margin.bottom}" stroke="#34475a" stroke-width="1.5"/>
        ${yLabels}${xLabels}
        <text x="${width / 2}" y="${height - 10}" text-anchor="middle" fill="#34475a" font-size="13">${escapeHtml(xLabel)}</text>
        ${yLabel ? `<text x="18" y="${height / 2}" text-anchor="middle" transform="rotate(-90 18 ${height / 2})" fill="#34475a" font-size="13">${escapeHtml(yLabel)}</text>` : ""}
      `;
    }

    function svgFrame(content, legend = "") {
      return `<svg viewBox="0 0 780 330" role="img" aria-label="数据图表">${content}${legend}</svg>`;
    }

    function niceTicks(min, max, count = 5) {
      if (!Number.isFinite(min) || !Number.isFinite(max)) return [0, 1];
      if (min === max) return [min - 1, min, min + 1];
      const ticks = [];
      for (let i = 0; i <= count; i++) ticks.push(min + (max - min) * i / count);
      return ticks;
    }

    function emptyChart(message) {
      return `<div class="empty">${escapeHtml(message)}</div>`;
    }

    function legendSvg(items, x = 565, y = 24) {
      return items.map((item, i) => `
        <rect x="${x}" y="${y + i * 20}" width="11" height="11" fill="${item.color}"/>
        <text x="${x + 17}" y="${y + 10 + i * 20}" fill="${MUTED}" font-size="12">${escapeHtml(item.label)}</text>
      `).join("");
    }

    function renderHistogram(targetId, rows, key, xLabel, bins = 10) {
      const values = finiteValues(rows, key);
      if (!values.length) {
        document.getElementById(targetId).innerHTML = emptyChart("当前筛选条件下没有可展示数据。");
        return;
      }
      const width = 780, height = 330;
      const margin = { top: 24, right: 42, bottom: 58, left: 66 };
      const min = Math.min(...values), max = Math.max(...values);
      const low = Math.floor(min), high = Math.ceil(max === min ? max + 1 : max);
      const binCount = Math.max(4, bins);
      const step = (high - low) / binCount || 1;
      const groups = ["brace_only", "brace_training"];
      const counts = groups.map(group => Array(binCount).fill(0));
      rows.forEach(row => {
        const value = numberOrNull(row[key]);
        if (value === null) return;
        const groupIndex = groups.indexOf(row.treatment_group);
        if (groupIndex < 0) return;
        const bin = Math.min(binCount - 1, Math.max(0, Math.floor((value - low) / step)));
        counts[groupIndex][bin] += 1;
      });
      const maxCount = Math.max(1, ...counts.flat());
      const scales = chartScales(width, height, margin, low, high, 0, maxCount * 1.15);
      const xTicks = niceTicks(low, high, 5);
      const yTicks = niceTicks(0, maxCount * 1.15, 4);
      const barW = (width - margin.left - margin.right) / binCount / 2.35;
      let bars = "";
      for (let b = 0; b < binCount; b++) {
        groups.forEach((group, gi) => {
          const x0 = scales.x(low + b * step) + gi * barW + 3;
          const y0 = scales.y(counts[gi][b]);
          const h = height - margin.bottom - y0;
          const start = low + b * step;
          const end = start + step;
          bars += `<rect x="${x0}" y="${y0}" width="${Math.max(1, barW - 4)}" height="${Math.max(0, h)}" fill="${GROUP_COLOR[group]}" opacity="0.88"><title>${escapeHtml(`${GROUP_LABEL[group]}｜${fmt(start, 1)}-${fmt(end, 1)}｜${counts[gi][b]} 例`)}</title></rect>`;
        });
      }
      document.getElementById(targetId).innerHTML = svgFrame(
        axes(width, height, margin, xLabel, "人数", xTicks, yTicks, scales.x, scales.y) +
        bars +
        legendSvg([{ label: "纯支具", color: GROUP_COLOR.brace_only }, { label: "支具+训练", color: GROUP_COLOR.brace_training }])
      );
    }

    function quartiles(values) {
      const sorted = values.slice().sort((a, b) => a - b);
      const q = p => {
        if (!sorted.length) return null;
        const idx = (sorted.length - 1) * p;
        const lo = Math.floor(idx), hi = Math.ceil(idx);
        return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
      };
      return { min: sorted[0], q1: q(0.25), med: q(0.5), q3: q(0.75), max: sorted[sorted.length - 1] };
    }

    function renderTreatmentBox(rows) {
      const complete = outcomeComplete(rows);
      const values = finiteValues(complete, "cobb_change");
      if (!values.length) {
        document.getElementById("chart-treatment-box").innerHTML = emptyChart("当前筛选条件下没有疗效完整记录。");
        return;
      }
      const width = 780, height = 330;
      const margin = { top: 24, right: 42, bottom: 58, left: 66 };
      const min = Math.min(...values), max = Math.max(...values);
      const scales = chartScales(width, height, margin, 0, 3, min - 1, max + 1);
      const yTicks = niceTicks(min - 1, max + 1, 5);
      let body = axes(width, height, margin, "治疗方案", "Cobb 改善量（度）", [], yTicks, scales.x, scales.y);
      if (min < 0 && max > 0) body += `<line x1="${margin.left}" x2="${width - margin.right}" y1="${scales.y(0)}" y2="${scales.y(0)}" stroke="#8a96a3" stroke-dasharray="4 4"><title>0 度参考线</title></line>`;
      ["brace_only", "brace_training"].forEach((group, idx) => {
        const groupRows = complete.filter(row => row.treatment_group === group);
        const groupValues = finiteValues(groupRows, "cobb_change");
        if (!groupValues.length) return;
        const q = quartiles(groupValues);
        const cx = scales.x(idx + 1);
        const boxW = 86;
        body += `
          <line x1="${cx}" x2="${cx}" y1="${scales.y(q.min)}" y2="${scales.y(q.max)}" stroke="${GROUP_COLOR[group]}" stroke-width="3"><title>${escapeHtml(`${GROUP_LABEL[group]}：范围 ${fmt(q.min, 1)} 到 ${fmt(q.max, 1)} 度`)}</title></line>
          <rect x="${cx - boxW / 2}" y="${scales.y(q.q3)}" width="${boxW}" height="${Math.max(1, scales.y(q.q1) - scales.y(q.q3))}" fill="${GROUP_COLOR[group]}" opacity="0.72" stroke="#263645"><title>${escapeHtml(`${GROUP_LABEL[group]}：Q1 ${fmt(q.q1, 1)}，中位数 ${fmt(q.med, 1)}，Q3 ${fmt(q.q3, 1)}`)}</title></rect>
          <line x1="${cx - boxW / 2}" x2="${cx + boxW / 2}" y1="${scales.y(q.med)}" y2="${scales.y(q.med)}" stroke="#1f2933" stroke-width="3"/>
          <text x="${cx}" y="${height - 28}" text-anchor="middle" fill="${MUTED}" font-size="12">${escapeHtml(GROUP_LABEL[group])}</text>
        `;
        groupRows.forEach((row, i) => {
          const value = numberOrNull(row.cobb_change);
          if (value === null) return;
          const offset = ((i % 9) - 4) * 7;
          body += `<circle cx="${cx + offset}" cy="${scales.y(value)}" r="3.2" fill="#ffffff" stroke="${GROUP_COLOR[group]}" stroke-width="1.4" opacity="0.86"><title>${escapeHtml(`${GROUP_LABEL[group]}｜Cobb 改善量 ${fmt(value, 1)} 度`)}</title></circle>`;
        });
      });
      document.getElementById("chart-treatment-box").innerHTML = svgFrame(body);
    }

    function renderScatter(rows, targetId, xKey, yKey, xLabel, yLabel) {
      const points = rows.map(row => ({
        x: numberOrNull(row[xKey]),
        y: numberOrNull(row[yKey]),
        group: row.treatment_group
      })).filter(point => point.x !== null && point.y !== null);
      if (!points.length) {
        document.getElementById(targetId).innerHTML = emptyChart("当前筛选条件下没有可展示数据。");
        return;
      }
      const width = 780, height = 330;
      const margin = { top: 24, right: 42, bottom: 58, left: 66 };
      const xs = points.map(point => point.x), ys = points.map(point => point.y);
      const xMin = Math.min(...xs), xMax = Math.max(...xs), yMin = Math.min(...ys), yMax = Math.max(...ys);
      const scales = chartScales(width, height, margin, xMin - 1, xMax + 1, yMin - 1, yMax + 1);
      let content = axes(width, height, margin, xLabel, yLabel, niceTicks(xMin - 1, xMax + 1), niceTicks(yMin - 1, yMax + 1), scales.x, scales.y);
      if (yMin < 0 && yMax > 0) content += `<line x1="${margin.left}" x2="${width - margin.right}" y1="${scales.y(0)}" y2="${scales.y(0)}" stroke="#8a96a3" stroke-dasharray="4 4"><title>0 度参考线</title></line>`;
      content += points.map(point => `<circle cx="${scales.x(point.x)}" cy="${scales.y(point.y)}" r="4.5" fill="${GROUP_COLOR[point.group] || "#637381"}" opacity="0.82"><title>${escapeHtml(`${GROUP_LABEL[point.group] || "未知"}｜${xLabel} ${fmt(point.x, 1)}｜${yLabel} ${fmt(point.y, 1)}`)}</title></circle>`).join("") +
        legendSvg([{ label: "纯支具", color: GROUP_COLOR.brace_only }, { label: "支具+训练", color: GROUP_COLOR.brace_training }]);
      document.getElementById(targetId).innerHTML = svgFrame(content);
    }

    function renderImprovedRate(rows) {
      const complete = outcomeComplete(rows);
      const width = 780, height = 330;
      const margin = { top: 24, right: 42, bottom: 58, left: 66 };
      const scales = chartScales(width, height, margin, 0, 3, 0, 1);
      let content = axes(width, height, margin, "治疗方案", "比例", [], niceTicks(0, 1), scales.x, scales.y);
      ["brace_only", "brace_training"].forEach((group, idx) => {
        const part = complete.filter(row => row.treatment_group === group);
        const events = part.filter(row => Number(row.improved_5deg) === 1).length;
        const rate = part.length ? events / part.length : 0;
        const cx = scales.x(idx + 1);
        const barW = 92;
        content += `
          <rect x="${cx - barW / 2}" y="${scales.y(rate)}" width="${barW}" height="${height - margin.bottom - scales.y(rate)}" fill="${GROUP_COLOR[group]}" opacity="0.88"><title>${escapeHtml(`${GROUP_LABEL[group]}：${events}/${part.length}，${fmtPct(rate, 1)}`)}</title></rect>
          <text x="${cx}" y="${scales.y(rate) - 8}" text-anchor="middle" fill="#1f2933" font-size="13">${fmtPct(rate, 1)}</text>
          <text x="${cx}" y="${height - 30}" text-anchor="middle" fill="${MUTED}" font-size="12">${escapeHtml(GROUP_LABEL[group])}</text>
          <text x="${cx}" y="${height - 12}" text-anchor="middle" fill="${MUTED}" font-size="11">${events}/${part.length}</text>
        `;
      });
      document.getElementById("chart-improved-rate").innerHTML = svgFrame(content);
    }

    function renderPsmBalance() {
      const rows = DASHBOARD_DATA.psm_balance || [];
      const variables = [...new Set(rows.map(row => row.variable))];
      if (!variables.length) {
        document.getElementById("chart-psm-balance").innerHTML = emptyChart("暂无 PSM 平衡数据。");
        return;
      }
      const width = 780, height = 330;
      const margin = { top: 24, right: 42, bottom: 58, left: 140 };
      const max = Math.max(0.5, ...rows.map(row => Number(row.abs_smd) || 0)) * 1.1;
      const scales = chartScales(width, height, margin, 0, max, -0.5, variables.length - 0.5);
      let content = axes(width, height, margin, "SMD 绝对值", "", niceTicks(0, max), [], scales.x, scales.y);
      content += `<line x1="${scales.x(0.1)}" x2="${scales.x(0.1)}" y1="${margin.top}" y2="${height - margin.bottom}" stroke="#b7791f" stroke-width="1.5" stroke-dasharray="4 4"><title>SMD 0.1 平衡参考线</title></line>`;
      content += `<text x="${scales.x(0.1) + 5}" y="${margin.top + 12}" fill="#8a5a10" font-size="11">SMD=0.1</text>`;
      variables.forEach((variable, i) => {
        const before = rows.find(row => row.variable === variable && row.stage === "before_matching");
        const after = rows.find(row => row.variable === variable && row.stage === "after_matching");
        const y = scales.y(i);
        content += `<text x="${margin.left - 8}" y="${y + 4}" text-anchor="end" fill="${MUTED}" font-size="12">${escapeHtml(labelFor(variable))}</text>`;
        [[before, "#7b8794", -8], [after, "#2a9d8f", 8]].forEach(item => {
          if (!item[0]) return;
          const val = Number(item[0].abs_smd) || 0;
          const stageLabel = item[0].stage === "before_matching" ? "匹配前" : "匹配后";
          content += `<rect x="${margin.left}" y="${y + item[2] - 5}" width="${Math.max(0, scales.x(val) - margin.left)}" height="10" fill="${item[1]}"><title>${escapeHtml(`${labelFor(variable)}｜${stageLabel} SMD=${fmt(val, 3)}`)}</title></rect>`;
        });
      });
      content += legendSvg([{ label: "匹配前", color: "#7b8794" }, { label: "匹配后", color: "#2a9d8f" }], 605, 24);
      document.getElementById("chart-psm-balance").innerHTML = svgFrame(content);
    }

    function renderPsmSensitivity() {
      const rows = (DASHBOARD_DATA.psm_sensitivity || []).filter(row => row.att_cobb_change !== null);
      if (!rows.length) {
        document.getElementById("chart-psm-sensitivity").innerHTML = emptyChart("暂无 PSM 敏感性数据。");
        return;
      }
      const width = 780, height = 330;
      const margin = { top: 24, right: 42, bottom: 68, left: 66 };
      const values = rows.flatMap(row => [Number(row.att_ci_lower), Number(row.att_ci_upper), Number(row.att_cobb_change)]).filter(Number.isFinite);
      const min = Math.min(...values), max = Math.max(...values);
      const scales = chartScales(width, height, margin, -0.5, rows.length - 0.5, min - 0.5, max + 0.5);
      let content = axes(width, height, margin, "卡尺设置", "ATT（度）", [], niceTicks(min - 0.5, max + 0.5), scales.x, scales.y);
      if (min < 0 && max > 0) content += `<line x1="${margin.left}" x2="${width - margin.right}" y1="${scales.y(0)}" y2="${scales.y(0)}" stroke="#6b7280"/>`;
      rows.forEach((row, i) => {
        const x = scales.x(i);
        const y = scales.y(Number(row.att_cobb_change));
        content += `
          <line x1="${x}" x2="${x}" y1="${scales.y(Number(row.att_ci_lower))}" y2="${scales.y(Number(row.att_ci_upper))}" stroke="#34475a" stroke-width="2.5"><title>${escapeHtml(`${formatCaliper(row.caliper_label)}：区间 ${fmt(row.att_ci_lower, 2)} 到 ${fmt(row.att_ci_upper, 2)} 度`)}</title></line>
          <circle cx="${x}" cy="${y}" r="6" fill="#2a9d8f" stroke="#1f2933"><title>${escapeHtml(`${formatCaliper(row.caliper_label)}：ATT ${fmt(row.att_cobb_change, 2)} 度，匹配 ${row.matched_pairs_outcome_complete} 对`)}</title></circle>
          <text x="${x}" y="${height - 42}" text-anchor="middle" fill="${MUTED}" font-size="11">${escapeHtml(formatCaliper(row.caliper_label))}</text>
          <text x="${x}" y="${height - 24}" text-anchor="middle" fill="${MUTED}" font-size="11">n=${escapeHtml(row.matched_pairs_outcome_complete)}</text>
        `;
      });
      document.getElementById("chart-psm-sensitivity").innerHTML = svgFrame(content);
    }

    function renderPredictionScatter() {
      const points = (DASHBOARD_DATA.model_predictions || []).map(row => ({
        x: numberOrNull(row.actual_cobb_change),
        y: numberOrNull(row.predicted_cobb_change),
        group: Number(row.treatment_binary) === 1 ? "brace_training" : "brace_only"
      })).filter(point => point.x !== null && point.y !== null);
      if (!points.length) {
        document.getElementById("chart-prediction").innerHTML = emptyChart("暂无预测结果数据。");
        return;
      }
      const width = 780, height = 330;
      const margin = { top: 24, right: 42, bottom: 58, left: 66 };
      const all = points.flatMap(point => [point.x, point.y]);
      const min = Math.min(...all) - 1, max = Math.max(...all) + 1;
      const scales = chartScales(width, height, margin, min, max, min, max);
      const content = axes(width, height, margin, "实际 Cobb 改善量", "预测 Cobb 改善量", niceTicks(min, max), niceTicks(min, max), scales.x, scales.y) +
        `<line x1="${scales.x(min)}" x2="${scales.x(max)}" y1="${scales.y(min)}" y2="${scales.y(max)}" stroke="#6b7280"/>` +
        points.map(point => `<circle cx="${scales.x(point.x)}" cy="${scales.y(point.y)}" r="4" fill="${GROUP_COLOR[point.group]}" opacity="0.78"><title>${escapeHtml(`${GROUP_LABEL[point.group]}｜实际 ${fmt(point.x, 1)} 度｜预测 ${fmt(point.y, 1)} 度`)}</title></circle>`).join("") +
        legendSvg([{ label: "纯支具", color: GROUP_COLOR.brace_only }, { label: "支具+训练", color: GROUP_COLOR.brace_training }]);
      document.getElementById("chart-prediction").innerHTML = svgFrame(content);
    }

    function renderAblation() {
      const rows = (DASHBOARD_DATA.model_feature_ablation || []).filter(row => row.removed_feature !== "none_full_model");
      if (!rows.length) {
        document.getElementById("chart-ablation").innerHTML = emptyChart("暂无特征消融数据。");
        return;
      }
      const width = 780, height = 330;
      const margin = { top: 24, right: 42, bottom: 58, left: 170 };
      const values = rows.map(row => Number(row.delta_rmse_vs_full)).filter(Number.isFinite);
      const min = Math.min(0, ...values), max = Math.max(0, ...values);
      const scales = chartScales(width, height, margin, min - 0.02, max + 0.02, -0.5, rows.length - 0.5);
      let content = axes(width, height, margin, "相对完整模型的 RMSE 变化", "", niceTicks(min - 0.02, max + 0.02), [], scales.x, scales.y);
      content += `<line x1="${scales.x(0)}" x2="${scales.x(0)}" y1="${margin.top}" y2="${height - margin.bottom}" stroke="#6b7280"/>`;
      rows.forEach((row, i) => {
        const value = Number(row.delta_rmse_vs_full);
        const y = scales.y(i);
        const x0 = scales.x(0);
        const x1 = scales.x(value);
        content += `
          <text x="${margin.left - 8}" y="${y + 4}" text-anchor="end" fill="${MUTED}" font-size="11">${escapeHtml(labelFor(row.removed_feature))}</text>
          <rect x="${Math.min(x0, x1)}" y="${y - 6}" width="${Math.abs(x1 - x0)}" height="12" fill="${value >= 0 ? "#2a9d8f" : "#c85050"}"><title>${escapeHtml(`${labelFor(row.removed_feature)}：Delta RMSE ${fmt(value, 3)}`)}</title></rect>
        `;
      });
      document.getElementById("chart-ablation").innerHTML = svgFrame(content);
    }

    function renderSummaryList() {
      const rows = DASHBOARD_DATA.executive_summary || [];
      document.getElementById("summary-list").innerHTML = rows.map(row => `
        <div class="summary-row">
          <strong>${escapeHtml(translateText(row.metric))}</strong>
          <span>${fmt(row.value, 3)}</span>
          <small>${escapeHtml(translateText(row.detail))}</small>
        </div>
      `).join("");
    }

    function renderGenericTable(targetId, rows, columns = null) {
      if (!rows || !rows.length) {
        document.getElementById(targetId).innerHTML = emptyChart("暂无可展示记录。");
        return;
      }
      const cols = columns || Object.keys(rows[0]);
      const head = `<thead><tr>${cols.map(col => `<th>${escapeHtml(labelFor(col))}</th>`).join("")}</tr></thead>`;
      const body = rows.map(row => `<tr>${cols.map(col => `<td>${escapeHtml(displayValue(col, row[col]))}</td>`).join("")}</tr>`).join("");
      document.getElementById(targetId).innerHTML = `<table>${head}<tbody>${body}</tbody></table>`;
    }

    function sortedCohortRows(rows) {
      return rows.slice().sort((a, b) => {
        const av = a[sortState.key], bv = b[sortState.key];
        const na = numberOrNull(av), nb = numberOrNull(bv);
        let cmp;
        if (na !== null && nb !== null) cmp = na - nb;
        else cmp = String(av ?? "").localeCompare(String(bv ?? ""));
        return sortState.dir === "asc" ? cmp : -cmp;
      });
    }

    function searchedCohortRows(rows) {
      const query = cohortTableState.search.trim().toLowerCase();
      const sorted = sortedCohortRows(rows);
      if (!query) return sorted;
      return sorted.filter(row => COHORT_COLUMNS.some(col => String(formatCell(col, row[col])).toLowerCase().includes(query)));
    }

    function pagedCohortRows(rows) {
      const displayRows = searchedCohortRows(rows);
      if (cohortTableState.pageSize === "all") {
        cohortTableState.page = 1;
        return { displayRows, pageRows: displayRows, totalPages: 1 };
      }
      const pageSize = Number(cohortTableState.pageSize) || 30;
      const totalPages = Math.max(1, Math.ceil(displayRows.length / pageSize));
      cohortTableState.page = Math.min(Math.max(1, cohortTableState.page), totalPages);
      const start = (cohortTableState.page - 1) * pageSize;
      return { displayRows, pageRows: displayRows.slice(start, start + pageSize), totalPages };
    }

    function renderCohortTable(rows) {
      const { displayRows, pageRows, totalPages } = pagedCohortRows(rows);
      const head = `<thead><tr>${COHORT_COLUMNS.map(col => `<th data-sort="${col}">${escapeHtml(labelFor(col))}${sortState.key === col ? (sortState.dir === "asc" ? " ↑" : " ↓") : ""}</th>`).join("")}</tr></thead>`;
      const body = pageRows.length
        ? pageRows.map(row => `<tr>${COHORT_COLUMNS.map(col => `<td>${escapeHtml(formatCell(col, row[col]))}</td>`).join("")}</tr>`).join("")
        : `<tr><td colspan="${COHORT_COLUMNS.length}">当前搜索条件下没有记录。</td></tr>`;
      document.getElementById("cohort-table").innerHTML = `<table>${head}<tbody>${body}</tbody></table>`;
      document.getElementById("cohort-page-info").textContent =
        `第 ${cohortTableState.page}/${totalPages} 页，共 ${displayRows.length} 条`;
      document.getElementById("cohort-prev").disabled = cohortTableState.page <= 1;
      document.getElementById("cohort-next").disabled = cohortTableState.page >= totalPages;
      document.querySelectorAll("#cohort-table th").forEach(th => {
        th.addEventListener("click", () => {
          const key = th.dataset.sort;
          if (sortState.key === key) sortState.dir = sortState.dir === "asc" ? "desc" : "asc";
          else sortState = { key, dir: "asc" };
          cohortTableState.page = 1;
          renderCohortTable(filteredPatients());
        });
      });
    }

    function csvEscape(value) {
      const text = String(value ?? "");
      return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
    }

    function exportCohortCsv() {
      const rows = searchedCohortRows(filteredPatients());
      const lines = [
        COHORT_COLUMNS.map(col => csvEscape(labelFor(col))).join(","),
        ...rows.map(row => COHORT_COLUMNS.map(col => csvEscape(formatCell(col, row[col]))).join(","))
      ];
      const blob = new Blob(["\ufeff" + lines.join("\n")], { type: "text/csv;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "scoliosis_filtered_cohort.csv";
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    }

    function formatCell(key, value) {
      if (value === null || value === undefined) return "";
      if (["cobb_change_pct"].includes(key)) return fmtPct(value, 1);
      if (["pre_cobb", "post_cobb", "cobb_change", "training_exposure_hours"].includes(key)) return fmt(value, 1);
      if (key === "treatment_group") return GROUP_LABEL[value] || value;
      if (key === "sex") return SEX_LABEL[value] || value;
      if (key === "improved_5deg") return Number(value) === 1 ? "达到" : "未达到";
      return value;
    }

    function renderFilterSummary(rows) {
      const complete = outcomeComplete(rows);
      const summaryRows = [
        { metric: "筛选后记录", value: rows.length, unit: "例" },
        { metric: "疗效完整记录", value: complete.length, unit: "例" },
        { metric: "平均年龄", value: fmt(mean(finiteValues(rows, "age")), 1), unit: "岁" },
        { metric: "平均干预前 Cobb", value: fmt(mean(finiteValues(rows, "pre_cobb")), 1), unit: "度" },
        { metric: "平均干预周期", value: fmt(mean(finiteValues(rows, "intervention_months")), 1), unit: "月" },
        { metric: "平均 Cobb 改善量", value: fmt(mean(finiteValues(complete, "cobb_change")), 2), unit: "度" }
      ];
      renderGenericTable("filter-summary-table", summaryRows, ["metric", "value", "unit"]);
    }

    function renderStaticTables() {
      renderGenericTable("inference-table", DASHBOARD_DATA.inference, ["analysis", "estimand", "estimate", "p_value", "ci_lower", "ci_upper", "n_obs"]);
      renderGenericTable("binary-table", DASHBOARD_DATA.binary_outcome, ["analysis", "estimand", "estimate", "p_value", "ci_lower", "ci_upper", "n_obs", "events"]);
      renderGenericTable("model-table", DASHBOARD_DATA.model_metrics, ["model", "alpha", "n_splits", "mae", "rmse", "r2"]);
      renderGenericTable("quality-table", DASHBOARD_DATA.data_quality, ["check", "value", "status", "note"]);
    }

    function renderSvgCharts(rows) {
      renderHistogram("chart-age", rows, "age", "年龄（岁）", 10);
      renderHistogram("chart-pre-cobb", rows, "pre_cobb", "干预前 Cobb 角（度）", 10);
      renderScatter(outcomeComplete(rows), "chart-duration-scatter", "intervention_months", "cobb_change", "干预周期（月）", "Cobb 改善量（度）");
      renderTreatmentBox(rows);
      renderImprovedRate(rows);
      renderPsmBalance();
      renderPsmSensitivity();
      renderPredictionScatter();
      renderAblation();
    }

    function renderDashboard() {
      const rows = filteredPatients();
      renderMetrics(rows);
      renderFilterChips(rows);
      renderInsightStrip();
      renderSvgCharts(rows);
      renderSummaryList();
      renderCohortTable(rows);
      renderFilterSummary(rows);
      renderStaticTables();
    }

    function switchTab(tabName) {
      document.querySelectorAll(".tab-button").forEach(button => {
        button.classList.toggle("active", button.dataset.tab === tabName);
      });
      document.querySelectorAll(".tab-panel").forEach(panel => {
        panel.classList.toggle("active", panel.id === `tab-${tabName}`);
      });
    }

    function setupEvents() {
      document.querySelectorAll("aside select, aside input").forEach(input => {
        input.addEventListener("input", () => {
          readFilterStateFromInputs();
          cohortTableState.page = 1;
          renderDashboard();
        });
      });
      document.getElementById("reset-filters").addEventListener("click", () => {
        filterState = createFilterState();
        setFilterInputsFromState();
        cohortTableState.page = 1;
        renderDashboard();
      });
      document.getElementById("cohort-search").addEventListener("input", event => {
        cohortTableState.search = event.target.value;
        cohortTableState.page = 1;
        renderCohortTable(filteredPatients());
      });
      document.getElementById("cohort-page-size").addEventListener("change", event => {
        cohortTableState.pageSize = event.target.value === "all" ? "all" : Number(event.target.value);
        cohortTableState.page = 1;
        renderCohortTable(filteredPatients());
      });
      document.getElementById("cohort-prev").addEventListener("click", () => {
        cohortTableState.page -= 1;
        renderCohortTable(filteredPatients());
      });
      document.getElementById("cohort-next").addEventListener("click", () => {
        cohortTableState.page += 1;
        renderCohortTable(filteredPatients());
      });
      document.getElementById("cohort-export").addEventListener("click", exportCohortCsv);
      document.querySelectorAll(".tab-button").forEach(button => {
        button.addEventListener("click", () => switchTab(button.dataset.tab));
      });
    }

    function initializeDashboard() {
      filterState = createFilterState();
      setFilterInputsFromState();
      setupEvents();
      renderDashboard();
    }

    initializeDashboard();
  </script>
</body>
</html>
"""
