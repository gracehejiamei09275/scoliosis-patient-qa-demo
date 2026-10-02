from __future__ import annotations

import argparse
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from .config import DEFAULT_DATA_FILE, DEFAULT_OUTPUT_DIR
from .account_store import AccountStore
from .model_service import artifact_path, build_patient_features, ensure_model_artifact
from .patient_qa import run_patient_qa
from .qa_router import QuestionType, route_question


DEFAULT_QUESTION = "我13岁，支具加训练8个月，大概可能改善多少？"
SAMPLE_PATIENT = {
    "age": 13,
    "sex": "female",
    "bone_age_stage": 4,
    "pre_cobb": 28,
    "post_cobb": None,
    "intervention_months": 8,
    "treatment_group": "brace_training",
    "weekly_training_hours": 3,
}


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


def _json_bytes(payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> tuple[int, bytes]:
    return int(status), json.dumps(
        payload,
        ensure_ascii=False,
        default=_json_default,
    ).encode("utf-8")


def build_index_html(default_llm: str = "mock") -> str:
    static_page = Path(__file__).resolve().parents[1] / "docs" / "index.html"
    if static_page.exists():
        return static_page.read_text(encoding="utf-8")
    sample_json = json.dumps(SAMPLE_PATIENT, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>脊柱侧弯疗效问答</title>
  <style>
    :root {{
      --bg: #f5f7fa;
      --surface: #ffffff;
      --line: #dce3ec;
      --text: #202631;
      --muted: #667386;
      --blue: #2d5f9a;
      --green: #287a63;
      --orange: #ad5f22;
      --red: #9d3333;
      --soft-blue: #eaf2fb;
      --soft-green: #eaf6f1;
      --soft-orange: #fff4ea;
      --shadow: 0 12px 32px rgba(32, 38, 49, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.5;
    }}
    header {{
      background: var(--surface);
      border-bottom: 1px solid var(--line);
    }}
    .topbar {{
      max-width: 1220px;
      margin: 0 auto;
      padding: 24px;
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 18px;
      align-items: center;
    }}
    h1 {{
      margin: 0;
      font-size: 26px;
      letter-spacing: 0;
    }}
    .subtitle {{
      margin: 8px 0 0;
      color: var(--muted);
      font-size: 14px;
    }}
    .mode {{
      display: flex;
      gap: 8px;
      align-items: center;
      background: var(--bg);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
    }}
    .mode label {{
      font-size: 13px;
      color: var(--muted);
    }}
    select, input, textarea {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      color: var(--text);
      background: #fff;
      font: inherit;
    }}
    select:focus, input:focus, textarea:focus {{
      outline: 3px solid rgba(45, 95, 154, 0.14);
      border-color: var(--blue);
    }}
    main {{
      max-width: 1220px;
      margin: 0 auto;
      padding: 24px;
    }}
    .notice {{
      border: 1px solid #f0c9a6;
      background: var(--soft-orange);
      color: #6f3a14;
      padding: 12px 14px;
      border-radius: 8px;
      margin-bottom: 18px;
      font-size: 14px;
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(300px, 420px) minmax(0, 1fr);
      gap: 18px;
      align-items: start;
    }}
    section {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 20px;
    }}
    h2 {{
      margin: 0 0 16px;
      font-size: 18px;
    }}
    .form-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    .field {{
      display: grid;
      gap: 6px;
    }}
    .field.full {{
      grid-column: 1 / -1;
    }}
    label {{
      color: var(--muted);
      font-size: 13px;
    }}
    textarea {{
      min-height: 118px;
      resize: vertical;
    }}
    .actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 16px;
    }}
    button {{
      border: 1px solid transparent;
      border-radius: 8px;
      padding: 10px 14px;
      font: inherit;
      cursor: pointer;
      min-height: 42px;
    }}
    .primary {{
      color: #fff;
      background: var(--blue);
    }}
    .secondary {{
      color: var(--text);
      background: #fff;
      border-color: var(--line);
    }}
    .primary:disabled {{
      opacity: 0.58;
      cursor: wait;
    }}
    .result {{
      min-height: 232px;
      border: 1px solid var(--line);
      background: #fbfcfe;
      border-radius: 8px;
      padding: 16px;
      white-space: pre-wrap;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin: 18px 0;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fff;
      min-height: 82px;
    }}
    .metric .label {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
    }}
    .metric .value {{
      font-size: 20px;
      font-weight: 700;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      min-height: 28px;
      padding: 4px 9px;
      border-radius: 999px;
      font-size: 12px;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--muted);
    }}
    .pill.good {{
      background: var(--soft-green);
      border-color: #b8dece;
      color: var(--green);
    }}
    .pill.warn {{
      background: var(--soft-orange);
      border-color: #efc29d;
      color: var(--orange);
    }}
    .contrib {{
      display: grid;
      gap: 8px;
      margin-top: 10px;
    }}
    .bar-row {{
      display: grid;
      grid-template-columns: 190px minmax(0, 1fr) 64px;
      gap: 10px;
      align-items: center;
      font-size: 13px;
    }}
    .bar-track {{
      height: 10px;
      border-radius: 999px;
      background: #edf1f6;
      overflow: hidden;
    }}
    .bar {{
      height: 100%;
      width: 0;
      background: var(--green);
    }}
    .bar.neg {{
      background: var(--orange);
    }}
    .status {{
      margin-top: 12px;
      color: var(--muted);
      font-size: 13px;
    }}
    .error {{
      color: var(--red);
    }}
    .footer-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
      margin-top: 18px;
    }}
    .small {{
      color: var(--muted);
      font-size: 13px;
    }}
    code {{
      color: var(--blue);
      overflow-wrap: anywhere;
    }}
    @media (max-width: 900px) {{
      .topbar, .layout, .footer-grid {{
        grid-template-columns: 1fr;
      }}
      .metrics {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
    }}
    @media (max-width: 560px) {{
      main, .topbar {{ padding: 16px; }}
      .form-grid, .metrics {{ grid-template-columns: 1fr; }}
      .bar-row {{ grid-template-columns: 1fr; gap: 5px; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="topbar">
      <div>
        <h1>脊柱侧弯疗效问答</h1>
        <p class="subtitle">ML 预测 + LLM 解释，本地运行的患者友好型数据科学 demo</p>
      </div>
      <div class="mode">
        <label for="llmProvider">LLM</label>
        <select id="llmProvider">
          <option value="mock" {"selected" if default_llm == "mock" else ""}>Mock</option>
          <option value="http" {"selected" if default_llm == "http" else ""}>HTTP</option>
        </select>
      </div>
    </div>
  </header>
  <main>
    <div class="notice">本系统仅用于探索性数据分析展示，不构成临床诊断/建议。样本量有限、非随机分组，个体治疗请咨询医生。</div>
    <div class="layout">
      <section>
        <h2>患者变量</h2>
        <div class="form-grid">
          <div class="field">
            <label for="age">年龄</label>
            <input id="age" type="number" min="1" max="30" step="0.1" value="13">
          </div>
          <div class="field">
            <label for="sex">性别</label>
            <select id="sex">
              <option value="female">女</option>
              <option value="male">男</option>
            </select>
          </div>
          <div class="field">
            <label for="boneAge">骨龄程度</label>
            <input id="boneAge" type="number" min="0" max="10" step="0.1" value="4">
          </div>
          <div class="field">
            <label for="preCobb">干预前 Cobb 角</label>
            <input id="preCobb" type="number" min="1" max="90" step="0.1" value="28">
          </div>
          <div class="field">
            <label for="postCobb">干预后 Cobb 角</label>
            <input id="postCobb" type="number" min="0" max="90" step="0.1" placeholder="可选">
          </div>
          <div class="field">
            <label for="months">干预周期（月）</label>
            <input id="months" type="number" min="0.1" max="60" step="0.1" value="8">
          </div>
          <div class="field">
            <label for="treatment">治疗方案</label>
            <select id="treatment">
              <option value="brace_training">支具+训练</option>
              <option value="brace_only">纯支具</option>
            </select>
          </div>
          <div class="field">
            <label for="hours">每周训练时间</label>
            <input id="hours" type="number" min="0" max="40" step="0.1" value="3">
          </div>
          <div class="field full">
            <label for="question">问题</label>
            <textarea id="question">{DEFAULT_QUESTION}</textarea>
          </div>
        </div>
        <div class="actions">
          <button id="submitBtn" class="primary">提交问答</button>
          <button id="sampleBtn" class="secondary">填入示例</button>
          <button id="clearBtn" class="secondary">清空结果</button>
        </div>
        <div id="status" class="status"></div>
      </section>
      <section>
        <h2>回答</h2>
        <div id="answer" class="result">提交后将在这里显示回答。</div>
        <div class="metrics">
          <div class="metric">
            <div class="label">预测改善量</div>
            <div id="metricChange" class="value">--</div>
          </div>
          <div class="metric">
            <div class="label">5度阈值</div>
            <div id="metricImproved" class="value"><span class="pill">--</span></div>
          </div>
          <div class="metric">
            <div class="label">RMSE</div>
            <div id="metricRmse" class="value">--</div>
          </div>
          <div class="metric">
            <div class="label">R²</div>
            <div id="metricR2" class="value">--</div>
          </div>
        </div>
        <h2>主要贡献项</h2>
        <div id="contrib" class="contrib"><span class="small">暂无预测结果。</span></div>
      </section>
    </div>
    <div class="footer-grid">
      <section>
        <h2>隐私</h2>
        <p class="small">界面只提交结构化患者变量和问题；后端 QA 上下文使用聚合统计，不向 LLM 传原始 Excel 行级记录、日期字段或原始 Cobb 字符串。</p>
      </section>
      <section>
        <h2>运行状态</h2>
        <p id="health" class="small">正在读取服务状态。</p>
        <p id="logPath" class="small"></p>
      </section>
    </div>
  </main>
  <script>
    const samplePatient = {sample_json};
    const defaultQuestion = {json.dumps(DEFAULT_QUESTION, ensure_ascii=False)};
    const fields = {{
      age: document.getElementById('age'),
      sex: document.getElementById('sex'),
      boneAge: document.getElementById('boneAge'),
      preCobb: document.getElementById('preCobb'),
      postCobb: document.getElementById('postCobb'),
      months: document.getElementById('months'),
      treatment: document.getElementById('treatment'),
      hours: document.getElementById('hours'),
      question: document.getElementById('question'),
      llmProvider: document.getElementById('llmProvider')
    }};

    function numericValue(input, optional=false) {{
      if (optional && input.value.trim() === '') return null;
      const value = Number(input.value);
      if (!Number.isFinite(value)) throw new Error(input.previousElementSibling.textContent + ' 需要填写数字');
      return value;
    }}

    function syncTreatment() {{
      const braceOnly = fields.treatment.value === 'brace_only';
      fields.hours.disabled = braceOnly;
      if (braceOnly) fields.hours.value = '0';
    }}

    function buildPatient() {{
      const patient = {{
        age: numericValue(fields.age),
        sex: fields.sex.value,
        bone_age_stage: numericValue(fields.boneAge),
        pre_cobb: numericValue(fields.preCobb),
        intervention_months: numericValue(fields.months),
        treatment_group: fields.treatment.value
      }};
      const post = numericValue(fields.postCobb, true);
      if (post !== null) patient.post_cobb = post;
      if (fields.treatment.value === 'brace_training') {{
        patient.weekly_training_hours = numericValue(fields.hours);
      }} else {{
        patient.weekly_training_hours = 0;
      }}
      return patient;
    }}

    function fillSample() {{
      fields.age.value = samplePatient.age;
      fields.sex.value = samplePatient.sex;
      fields.boneAge.value = samplePatient.bone_age_stage;
      fields.preCobb.value = samplePatient.pre_cobb;
      fields.postCobb.value = '';
      fields.months.value = samplePatient.intervention_months;
      fields.treatment.value = samplePatient.treatment_group;
      fields.hours.value = samplePatient.weekly_training_hours;
      fields.question.value = defaultQuestion;
      syncTreatment();
    }}

    function setStatus(message, error=false) {{
      const status = document.getElementById('status');
      status.textContent = message;
      status.className = error ? 'status error' : 'status';
    }}

    function clearResult() {{
      document.getElementById('answer').textContent = '提交后将在这里显示回答。';
      document.getElementById('metricChange').textContent = '--';
      document.getElementById('metricImproved').innerHTML = '<span class="pill">--</span>';
      document.getElementById('metricRmse').textContent = '--';
      document.getElementById('metricR2').textContent = '--';
      document.getElementById('contrib').innerHTML = '<span class="small">暂无预测结果。</span>';
      document.getElementById('logPath').textContent = '';
      setStatus('');
    }}

    function renderPrediction(prediction) {{
      if (!prediction) {{
        document.getElementById('metricChange').textContent = '--';
        document.getElementById('metricImproved').innerHTML = '<span class="pill">--</span>';
        document.getElementById('metricRmse').textContent = '--';
        document.getElementById('metricR2').textContent = '--';
        document.getElementById('contrib').innerHTML = '<span class="small">暂无预测结果。</span>';
        return;
      }}
      document.getElementById('metricChange').textContent = prediction.predicted_cobb_change.toFixed(2) + '°';
      const cls = prediction.predicted_improved_5deg ? 'pill good' : 'pill warn';
      const label = prediction.predicted_improved_5deg ? '达到' : '未达到';
      document.getElementById('metricImproved').innerHTML = `<span class="${{cls}}">${{label}}</span>`;
      document.getElementById('metricRmse').textContent = prediction.model_metrics.rmse.toFixed(2);
      document.getElementById('metricR2').textContent = prediction.model_metrics.r2.toFixed(3);
      const maxAbs = Math.max(...prediction.feature_contributions.slice(0, 5).map(item => Math.abs(item.contribution)), 0.01);
      document.getElementById('contrib').innerHTML = prediction.feature_contributions.slice(0, 5).map(item => {{
        const width = Math.max(5, Math.abs(item.contribution) / maxAbs * 100);
        const neg = item.contribution < 0 ? ' neg' : '';
        return `<div class="bar-row"><span>${{item.feature}}</span><div class="bar-track"><div class="bar${{neg}}" style="width:${{width}}%"></div></div><span>${{item.contribution >= 0 ? '+' : ''}}${{item.contribution.toFixed(2)}}</span></div>`;
      }}).join('');
    }}

    async function submitQa() {{
      const button = document.getElementById('submitBtn');
      button.disabled = true;
      setStatus('正在生成回答。');
      try {{
        const response = await fetch('/api/qa', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{
            question: fields.question.value.trim(),
            patient: buildPatient(),
            llm_provider: fields.llmProvider.value
          }})
        }});
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || '请求失败');
        document.getElementById('answer').textContent = data.answer || '';
        renderPrediction(data.prediction);
        document.getElementById('logPath').textContent = data.qa_logged ? '问答记录已保存在服务器。' : '';
        setStatus('已完成。');
      }} catch (error) {{
        setStatus(error.message, true);
      }} finally {{
        button.disabled = false;
      }}
    }}

    async function loadHealth() {{
      try {{
        const response = await fetch('/api/health');
        const data = await response.json();
        document.getElementById('health').innerHTML = `默认 LLM：<code>${{data.default_llm}}</code>；模型 artifact：<code>${{data.model_artifact_exists ? '已存在' : '未生成'}}</code>`;
      }} catch (error) {{
        document.getElementById('health').textContent = '服务状态读取失败。';
      }}
    }}

    fields.treatment.addEventListener('change', syncTreatment);
    document.getElementById('sampleBtn').addEventListener('click', fillSample);
    document.getElementById('clearBtn').addEventListener('click', clearResult);
    document.getElementById('submitBtn').addEventListener('click', submitQa);
    syncTreatment();
    loadHealth();
  </script>
</body>
</html>"""


class WebAppState:
    def __init__(
        self,
        *,
        data_file: Path,
        output_dir: Path,
        default_llm: str,
        account_store: AccountStore,
        auth_required: bool,
    ) -> None:
        self.data_file = data_file
        self.output_dir = output_dir
        self.default_llm = default_llm
        self.account_store = account_store
        self.auth_required = auth_required
        self.allowed_origins = {
            value.strip()
            for value in os.environ.get(
                "ALLOWED_ORIGINS",
                "https://gracehejiamei09275.github.io,http://localhost:8000,http://127.0.0.1:8000",
            ).split(",")
            if value.strip()
        }


def make_handler(state: WebAppState) -> type[BaseHTTPRequestHandler]:
    class PatientQAHandler(BaseHTTPRequestHandler):
        server_version = "PatientQAWeb/2.0"

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            origin = self.headers.get("Origin", "")
            if origin in state.allowed_origins:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            code, body = _json_bytes(payload, status)
            self._send(code, body, "application/json; charset=utf-8")

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 131_072:
                raise ValueError("请求内容为空或过大。")
            raw = self.rfile.read(length).decode("utf-8")
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("请求内容必须是 JSON 对象。")
            return payload

        def _raw_token(self) -> str | None:
            authorization = self.headers.get("Authorization", "")
            if authorization.startswith("Bearer "):
                return authorization[7:].strip()
            return None

        def _current_user(self) -> dict[str, Any] | None:
            return state.account_store.user_for_token(self._raw_token())

        def _require_user(self) -> dict[str, Any] | None:
            user = self._current_user()
            if user is None:
                self._send_json({"error": "请先登录账号。"}, HTTPStatus.UNAUTHORIZED)
            return user

        def _require_admin(self) -> dict[str, Any] | None:
            user = self._require_user()
            if user is not None and user["role"] != "admin":
                self._send_json({"error": "仅管理员可以查看该页面。"}, HTTPStatus.FORBIDDEN)
                return None
            return user

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                html = build_index_html(state.default_llm).encode("utf-8")
                self._send(HTTPStatus.OK, html, "text/html; charset=utf-8")
                return
            if path == "/api/health":
                self._send_json(
                    {
                        "status": "ok",
                        "default_llm": state.default_llm,
                        "auth_required": state.auth_required,
                        "model_artifact_exists": artifact_path(state.output_dir).exists(),
                    }
                )
                return
            if path == "/api/me":
                user = self._require_user()
                if user is not None:
                    self._send_json({"user": user})
                return
            if path == "/api/history":
                user = self._require_user()
                if user is not None:
                    self._send_json({"items": state.account_store.history(user["id"])})
                return
            if path == "/api/admin/stats":
                if self._require_admin() is not None:
                    self._send_json(state.account_store.admin_stats())
                return
            if path == "/api/admin/users":
                if self._require_admin() is not None:
                    self._send_json({"items": state.account_store.admin_users()})
                return
            if path == "/api/admin/assessments":
                if self._require_admin() is not None:
                    self._send_json({"items": state.account_store.admin_assessments()})
                return
            self._send_json({"error": "Not found."}, HTTPStatus.NOT_FOUND)

        def do_HEAD(self) -> None:
            path = urlparse(self.path).path
            if path in {"/", "/api/health", "/api/me", "/api/history"}:
                self.send_response(HTTPStatus.OK)
                content_type = "text/html; charset=utf-8" if path == "/" else "application/json; charset=utf-8"
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                return
            self.send_response(HTTPStatus.NOT_FOUND)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

        def do_OPTIONS(self) -> None:
            self.send_response(HTTPStatus.NO_CONTENT)
            origin = self.headers.get("Origin", "")
            if origin in state.allowed_origins:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
            self.send_header("Access-Control-Max-Age", "600")
            self.end_headers()

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            try:
                request_payload = self._read_json()
            except (ValueError, json.JSONDecodeError) as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return

            if path == "/api/register":
                try:
                    user, token = state.account_store.register(
                        email=str(request_payload.get("email", "")),
                        password=str(request_payload.get("password", "")),
                        display_name=str(request_payload.get("display_name", "")),
                        consent=request_payload.get("consent") is True,
                    )
                except ValueError as exc:
                    self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                    return
                self._send_json({"user": user, "token": token}, HTTPStatus.CREATED)
                return

            if path == "/api/login":
                try:
                    user, token = state.account_store.login(
                        email=str(request_payload.get("email", "")),
                        password=str(request_payload.get("password", "")),
                    )
                except ValueError as exc:
                    self._send_json({"error": str(exc)}, HTTPStatus.UNAUTHORIZED)
                    return
                self._send_json({"user": user, "token": token})
                return

            if path == "/api/logout":
                state.account_store.logout(self._raw_token())
                self._send_json({"ok": True})
                return

            if path != "/api/qa":
                self._send_json({"error": "Not found."}, HTTPStatus.NOT_FOUND)
                return

            user = self._current_user()
            if state.auth_required and user is None:
                self._send_json({"error": "请先注册或登录后再提交。"}, HTTPStatus.UNAUTHORIZED)
                return

            question = str(request_payload.get("question", "")).strip()
            patient = request_payload.get("patient")
            llm_provider = str(request_payload.get("llm_provider") or state.default_llm)
            if not question:
                self._send_json({"error": "Field 'question' is required."}, HTTPStatus.BAD_REQUEST)
                return
            if llm_provider not in {"mock", "http"}:
                self._send_json({"error": "Field 'llm_provider' must be mock or http."}, HTTPStatus.BAD_REQUEST)
                return
            if patient is not None and not isinstance(patient, dict):
                self._send_json({"error": "Field 'patient' must be an object."}, HTTPStatus.BAD_REQUEST)
                return

            route = route_question(question, has_patient_payload=patient is not None)
            if route.question_type != QuestionType.UNSAFE_MEDICAL_ADVICE:
                if route.question_type == QuestionType.PREDICTION and patient is None:
                    self._send_json({"error": "Prediction questions require a patient object."}, HTTPStatus.BAD_REQUEST)
                    return
                if patient is not None:
                    try:
                        build_patient_features(patient)
                    except ValueError as exc:
                        self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                        return

            try:
                result = run_patient_qa(
                    question=question,
                    patient=patient,
                    data_file=state.data_file,
                    output_dir=state.output_dir,
                    llm_provider=llm_provider,
                )
            except Exception as exc:
                self._send_json({"error": f"{type(exc).__name__}: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
                return

            response = {
                "answer": result.get("answer"),
                "route": result.get("route"),
                "prediction": result.get("prediction"),
                "privacy_policy": result.get("privacy_policy"),
                "qa_logged": bool(result.get("qa_log")),
            }
            if user is not None and isinstance(patient, dict):
                response["assessment_id"] = state.account_store.save_assessment(
                    user_id=user["id"],
                    question=question,
                    patient=patient,
                    answer=str(result.get("answer") or ""),
                    prediction=result.get("prediction"),
                )
            self._send_json(response)

    return PatientQAHandler


def build_server(
    *,
    host: str,
    port: int,
    data_file: Path | str = DEFAULT_DATA_FILE,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    default_llm: str = "mock",
    database_url: str | None = None,
    auth_required: bool | None = None,
) -> ThreadingHTTPServer:
    resolved_output_dir = Path(output_dir)
    state = WebAppState(
        data_file=Path(data_file),
        output_dir=resolved_output_dir,
        default_llm=default_llm,
        account_store=AccountStore(
            database_url if database_url is not None else os.environ.get("DATABASE_URL"),
            sqlite_path=resolved_output_dir / "patient_accounts.sqlite3",
        ),
        auth_required=(
            auth_required
            if auth_required is not None
            else os.environ.get("AUTH_REQUIRED", "0").lower() in {"1", "true", "yes"}
        ),
    )
    ensure_model_artifact(data_file=state.data_file, output_dir=state.output_dir)
    handler = make_handler(state)
    return ThreadingHTTPServer((host, port), handler)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local patient QA web interface.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_FILE), help="Input Excel workbook path.")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--llm", default="mock", choices=["mock", "http"], help="Default LLM mode.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", default=8000, type=int, help="Port to bind.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    server = build_server(
        host=args.host,
        port=args.port,
        data_file=args.data,
        output_dir=args.out,
        default_llm=args.llm,
    )
    host, port = server.server_address
    print(f"Patient QA web app running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

