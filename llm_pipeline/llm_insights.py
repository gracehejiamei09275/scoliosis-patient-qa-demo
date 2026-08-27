from __future__ import annotations

import json
import os
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd


SAFETY_TERMS = "探索性；非临床诊断/建议；样本量限制；非随机分组。"
EXCLUDED_ROW_LEVEL_FIELDS = [
    "source_id",
    "brace_scan_date",
    "followup_date",
    "pre_cobb_raw",
    "post_cobb_raw",
]


def _records(df: pd.DataFrame, max_rows: int = 30) -> list[dict[str, Any]]:
    safe = df.head(max_rows).replace({pd.NA: None})
    return json.loads(safe.to_json(orient="records", force_ascii=False))


def build_llm_payload(
    *,
    data_quality: pd.DataFrame,
    eda_summary: pd.DataFrame,
    treatment_snapshot: pd.DataFrame,
    inference: pd.DataFrame,
    binary_outcome: pd.DataFrame,
    psm_effect: pd.DataFrame,
    psm_balance: pd.DataFrame,
    psm_sensitivity: pd.DataFrame,
    model_metrics: pd.DataFrame,
    model_feature_importance: pd.DataFrame,
    row_count: int,
) -> dict[str, Any]:
    """Build an aggregate-only payload for LLM interpretation."""
    return {
        "privacy_policy": {
            "row_level_records_sent": False,
            "excluded_fields": EXCLUDED_ROW_LEVEL_FIELDS,
            "note": "Only aggregate statistics are provided to the LLM insight layer.",
        },
        "project_context": {
            "audience": "course or portfolio demo",
            "outcome_definition": "cobb_change = pre_cobb - post_cobb; improved_5deg = cobb_change >= 5",
            "clinical_boundary": SAFETY_TERMS,
            "row_count": int(row_count),
        },
        "data_quality": _records(data_quality),
        "eda_snapshot": _records(treatment_snapshot),
        "eda_summary_sample": _records(eda_summary, max_rows=16),
        "inference": _records(inference),
        "binary_outcome": _records(binary_outcome),
        "psm_effect": _records(psm_effect),
        "psm_balance": _records(psm_balance),
        "psm_sensitivity": _records(psm_sensitivity),
        "model_metrics": _records(model_metrics),
        "model_feature_importance": _records(model_feature_importance),
    }


def build_prompt(payload: dict[str, Any]) -> str:
    return (
        "你是一个数据科学项目报告助手。请基于下面的聚合统计结果生成中文解释，"
        "只讨论探索性数据分析、统计推断、PSM 和预测模型表现。必须明确说明："
        "这不是临床诊断/建议，样本量有限，分组不是随机实验，PSM 不能证明因果确定性。\n\n"
        f"聚合数据如下：\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


class LLMClient(ABC):
    @abstractmethod
    def complete(self, prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class MockLLMClient(LLMClient):
    def complete(self, prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
        inference = pd.DataFrame(payload["inference"])
        psm = pd.DataFrame(payload["psm_effect"])
        metrics = pd.DataFrame(payload["model_metrics"])
        raw = inference[inference["analysis"].eq("raw_mean_difference")].iloc[0]
        psm_row = psm.iloc[0]
        ridge = metrics[metrics["model"].eq("ridge")].sort_values("rmse").iloc[0]
        return {
            "provider": "mock",
            "safety_terms": SAFETY_TERMS,
            "sections": {
                "executive_summary": (
                    "这是一个面向课程/作品集展示的探索性 full-pipeline 数据科学项目。"
                    "分析覆盖清洗、EDA、统计推断、PSM 和预测模型，但不是临床诊断/建议。"
                ),
                "data_quality": (
                    f"当前聚合结果显示样本量限制明显：总样本量为 {payload['project_context']['row_count']}，"
                    "且治疗方案不是随机分配，因此所有结论应作为探索性证据阅读。"
                ),
                "statistical_interpretation": (
                    f"原始组间均值差为 {float(raw['estimate']):.2f} 度，"
                    f"置换检验 p={float(raw['p_value']):.3f}。这提示当前数据中没有稳定的统计学优势信号。"
                ),
                "psm_interpretation": (
                    f"PSM 匹配后的 ATT 为 {float(psm_row['estimate']):.2f} 度，"
                    f"使用 {int(psm_row['matched_pairs'])} 对结局完整匹配样本。"
                    "PSM 有助于降低已观测基线差异，但不能消除未观测混杂。"
                ),
                "modeling_interpretation": (
                    f"Ridge 交叉验证 RMSE 约为 {float(ridge['rmse']):.2f}，"
                    f"R2 约为 {float(ridge['r2']):.3f}。预测模型更适合展示变量工程和误差分析，"
                    "不应用于个体化临床决策。"
                ),
                "figure_captions": {
                    "age_distribution.png": "两组患者年龄分布，用于观察基线构成差异。",
                    "pre_cobb_distribution.png": "干预前 Cobb 角分布，用于评估病情基线可比性。",
                    "cobb_change_by_group.png": "两组平均 Cobb 改善量对比，仅作探索性描述。",
                    "duration_vs_cobb_change.png": "干预周期与 Cobb 改善量的关系，提示疗程与疗效可能相关。",
                    "psm_balance_smd.png": "匹配前后 SMD，用于评估 PSM 是否改善协变量平衡。",
                    "model_predictions.png": "交叉验证预测值与真实改善量对比，用于评估模型误差。",
                },
            },
        }


class HTTPJSONLLMClient(LLMClient):
    """Minimal provider-agnostic HTTP adapter for OpenAI-compatible JSON APIs."""

    def __init__(self, env: dict[str, str]) -> None:
        self.api_url = normalize_chat_completions_url(env.get("LLM_API_URL", ""))
        self.api_key = env.get("LLM_API_KEY", "")
        self.model = env.get("LLM_MODEL", "default")
        if not self.api_url:
            raise ValueError("LLM_API_URL is required for http LLM provider.")

    def complete(self, prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Return a concise Chinese explanation. Do not provide diagnosis, "
                            "treatment selection, or individualized medical advice."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(self.api_url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = json.loads(response.read().decode("utf-8"))
        content = (
            raw.get("choices", [{}])[0]
            .get("message", {})
            .get("content", json.dumps(raw, ensure_ascii=False))
        )
        return {
            "provider": "http",
            "safety_terms": SAFETY_TERMS,
            "sections": {
                "executive_summary": content,
                "data_quality": "由外部 LLM 生成；请结合结构化表格审阅。",
                "statistical_interpretation": "由外部 LLM 生成；请结合结构化表格审阅。",
                "psm_interpretation": "由外部 LLM 生成；请结合结构化表格审阅。",
                "modeling_interpretation": "由外部 LLM 生成；请结合结构化表格审阅。",
                "figure_captions": {},
            },
        }


def normalize_chat_completions_url(url: str) -> str:
    cleaned = url.strip().rstrip("/")
    if not cleaned:
        return ""
    if cleaned.endswith("/chat/completions"):
        return cleaned
    if cleaned.endswith("/v1"):
        return f"{cleaned}/chat/completions"
    return cleaned


def load_env(path: Path | None = None) -> dict[str, str]:
    values = dict(os.environ)
    dotenv_path = path or Path(".env")
    if dotenv_path.exists():
        for line in dotenv_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def get_llm_client(provider: str, env: dict[str, str] | None = None) -> LLMClient:
    provider = provider.lower()
    if provider == "mock":
        return MockLLMClient()
    if provider == "http":
        return HTTPJSONLLMClient(env or load_env())
    raise ValueError(f"Unsupported LLM provider: {provider}. Expected 'mock' or 'http'.")


def generate_llm_insights(payload: dict[str, Any], provider: str = "mock") -> dict[str, Any]:
    prompt = build_prompt(payload)
    client = get_llm_client(provider)
    result = client.complete(prompt, payload)
    result["input_policy"] = payload["privacy_policy"]
    result["prompt_audit"] = {
        "aggregate_only": True,
        "excluded_fields": EXCLUDED_ROW_LEVEL_FIELDS,
        "prompt_character_count": len(prompt),
    }
    return result
