from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .causal_psm import run_psm
from .cleaning import load_clean_data
from .config import DEFAULT_DATA_FILE, DEFAULT_OUTPUT_DIR
from .inference import run_binary_outcome_analysis, run_inference
from .llm_insights import EXCLUDED_ROW_LEVEL_FIELDS, get_llm_client
from .model_service import SAFETY_NOTE, predict_patient
from .modeling import run_modeling
from .qa_router import QuestionType, RouteResult


REQUIRED_SAFETY_PHRASE = "探索性；不是临床诊断/建议；样本量限制；非随机分组；需咨询医生。"


def _read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def _records(df: pd.DataFrame, max_rows: int = 12) -> list[dict[str, Any]]:
    return json.loads(df.head(max_rows).replace({pd.NA: None}).to_json(orient="records", force_ascii=False))


def load_project_context(
    data_file: Path | str = DEFAULT_DATA_FILE,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Load aggregate-only context for QA answers."""
    output_dir = Path(output_dir)
    table_dir = output_dir / "tables"
    inference = _read_csv_if_exists(table_dir / "inference_results.csv")
    binary_outcome = _read_csv_if_exists(table_dir / "binary_outcome_results.csv")
    psm_effect = _read_csv_if_exists(table_dir / "psm_effect.csv")
    model_metrics = _read_csv_if_exists(table_dir / "model_metrics.csv")

    if any(frame is None for frame in [inference, binary_outcome, psm_effect, model_metrics]):
        cleaned = load_clean_data(data_file)
        inference, _ = run_inference(cleaned)
        binary_outcome, _ = run_binary_outcome_analysis(cleaned)
        psm = run_psm(cleaned)
        modeling = run_modeling(cleaned)
        psm_effect = psm["effect"]
        model_metrics = modeling["metrics"]
    else:
        cleaned = None

    assert inference is not None
    assert binary_outcome is not None
    assert psm_effect is not None
    assert model_metrics is not None

    row_count = int(len(cleaned)) if cleaned is not None else None
    if row_count is None:
        manifest_path = output_dir / "run_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            row_count = int(manifest.get("n_rows_cleaned", 0))
        else:
            row_count = int(load_clean_data(data_file).shape[0])

    return {
        "privacy_policy": {
            "row_level_records_sent": False,
            "excluded_fields": EXCLUDED_ROW_LEVEL_FIELDS,
            "note": "QA context uses aggregate tables and normalized patient input only.",
        },
        "row_count": row_count,
        "inference": _records(inference),
        "binary_outcome": _records(binary_outcome),
        "psm_effect": _records(psm_effect),
        "model_metrics": _records(model_metrics),
    }


def _lookup_row(rows: list[dict[str, Any]], key: str, value: str) -> dict[str, Any] | None:
    for row in rows:
        if row.get(key) == value:
            return row
    return None


def build_qa_prompt(payload: dict[str, Any]) -> str:
    return (
        "你是患者友好的数据科学解释助手。请基于去标识化患者输入、机器学习预测结果和聚合统计背景回答。"
        "不要给诊断或治疗建议。必须包含：探索性、不是临床诊断/建议、样本量限制、非随机分组、需咨询医生。\n\n"
        f"QA payload:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def _mock_prediction_answer(payload: dict[str, Any]) -> str:
    prediction = payload["prediction"]
    patient = prediction["normalized_patient"]
    metrics = prediction["model_metrics"]
    actual = ""
    if patient.get("actual_cobb_change") is not None:
        actual = (
            f"\n你提供了干预后 Cobb 角，因此可以直接计算实际改善量："
            f"{patient['actual_cobb_change']:.2f} 度；"
            f"{'达到' if patient['actual_improved_5deg'] else '未达到'} 5 度改善阈值。"
        )
    top_features = prediction["feature_contributions"][:3]
    feature_text = "、".join(
        f"{item['feature']}({item['contribution']:+.2f})" for item in top_features
    )
    return (
        f"根据当前探索性 Ridge 模型，输入患者在“{patient['treatment_label']}”方案下的预测 Cobb 改善量"
        f"约为 {prediction['predicted_cobb_change']:.2f} 度；"
        f"{'模型预测达到' if prediction['predicted_improved_5deg'] else '模型预测未达到'} 5 度改善阈值。"
        f"\n{prediction['prediction_interval_note']}"
        f"\n交叉验证指标：RMSE={metrics['rmse']:.2f}，MAE={metrics['mae']:.2f}，R2={metrics['r2']:.3f}。"
        f"\n主要模型贡献项约为：{feature_text}。这些贡献只是线性模型解释，不代表因果关系。"
        f"{actual}"
        f"\n安全边界：{REQUIRED_SAFETY_PHRASE}"
    )


def _flatten_json_text(value: Any, indent: int = 0) -> list[str]:
    prefix = "  " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.extend(_flatten_json_text(item, indent + 1))
            else:
                lines.append(f"{prefix}{key}: {item}")
        return lines
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.extend(_flatten_json_text(item, indent))
            else:
                lines.append(f"{prefix}- {item}")
        return lines
    return [f"{prefix}{value}"]


def _format_external_llm_answer(content: str) -> str:
    text = str(content).strip()
    if not text:
        return ""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        answer = text
    else:
        answer = "\n".join(_flatten_json_text(parsed))
    if "不是临床诊断/建议" not in answer:
        answer = f"{answer}\n安全边界：{REQUIRED_SAFETY_PHRASE}"
    return answer


def _mock_education_answer(question: str) -> str:
    if "psm" in question.lower() or "倾向评分" in question:
        return (
            "PSM 倾向评分匹配是把基线特征相似的两类患者进行匹配，帮助减少医生选择治疗方案时造成的已观测基线偏差。"
            "它适合做探索性因果敏感性分析，但不能消除未观测混杂，也不能证明因果确定性。"
            f"\n安全边界：{REQUIRED_SAFETY_PHRASE}"
        )
    return (
        "Cobb 改善量在本项目中定义为：干预前 Cobb 角 - 干预后 Cobb 角。"
        "数值越大表示角度下降越多；若改善量 >=5 度，项目会标记为 improved_5deg。"
        f"\n安全边界：{REQUIRED_SAFETY_PHRASE}"
    )


def _mock_outcome_answer(payload: dict[str, Any]) -> str:
    patient = payload.get("prediction", {}).get("normalized_patient")
    if patient and patient.get("actual_cobb_change") is not None:
        return (
            f"按项目定义，Cobb 改善量 = 干预前 {patient['pre_cobb']:.2f} - 干预后 {patient['post_cobb']:.2f}，"
            f"所以实际改善量为 {patient['actual_cobb_change']:.2f} 度；"
            f"{'达到' if patient['actual_improved_5deg'] else '未达到'} 5 度改善阈值。"
            f"\n安全边界：{REQUIRED_SAFETY_PHRASE}"
        )
    return (
        "如果提供干预前和干预后 Cobb 角，我可以按 pre_cobb - post_cobb 计算改善量。"
        f"\n安全边界：{REQUIRED_SAFETY_PHRASE}"
    )


def _mock_research_answer(context: dict[str, Any]) -> str:
    raw = _lookup_row(context["inference"], "analysis", "raw_mean_difference")
    adjusted = _lookup_row(context["inference"], "analysis", "covariate_adjusted_ols")
    psm = context["psm_effect"][0] if context["psm_effect"] else {}
    raw_text = ""
    if raw:
        raw_text += f"原始均值差约为 {float(raw['estimate']):.2f} 度，p={float(raw['p_value']):.3f}。"
    if adjusted:
        raw_text += f"协变量调整后治疗系数约为 {float(adjusted['estimate']):.2f} 度。"
    if psm:
        raw_text += f"PSM ATT 约为 {float(psm['estimate']):.2f} 度，匹配对数 {int(psm['matched_pairs'])}。"
    return (
        f"当前聚合结果显示：{raw_text}"
        "这些结果更适合作为探索性研究展示，不能说明某个治疗方案一定更好。"
        f"\n安全边界：{REQUIRED_SAFETY_PHRASE}"
    )


def _unsafe_answer() -> str:
    return (
        "这个问题涉及个体化诊断、治疗选择或治疗方案制定，我不能给出具体医疗建议。"
        "这个系统只能解释数据项目中的探索性模型输出和统计结果。"
        f"\n安全边界：{REQUIRED_SAFETY_PHRASE}"
    )


def compose_answer(
    *,
    question: str,
    route: RouteResult,
    context: dict[str, Any],
    artifact: dict[str, Any],
    patient: dict[str, Any] | None = None,
    llm_provider: str = "mock",
) -> dict[str, Any]:
    prediction = None
    error = None
    if patient is not None and route.question_type != QuestionType.UNSAFE_MEDICAL_ADVICE:
        try:
            prediction = predict_patient(patient, artifact)
        except ValueError as exc:
            error = str(exc)

    payload = {
        "question": question,
        "route": {"question_type": route.question_type.value, "reason": route.reason},
        "patient_input": prediction["normalized_patient"] if prediction else None,
        "prediction": prediction,
        "context": context,
        "privacy_policy": context["privacy_policy"],
        "safety_requirement": REQUIRED_SAFETY_PHRASE,
    }

    if route.question_type == QuestionType.UNSAFE_MEDICAL_ADVICE:
        answer = _unsafe_answer()
    elif error:
        answer = f"无法完成预测：{error}\n安全边界：{REQUIRED_SAFETY_PHRASE}"
    elif route.question_type == QuestionType.PREDICTION:
        if prediction is None:
            answer = (
                "要进行预测，请提供结构化 patient JSON：age、sex、bone_age_stage、pre_cobb、"
                "intervention_months、treatment_group；支具+训练还需要 weekly_training_hours。"
                f"\n安全边界：{REQUIRED_SAFETY_PHRASE}"
            )
        elif llm_provider == "mock":
            answer = _mock_prediction_answer(payload)
        else:
            try:
                client = get_llm_client(llm_provider)
                llm_result = client.complete(build_qa_prompt(payload), payload)
                raw_answer = llm_result.get("sections", {}).get("executive_summary", "")
                answer = _format_external_llm_answer(str(raw_answer))
            except Exception as exc:
                answer = (
                    _mock_prediction_answer(payload)
                    + f"\n外部 LLM 调用失败，已回退到离线解释层。错误类型：{type(exc).__name__}。"
                )
    elif route.question_type == QuestionType.OUTCOME_CALCULATION:
        answer = _mock_outcome_answer(payload)
    elif route.question_type == QuestionType.RESEARCH_SUMMARY:
        answer = _mock_research_answer(context)
    elif route.question_type == QuestionType.EDUCATION:
        answer = _mock_education_answer(question)
    else:
        answer = (
            "我可以回答 Cobb 改善量、PSM、研究统计结论，或在提供结构化变量后解释探索性模型预测。"
            f"\n安全边界：{REQUIRED_SAFETY_PHRASE}"
        )

    payload["answer"] = answer
    payload["prompt_audit"] = {
        "aggregate_context_only": True,
        "excluded_fields": EXCLUDED_ROW_LEVEL_FIELDS,
        "contains_row_level_training_records": False,
        "prompt_preview": build_qa_prompt(payload)[:1800],
    }
    return payload
