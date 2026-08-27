from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class QuestionType(str, Enum):
    EDUCATION = "education"
    OUTCOME_CALCULATION = "outcome_calculation"
    PREDICTION = "prediction"
    RESEARCH_SUMMARY = "research_summary"
    UNSAFE_MEDICAL_ADVICE = "unsafe_medical_advice"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RouteResult:
    question_type: QuestionType
    reason: str


UNSAFE_PATTERNS = [
    "应该选",
    "该选",
    "推荐治疗",
    "治疗方案",
    "怎么治疗",
    "要不要做",
    "要不要选",
    "能不能停",
    "可以停",
    "替代医生",
    "诊断",
]

PREDICTION_PATTERNS = [
    "预测",
    "大概可能改善",
    "可能改善",
    "预计",
    "会改善多少",
    "能改善多少",
    "风险",
    "概率",
]

OUTCOME_PATTERNS = [
    "干预前",
    "干预后",
    "改善量",
    "改善了多少",
    "算不算改善",
]

EDUCATION_PATTERNS = [
    "是什么",
    "什么意思",
    "怎么理解",
    "cobb",
    "Cobb",
    "psm",
    "倾向评分",
    "rmse",
]

RESEARCH_PATTERNS = [
    "研究结论",
    "统计",
    "显著",
    "支具+训练是不是",
    "支具加训练是不是",
    "哪组更好",
    "psm结果",
    "总体",
]


def route_question(question: str, has_patient_payload: bool = False) -> RouteResult:
    text = question.strip()
    lowered = text.lower()
    if any(pattern in text for pattern in UNSAFE_PATTERNS):
        return RouteResult(
            QuestionType.UNSAFE_MEDICAL_ADVICE,
            "Question asks for treatment choice, diagnosis, or individualized medical advice.",
        )
    if has_patient_payload and any(pattern in text for pattern in PREDICTION_PATTERNS):
        return RouteResult(QuestionType.PREDICTION, "Patient payload and prediction intent detected.")
    if has_patient_payload and not any(pattern in text for pattern in OUTCOME_PATTERNS):
        return RouteResult(QuestionType.PREDICTION, "Patient payload provided; defaulting to prediction.")
    if any(pattern in text for pattern in PREDICTION_PATTERNS):
        return RouteResult(QuestionType.PREDICTION, "Prediction intent detected.")
    if any(pattern in text for pattern in EDUCATION_PATTERNS) or "cobb" in lowered:
        return RouteResult(QuestionType.EDUCATION, "Educational concept explanation intent detected.")
    if any(pattern in text for pattern in OUTCOME_PATTERNS):
        return RouteResult(QuestionType.OUTCOME_CALCULATION, "Outcome calculation or explanation intent detected.")
    if any(pattern in text for pattern in RESEARCH_PATTERNS):
        return RouteResult(QuestionType.RESEARCH_SUMMARY, "Research summary intent detected.")
    return RouteResult(QuestionType.UNKNOWN, "No confident route matched.")
