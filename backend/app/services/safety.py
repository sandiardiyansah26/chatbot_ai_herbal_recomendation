from __future__ import annotations

from dataclasses import dataclass, field

from app.models import ModelAssessment
from app.services.knowledge_base import KnowledgeChunk, TrainingRecord
from app.services.retrieval import RetrievedItem
from app.services.text import normalize


HIGH_RISK_TROPICAL_TERMS = {
    "dbd",
    "demam berdarah",
    "dengue",
    "malaria",
    "tuberkulosis",
    "tbc",
    "leptospirosis",
    "kusta",
    "filariasis",
    "kaki gajah",
    "tifoid",
    "tipes",
}


@dataclass(frozen=True)
class SafetyDecision:
    level: str
    red_flags: list[str] = field(default_factory=list)
    referral_reasons: list[str] = field(default_factory=list)
    requires_emergency_referral: bool = False
    requires_medical_referral: bool = False
    allow_herbal_support: bool = True
    disclaimer: str = (
        "Informasi ini adalah edukasi dan triase awal, bukan diagnosis medis final atau pengganti tenaga kesehatan."
    )

    def as_summary(self) -> dict[str, object]:
        return {
            "level": self.level,
            "red_flags": self.red_flags,
            "referral_reasons": self.referral_reasons,
            "requires_emergency_referral": self.requires_emergency_referral,
            "requires_medical_referral": self.requires_medical_referral,
            "allow_herbal_support": self.allow_herbal_support,
            "disclaimer": self.disclaimer,
        }


def evaluate_safety(
    *,
    red_flags: list[str],
    assessment: ModelAssessment | None = None,
    retrieved_items: list[RetrievedItem] | None = None,
    nlp_summary: dict[str, object] | None = None,
) -> SafetyDecision:
    flags = list(dict.fromkeys(red_flags))
    reasons: list[str] = []

    if assessment:
        flags.extend(flag for flag in assessment.red_flags if flag not in flags)
        if assessment.need_medical_referral:
            reasons.append("model menandai perlunya rujukan medis")
        if assessment.scope in {"critical", "internal_medicine"}:
            reasons.append(f"scope model: {assessment.scope}")

    suspected_text = " ".join(assessment.suspected_conditions if assessment else [])
    context_text = _context_titles(retrieved_items or [])
    nlp_text = " ".join(
        [
            " ".join(_summary_list(nlp_summary or {}, "extracted_symptoms")),
            " ".join(_summary_list(nlp_summary or {}, "risk_contexts")),
        ]
    )
    combined = normalize(" ".join([suspected_text, context_text, nlp_text]))

    matched_high_risk = sorted(term for term in HIGH_RISK_TROPICAL_TERMS if term in combined)
    if matched_high_risk:
        reasons.append("konteks mengarah ke penyakit tropis yang perlu konfirmasi medis: " + ", ".join(matched_high_risk[:4]))

    if flags:
        return SafetyDecision(
            level="emergency",
            red_flags=flags,
            referral_reasons=reasons or ["tanda bahaya terdeteksi pada anamnesis"],
            requires_emergency_referral=True,
            requires_medical_referral=True,
            allow_herbal_support=False,
        )

    if assessment and (assessment.need_medical_referral or assessment.scope in {"critical", "internal_medicine"}):
        return SafetyDecision(
            level="referral",
            red_flags=flags,
            referral_reasons=reasons,
            requires_medical_referral=True,
            allow_herbal_support=False,
        )

    if matched_high_risk:
        return SafetyDecision(
            level="caution",
            red_flags=flags,
            referral_reasons=reasons,
            requires_medical_referral=False,
            allow_herbal_support=True,
        )

    return SafetyDecision(level="self_care", red_flags=flags, allow_herbal_support=True)


def safety_assessment_to_model_assessment(decision: SafetyDecision, fallback: ModelAssessment | None = None) -> ModelAssessment:
    if fallback:
        suspected = fallback.suspected_conditions
        reasoning = fallback.reasoning or fallback.scope_reason
    else:
        suspected = []
        reasoning = ""
    return ModelAssessment(
        scope="critical" if decision.requires_emergency_referral else "internal_medicine",
        scope_reason="; ".join(decision.referral_reasons) or reasoning or "Safety layer menilai perlu rujukan medis.",
        suspected_conditions=suspected,
        reasoning=reasoning,
        red_flags=decision.red_flags,
        need_medical_referral=True,
        warning_notes=(
            "Segera cari bantuan medis bila ada tanda bahaya. Herbal tidak diposisikan sebagai terapi utama pada kondisi ini."
        ),
    )


def _context_titles(items: list[RetrievedItem]) -> str:
    titles: list[str] = []
    for item in items[:8]:
        titles.append(item.title)
        payload = item.payload
        if isinstance(payload, KnowledgeChunk):
            payload = payload.payload
        if isinstance(payload, TrainingRecord):
            titles.append(payload.topic)
            titles.extend(payload.warning_signs[:3])
    return " ".join(titles)


def _summary_list(summary: dict[str, object], key: str) -> list[str]:
    values = summary.get(key)
    if not isinstance(values, list):
        return []
    return [str(value).strip() for value in values if str(value).strip()]
