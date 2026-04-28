from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.config import ENABLE_TRANSFORMER_NLP, NLP_FALLBACK_MODEL, NLP_MODEL_FAMILY, NLP_PRIMARY_MODEL
from app.services.text import contains_phrase, normalize, tokenize


COLLOQUIAL_NORMALIZATION = {
    "ga": "tidak",
    "gak": "tidak",
    "ngga": "tidak",
    "nggak": "tidak",
    "engga": "tidak",
    "enggak": "tidak",
    "tdk": "tidak",
    "sy": "saya",
    "aq": "saya",
    "aku": "saya",
    "hrs": "harus",
    "krn": "karena",
    "bgt": "banget",
    "bgtt": "banget",
    "dgn": "dengan",
    "dlm": "dalam",
    "yg": "yang",
}

SYMPTOM_ALIASES = {
    "demam": ["demam", "panas badan", "badan panas", "meriang"],
    "demam tinggi mendadak": ["demam tinggi mendadak", "panas tinggi mendadak", "demam mendadak tinggi"],
    "sakit kepala": ["sakit kepala", "kepala sakit", "pusing", "kepala pusing"],
    "nyeri belakang mata": ["nyeri belakang mata", "sakit di belakang mata", "mata terasa nyeri"],
    "nyeri otot atau sendi": ["nyeri otot", "nyeri sendi", "badan pegal", "pegal linu", "ngilu sendi"],
    "menggigil": ["menggigil", "meriang menggigil", "kedinginan menggigil"],
    "berkeringat banyak": ["berkeringat banyak", "keringat banyak", "keringat dingin"],
    "mual": ["mual", "enek", "eneg"],
    "muntah": ["muntah", "muntah muntah"],
    "diare": ["diare", "mencret", "bab cair"],
    "nyeri perut": ["nyeri perut", "perut sakit", "mulas", "nyeri ulu hati", "perih lambung"],
    "ruam": ["ruam", "ruam merah", "bintik merah", "bentol", "biduran"],
    "perdarahan": ["mimisan", "gusi berdarah", "perdarahan", "bab berdarah", "bab hitam", "tinja hitam"],
    "batuk lama": ["batuk lama", "batuk 3 minggu", "batuk lebih dari 3 minggu", "batuk lebih 3 minggu"],
    "batuk darah": ["batuk darah", "dahak berdarah"],
    "keringat malam": ["keringat malam"],
    "berat badan turun": ["berat badan turun", "bb turun", "turun berat badan"],
    "sesak napas": ["sesak", "sesak napas", "susah napas", "sulit napas"],
    "mata atau kulit kuning": ["mata kuning", "kulit kuning", "kuning pada mata", "kuning pada kulit"],
    "bercak mati rasa": ["bercak mati rasa", "kulit mati rasa", "bercak kulit mati rasa"],
    "pembengkakan tungkai": ["kaki bengkak", "tungkai bengkak", "pembengkakan tungkai", "pembengkakan kaki"],
    "luka tanpa nyeri": ["luka tidak nyeri", "luka tanpa nyeri", "luka telapak kaki tidak terasa"],
    "sulit minum atau dehidrasi": ["dehidrasi", "sulit minum", "jarang kencing", "mulut sangat kering"],
    "kejang atau penurunan kesadaran": ["kejang", "pingsan", "penurunan kesadaran", "bingung berat"],
}

RISK_CONTEXT_ALIASES = {
    "riwayat daerah endemis malaria": ["daerah malaria", "endemis malaria", "perjalanan ke papua", "baru dari papua"],
    "paparan banjir atau tikus": ["banjir", "lumpur", "tikus", "air kotor", "leptospirosis"],
    "kontak erat tbc": ["kontak tbc", "serumah dengan tbc", "keluarga tbc"],
    "lingkungan banyak nyamuk": ["banyak nyamuk", "digigit nyamuk", "sarang nyamuk", "genangan air"],
    "makanan atau minuman kurang higienis": ["jajan sembarangan", "makanan tidak bersih", "minuman tidak bersih"],
    "kehamilan atau bayi": ["hamil", "bayi", "balita"],
}


@dataclass(frozen=True)
class SymptomMention:
    canonical: str
    alias: str
    negated: bool = False


@dataclass(frozen=True)
class ClinicalNLPResult:
    original_text: str
    normalized_text: str
    model_family: str
    primary_model: str
    fallback_model: str
    backend: str
    extracted_symptoms: list[str] = field(default_factory=list)
    negated_symptoms: list[str] = field(default_factory=list)
    risk_contexts: list[str] = field(default_factory=list)
    key_phrases: list[str] = field(default_factory=list)
    mentions: list[SymptomMention] = field(default_factory=list)
    confidence: float = 0.0

    def as_summary(self) -> dict[str, object]:
        return {
            "model_family": self.model_family,
            "primary_model": self.primary_model,
            "fallback_model": self.fallback_model,
            "backend": self.backend,
            "normalized_text": self.normalized_text,
            "extracted_symptoms": self.extracted_symptoms,
            "negated_symptoms": self.negated_symptoms,
            "risk_contexts": self.risk_contexts,
            "key_phrases": self.key_phrases,
            "confidence": self.confidence,
        }

    def retrieval_query(self) -> str:
        parts = [
            self.normalized_text,
            " ".join(self.extracted_symptoms),
            " ".join(self.risk_contexts),
            " ".join(self.key_phrases[:8]),
        ]
        return " ".join(part for part in parts if part.strip())


class IndonesianClinicalNLPProcessor:
    """IndoBERT/XLM-R compatible preprocessing layer for Indonesian symptoms.

    The prototype keeps this stage deterministic by default so it runs locally
    without downloading large transformer weights. When transformer inference is
    enabled in a prepared environment, this remains the integration point for
    IndoBERT/XLM-R token-classification while preserving the same output schema.
    """

    def __init__(
        self,
        *,
        model_family: str = NLP_MODEL_FAMILY,
        primary_model: str = NLP_PRIMARY_MODEL,
        fallback_model: str = NLP_FALLBACK_MODEL,
        enable_transformer: bool = ENABLE_TRANSFORMER_NLP,
    ):
        self.model_family = model_family
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.enable_transformer = enable_transformer
        self.backend = self._resolve_backend()

    def health(self) -> dict[str, object]:
        return {
            "model_family": self.model_family,
            "primary_model": self.primary_model,
            "fallback_model": self.fallback_model,
            "backend": self.backend,
            "transformer_enabled": self.enable_transformer,
            "task": "normalisasi_bahasa_indonesia_dan_ekstraksi_gejala",
        }

    def process(self, text: str) -> ClinicalNLPResult:
        normalized_text = self._normalize_indonesian(text)
        mentions = self._extract_symptom_mentions(normalized_text)
        risk_contexts = self._extract_risk_contexts(normalized_text)
        extracted = sorted({mention.canonical for mention in mentions if not mention.negated})
        negated = sorted({mention.canonical for mention in mentions if mention.negated})
        key_phrases = self._extract_key_phrases(normalized_text, extracted, risk_contexts)
        confidence = self._confidence_score(extracted, negated, risk_contexts)
        return ClinicalNLPResult(
            original_text=text,
            normalized_text=normalized_text,
            model_family=self.model_family,
            primary_model=self.primary_model,
            fallback_model=self.fallback_model,
            backend=self.backend,
            extracted_symptoms=extracted,
            negated_symptoms=negated,
            risk_contexts=risk_contexts,
            key_phrases=key_phrases,
            mentions=mentions,
            confidence=confidence,
        )

    def _resolve_backend(self) -> str:
        if not self.enable_transformer:
            return "deterministic_indobert_xlmr_compatible_extractor"
        try:
            __import__("transformers")
        except Exception:
            return "deterministic_fallback_transformers_not_installed"
        return "transformer_ready_indobert_xlmr_plus_rule_guardrails"

    def _normalize_indonesian(self, text: str) -> str:
        normalized = normalize(text)
        tokens = normalized.split()
        replaced = [COLLOQUIAL_NORMALIZATION.get(token, token) for token in tokens]
        normalized = " ".join(replaced)
        normalized = re.sub(r"\b(\d+)\s*minggu\b", r"\1 minggu", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def _extract_symptom_mentions(self, normalized_text: str) -> list[SymptomMention]:
        mentions: list[SymptomMention] = []
        for canonical, aliases in SYMPTOM_ALIASES.items():
            for alias in aliases:
                if contains_phrase(normalized_text, alias):
                    mentions.append(
                        SymptomMention(
                            canonical=canonical,
                            alias=alias,
                            negated=self._is_negated(normalized_text, alias),
                        )
                    )
                    break
        return mentions

    def _extract_risk_contexts(self, normalized_text: str) -> list[str]:
        contexts: list[str] = []
        for label, aliases in RISK_CONTEXT_ALIASES.items():
            if any(contains_phrase(normalized_text, alias) for alias in aliases):
                contexts.append(label)
        return sorted(set(contexts))

    def _extract_key_phrases(
        self,
        normalized_text: str,
        extracted_symptoms: list[str],
        risk_contexts: list[str],
    ) -> list[str]:
        phrases = [*extracted_symptoms, *risk_contexts]
        duration = _extract_duration_phrase(normalized_text)
        if duration:
            phrases.append(duration)
        return list(dict.fromkeys(phrases))

    @staticmethod
    def _is_negated(normalized_text: str, phrase: str) -> bool:
        return any(
            contains_phrase(normalized_text, f"{prefix} {phrase}")
            for prefix in ["tidak ada", "tidak", "tanpa", "bukan"]
        )

    @staticmethod
    def _confidence_score(
        extracted_symptoms: list[str],
        negated_symptoms: list[str],
        risk_contexts: list[str],
    ) -> float:
        signal_count = len(extracted_symptoms) + len(negated_symptoms) + len(risk_contexts)
        if signal_count == 0:
            return 0.15
        return round(min(0.95, 0.35 + signal_count * 0.12), 2)


def _extract_duration_phrase(normalized_text: str) -> str | None:
    patterns = [
        r"\b(?:sudah|selama|sejak|baru|sekitar|kurang lebih)\s+(?:[a-z]+\s+){0,3}?\d+\s*(?:jam|hari|minggu|bulan)\b",
        r"\b(?:sudah|selama|sejak|baru)\s+(?:tadi\s+(?:pagi|siang|sore|malam)|kemarin|semalam|tadi)\b",
        r"\b\d+\s*(?:jam|hari|minggu|bulan)\b",
        r"\bhari\s+ke\s+\d+\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized_text)
        if match:
            return match.group(0).strip()
    return None
