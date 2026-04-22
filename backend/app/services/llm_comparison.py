from __future__ import annotations

import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.config import (
    ENABLE_DUAL_LLM_COMPARISON,
    LEARNING_LOG_PATH,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL_A,
    OLLAMA_MODEL_A_FALLBACKS,
    OLLAMA_MODEL_B,
    OLLAMA_MODEL_B_FALLBACKS,
    OLLAMA_NUM_PREDICT_DEFAULT,
    OLLAMA_NUM_PREDICT_FOLLOW_UP,
    OLLAMA_NUM_PREDICT_OUT_OF_SCOPE,
    OLLAMA_NUM_PREDICT_RECOMMENDATION,
    OLLAMA_NUM_PREDICT_RED_FLAG,
    OLLAMA_TIMEOUT_SECONDS,
)
from app.models import ModelCandidate, ModelComparison, Recommendation, RetrievedContext
from app.services.recommendation import DISCLAIMER
from app.services.text import normalize, tokenize


SYSTEM_PROMPT = (
    "Anda adalah kandidat model GenAI untuk chatbot rekomendasi ramuan herbal berbahasa Indonesia. "
    "Gunakan hanya konteks RAG dan data anamnesis yang diberikan. Jangan membuat diagnosis medis final. "
    "Jangan mengklaim ramuan menyembuhkan penyakit. Jika ada tanda bahaya, arahkan ke tenaga kesehatan. "
    "Jika konteks cukup dan keluhan ringan, jawab dengan ramuan, bahan, cara pengolahan, dosis/kisaran, "
    "catatan kewaspadaan, dan disclaimer."
)
COMPARISON_RESPONSE_TYPES = {"recommendation", "follow_up"}


class DualLLMComparator:
    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        models: tuple[str, str] = (OLLAMA_MODEL_A, OLLAMA_MODEL_B),
        fallbacks: dict[str, list[str]] | None = None,
        timeout_seconds: float = OLLAMA_TIMEOUT_SECONDS,
        learning_log_path: Path = LEARNING_LOG_PATH,
        enabled: bool = ENABLE_DUAL_LLM_COMPARISON,
        num_predict_by_response_type: dict[str, int] | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.models = [model for model in models if model]
        fallback_map = fallbacks or {
            OLLAMA_MODEL_A: OLLAMA_MODEL_A_FALLBACKS,
            OLLAMA_MODEL_B: OLLAMA_MODEL_B_FALLBACKS,
        }
        self.fallbacks = {
            model: [fallback for fallback in fallback_models if fallback and fallback != model]
            for model, fallback_models in fallback_map.items()
        }
        self.timeout_seconds = timeout_seconds
        self.learning_log_path = learning_log_path
        self.enabled = enabled and len(self.models) >= 2
        defaults = {
            "default": OLLAMA_NUM_PREDICT_DEFAULT,
            "recommendation": OLLAMA_NUM_PREDICT_RECOMMENDATION,
            "follow_up": OLLAMA_NUM_PREDICT_FOLLOW_UP,
            "red_flag": OLLAMA_NUM_PREDICT_RED_FLAG,
            "out_of_scope": OLLAMA_NUM_PREDICT_OUT_OF_SCOPE,
        }
        overrides = num_predict_by_response_type or {}
        self.num_predict_by_response_type = {
            name: max(int(overrides.get(name, value)), 1)
            for name, value in defaults.items()
        }

    def health(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "base_url": self.base_url,
            "models": self.models,
            "fallbacks": self.fallbacks,
            "timeout_seconds": self.timeout_seconds,
            "response_types": sorted(COMPARISON_RESPONSE_TYPES),
            "num_predict_by_response_type": self.num_predict_by_response_type,
            "learning_log_path": str(self.learning_log_path),
        }

    def compare(
        self,
        *,
        session_id: str,
        user_message: str,
        response_type: str,
        baseline_reply: str,
        recommendation: Recommendation | None,
        retrieved_context: list[RetrievedContext],
        anamnesis_summary: dict[str, object],
        red_flags: list[str],
        follow_up_question: str | None,
    ) -> ModelComparison:
        if not self.enabled:
            return ModelComparison(enabled=False, note="Dual LLM comparison disabled or less than two models configured.")
        if not self._should_compare(response_type):
            return ModelComparison(
                enabled=False,
                note=f"Dual LLM comparison dilewati untuk response_type '{response_type}' agar guardrail tetap cepat dan deterministik.",
            )

        num_predict = self._num_predict_for(response_type)
        prompt = build_prompt(
            user_message=user_message,
            response_type=response_type,
            baseline_reply=baseline_reply,
            recommendation=recommendation,
            retrieved_context=retrieved_context,
            anamnesis_summary=anamnesis_summary,
            red_flags=red_flags,
            follow_up_question=follow_up_question,
        )

        futures = {}
        ordered_candidates: dict[str, ModelCandidate] = {}
        with ThreadPoolExecutor(max_workers=min(len(self.models), 4)) as executor:
            for model in self.models:
                futures[
                    executor.submit(
                        self._generate_and_score,
                        model=model,
                        prompt=prompt,
                        response_type=response_type,
                        recommendation=recommendation,
                        retrieved_context=retrieved_context,
                        red_flags=red_flags,
                        num_predict=num_predict,
                    )
                ] = model

            for future in as_completed(futures):
                model = futures[future]
                try:
                    ordered_candidates[model] = future.result()
                except Exception as error:  # pragma: no cover - defensive safety net.
                    ordered_candidates[model] = ModelCandidate(
                        model=model,
                        status="error",
                        score=0.0,
                        error=str(error),
                    )

        candidates = [
            ordered_candidates.get(model)
            or ModelCandidate(model=model, status="error", score=0.0, error="Candidate generation did not complete.")
            for model in self.models
        ]

        usable = [candidate for candidate in candidates if candidate.status == "ok" and candidate.reply.strip()]
        selected = max(usable, key=lambda item: item.score) if usable else None
        log_id = self._write_learning_log(
            {
                "id": str(uuid.uuid4()),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "session_id": session_id,
                "user_message": user_message,
                "response_type": response_type,
                "prompt": prompt,
                "baseline_reply": baseline_reply,
                "selected_model": selected.model if selected else None,
                "selected_reply": selected.reply if selected else None,
                "candidates": [candidate.model_dump() for candidate in candidates],
            }
        )

        note = None
        if not selected:
            note = (
                "Kandidat Ollama tidak menghasilkan jawaban final dalam budget token saat ini. "
                "Coba naikkan `OLLAMA_NUM_PREDICT_RECOMMENDATION` atau gunakan model yang lebih ringan "
                "bila ingin komparasi tetap aktif."
            )

        return ModelComparison(
            enabled=True,
            selected_model=selected.model if selected else None,
            selected_reply=selected.reply if selected else None,
            learning_log_id=log_id,
            candidates=candidates,
            note=note,
        )

    def _generate_and_score(
        self,
        *,
        model: str,
        prompt: str,
        response_type: str,
        recommendation: Recommendation | None,
        retrieved_context: list[RetrievedContext],
        red_flags: list[str],
        num_predict: int,
    ) -> ModelCandidate:
        started = time.perf_counter()
        attempted_models = [model, *self.fallbacks.get(model, [])]
        errors: list[str] = []

        for candidate_model in attempted_models:
            try:
                reply = self._chat(candidate_model, prompt, num_predict=num_predict)
                latency_ms = int((time.perf_counter() - started) * 1000)
                score, breakdown = score_reply(
                    reply=reply,
                    response_type=response_type,
                    recommendation=recommendation,
                    retrieved_context=retrieved_context,
                    red_flags=red_flags,
                )
                return ModelCandidate(
                    model=candidate_model,
                    status="ok",
                    reply=reply,
                    score=score,
                    latency_ms=latency_ms,
                    scoring_breakdown=breakdown,
                )
            except Exception as error:  # pragma: no cover - depends on local Ollama state.
                errors.append(f"{candidate_model}: {error}")
                if not is_missing_model_error(error):
                    break

        latency_ms = int((time.perf_counter() - started) * 1000)
        return ModelCandidate(
            model=model,
            status="error",
            score=0.0,
            latency_ms=latency_ms,
            error=" | ".join(errors),
        )

    def _chat(self, model: str, prompt: str, *, num_predict: int) -> str:
        payload = {
            "model": model,
            "stream": False,
            "think": False,
            "keep_alive": "15m",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "options": {
                "temperature": 0.2,
                "top_p": 0.85,
                "num_predict": num_predict,
            },
        }
        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP {error.code}: {body}") from error
        except URLError as error:
            raise RuntimeError(f"Ollama tidak dapat dihubungi di {self.base_url}: {error}") from error

        message = data.get("message") or {}
        content = str(message.get("content") or "").strip()
        thinking = str(message.get("thinking") or "").strip()
        if not content:
            if thinking:
                raise RuntimeError(
                    f"Ollama model {model} hanya mengembalikan thinking tanpa jawaban final. "
                    "Pastikan request memakai `think=false` atau naikkan budget token bila model tetap terpotong."
                )
            raise RuntimeError(f"Ollama model {model} tidak mengembalikan konten.")
        return strip_thinking(content)

    def _write_learning_log(self, row: dict[str, object]) -> str:
        self.learning_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.learning_log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
        return str(row["id"])

    def _num_predict_for(self, response_type: str) -> int:
        return self.num_predict_by_response_type.get(
            response_type,
            self.num_predict_by_response_type["default"],
        )

    @staticmethod
    def _should_compare(response_type: str) -> bool:
        return response_type in COMPARISON_RESPONSE_TYPES


def build_prompt(
    *,
    user_message: str,
    response_type: str,
    baseline_reply: str,
    recommendation: Recommendation | None,
    retrieved_context: list[RetrievedContext],
    anamnesis_summary: dict[str, object],
    red_flags: list[str],
    follow_up_question: str | None,
) -> str:
    recommendation_text = "-"
    if recommendation:
        recommendation_text = "\n".join(
            [
                f"Keluhan ringan: {recommendation.keluhan_ringan}",
                f"Ramuan: {recommendation.ramuan}",
                f"Bahan: {', '.join(recommendation.bahan)}",
                f"Cara pengolahan: {recommendation.cara_pengolahan}",
                f"Dosis: {recommendation.dosis_penggunaan}",
                f"Kewaspadaan: {recommendation.catatan_kewaspadaan}",
                f"Sumber: {recommendation.sumber_ringkas}",
            ]
        )

    contexts = "\n".join(
        f"- {item.type}:{item.title} | score={item.score} | evidence={item.evidence_level or '-'} | source={item.source or '-'}"
        for item in retrieved_context[:5]
    ) or "-"
    baseline = _trim_text(baseline_reply, max_length=900)

    return (
        "Tugas: buat respons chatbot herbal terbaik berdasarkan konteks berikut.\n\n"
        f"Input pengguna:\n{user_message}\n\n"
        f"Tipe respons sistem: {response_type}\n"
        f"Ringkasan anamnesis: {json.dumps(anamnesis_summary, ensure_ascii=False)}\n"
        f"Red flag terdeteksi: {', '.join(red_flags) if red_flags else '-'}\n"
        f"Pertanyaan follow-up bila ada:\n{follow_up_question or '-'}\n\n"
        f"Data rekomendasi terstruktur:\n{recommendation_text}\n\n"
        f"Konteks RAG:\n{contexts}\n\n"
        f"Baseline jawaban sistem:\n{baseline}\n\n"
        "Instruksi jawaban:\n"
        "- Jawab dalam Bahasa Indonesia yang natural dan interaktif.\n"
        "- Jika perlu anamnesis lanjutan, berikan pertanyaan bertahap yang relevan.\n"
        "- Jika rekomendasi herbal aman diberikan, sebutkan ramuan, bahan, cara pengolahan, dosis, dan kewaspadaan.\n"
        "- Jangan mengarang ramuan/dosis di luar konteks.\n"
        "- Jangan memberi diagnosis final atau klaim sembuh/pasti.\n"
        f"- Akhiri dengan disclaimer singkat: {DISCLAIMER}"
    )


def _trim_text(text: str, *, max_length: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_length:
        return compact
    return f"{compact[: max_length - 3].rstrip()}..."


def score_reply(
    *,
    reply: str,
    response_type: str,
    recommendation: Recommendation | None,
    retrieved_context: list[RetrievedContext],
    red_flags: list[str],
) -> tuple[float, dict[str, float]]:
    normalized = normalize(reply)
    breakdown = {
        "safety": safety_score(normalized, response_type, red_flags),
        "grounding": grounding_score(normalized, recommendation, retrieved_context),
        "completeness": completeness_score(normalized, response_type, recommendation),
        "language": language_score(reply),
        "penalty": penalty_score(normalized),
    }
    total = (
        0.34 * breakdown["safety"]
        + 0.3 * breakdown["grounding"]
        + 0.24 * breakdown["completeness"]
        + 0.12 * breakdown["language"]
        - breakdown["penalty"]
    )
    return round(max(0.0, min(1.0, total)), 4), {key: round(value, 4) for key, value in breakdown.items()}


def safety_score(normalized_reply: str, response_type: str, red_flags: list[str]) -> float:
    score = 0.35
    if "bukan diagnosis" in normalized_reply or "bukan pengganti" in normalized_reply:
        score += 0.25
    if "tenaga kesehatan" in normalized_reply or "fasilitas kesehatan" in normalized_reply:
        score += 0.18
    if response_type == "red_flag" and red_flags and any(normalize(flag) in normalized_reply for flag in red_flags):
        score += 0.12
    if "tanda bahaya" in normalized_reply or "kewaspadaan" in normalized_reply:
        score += 0.1
    return min(1.0, score)


def grounding_score(
    normalized_reply: str,
    recommendation: Recommendation | None,
    retrieved_context: list[RetrievedContext],
) -> float:
    expected_terms: set[str] = set()
    if recommendation:
        expected_terms.update(tokenize(recommendation.keluhan_ringan))
        expected_terms.update(tokenize(recommendation.ramuan))
        for ingredient in recommendation.bahan:
            expected_terms.update(tokenize(ingredient))
    for context in retrieved_context[:5]:
        expected_terms.update(tokenize(context.title))
    if not expected_terms:
        return 0.65

    matched = [term for term in expected_terms if term in normalized_reply]
    return min(1.0, len(matched) / max(8, len(expected_terms) * 0.55))


def completeness_score(normalized_reply: str, response_type: str, recommendation: Recommendation | None) -> float:
    if response_type == "follow_up":
        signals = ["sejak kapan", "demam", "sesak", "tanda bahaya", "ruam", "nyeri", "muntah"]
        return min(1.0, sum(1 for signal in signals if signal in normalized_reply) / 4)
    if response_type == "red_flag":
        signals = ["segera", "tenaga kesehatan", "fasilitas kesehatan", "tanda bahaya"]
        return min(1.0, sum(1 for signal in signals if signal in normalized_reply) / 3)
    if recommendation:
        signals = ["ramuan", "bahan", "cara", "dosis", "kewaspadaan", "sumber"]
        return min(1.0, sum(1 for signal in signals if signal in normalized_reply) / 5)
    return 0.55


def language_score(reply: str) -> float:
    length = len(reply.strip())
    if 280 <= length <= 1800:
        return 1.0
    if 120 <= length < 280 or 1800 < length <= 2600:
        return 0.75
    return 0.45


def penalty_score(normalized_reply: str) -> float:
    risky_phrases = [
        "pasti sembuh",
        "menyembuhkan",
        "obat utama",
        "tidak perlu dokter",
        "abaikan dokter",
        "diagnosis anda",
        "jawab dalam bahasa indonesia",
        "instruksi jawaban",
        "baseline jawaban",
        "konteks rag",
    ]
    return min(0.5, 0.12 * sum(1 for phrase in risky_phrases if phrase in normalized_reply))


def is_missing_model_error(error: Exception) -> bool:
    message = str(error).lower()
    return "http 404" in message and "not found" in message


def strip_thinking(content: str) -> str:
    normalized = content.strip()
    while True:
        lower = normalized.lower()
        if "<think>" not in lower or "</think>" not in lower:
            break
        start = lower.find("<think>")
        end = lower.find("</think>") + len("</think>")
        normalized = (normalized[:start] + normalized[end:]).strip()

    while True:
        lower = normalized.lower()
        if "<|channel>thought" not in lower or "<channel|>" not in lower:
            break
        start = lower.find("<|channel>thought")
        end = lower.find("<channel|>", start) + len("<channel|>")
        normalized = (normalized[:start] + normalized[end:]).strip()
    return normalized
