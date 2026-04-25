from __future__ import annotations

import json
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.config import (
    ENABLE_DUAL_LLM_COMPARISON,
    ENABLE_OPENAI_COMPARISON,
    LEARNING_LOG_PATH,
    MAX_ANAMNESIS_QUESTIONS,
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
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    OPENAI_TIMEOUT_SECONDS,
)
from app.models import ModelAssessment, ModelCandidate, ModelComparison, Recommendation, RetrievedContext
from app.services.anamnesis import SYMPTOM_CONCEPTS
from app.services.recommendation import DISCLAIMER
from app.services.text import contains_phrase, normalize, tokenize


SYSTEM_PROMPT = (
    "Anda adalah kandidat model GenAI untuk chatbot deteksi penyakit berbasis humanizing anamnesis berbahasa Indonesia. "
    "Tugas Anda adalah membantu menggali keluhan, mengusulkan dugaan kondisi, menyusun maksimal satu pertanyaan follow-up "
    "yang paling relevan, dan bila informasi cukup menyusun jawaban akhir yang aman. "
    "Gunakan hanya riwayat percakapan, ringkasan anamnesis, dan konteks retrieval yang diberikan. "
    "Jangan memberi diagnosis medis final. Jangan mengarang ramuan yang tidak ada di konteks retrieval. "
    "Jika kondisi mengarah ke red flag, penyakit kritis, atau penyakit dalam, arahkan ke tenaga kesehatan dan jangan "
    "memaksakan rekomendasi herbal. Output HARUS berupa satu object JSON valid tanpa markdown, penjelasan tambahan, atau teks di luar JSON."
)
COMPARISON_RESPONSE_TYPES = {"follow_up", "recommendation"}
SUPPORTED_SCOPES = {"supported", "internal_medicine", "critical", "unsupported"}
FOLLOW_UP_DETAIL_TERMS = (
    "berapa",
    "suhu",
    "derajat",
    "tinggi",
    "pola",
    "naik turun",
    "menetap",
    "mendadak",
    "seberapa",
    "skala",
    "parah",
    "berat",
    "lokasi",
    "bagian",
    "warna",
    "luas",
    "menyebar",
    "melepuh",
    "memburuk",
)
FOLLOW_UP_DURATION_TERMS = (
    "sejak kapan",
    "berapa lama",
    "sudah berapa",
    "kapan mulai",
    "mulai kapan",
    "durasi",
)
FOLLOW_UP_GENERIC_SYMPTOM_TERMS = (
    "gejala lain",
    "gejala selain",
    "keluhan lain",
    "keluhan selain",
)
OLLAMA_DURATION_FIELDS = (
    "total_duration",
    "load_duration",
    "prompt_eval_duration",
    "eval_duration",
)
OLLAMA_COUNT_FIELDS = (
    "prompt_eval_count",
    "eval_count",
)
OPENAI_USAGE_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
)


class DualLLMComparator:
    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        models: tuple[str, ...] | None = None,
        fallbacks: dict[str, list[str]] | None = None,
        timeout_seconds: float = OLLAMA_TIMEOUT_SECONDS,
        openai_base_url: str = OPENAI_BASE_URL,
        openai_api_key: str = OPENAI_API_KEY,
        openai_model: str = OPENAI_MODEL,
        openai_timeout_seconds: float = OPENAI_TIMEOUT_SECONDS,
        enable_openai: bool = ENABLE_OPENAI_COMPARISON,
        learning_log_path: Path = LEARNING_LOG_PATH,
        enabled: bool = ENABLE_DUAL_LLM_COMPARISON,
        num_predict_by_response_type: dict[str, int] | None = None,
        max_questions: int = MAX_ANAMNESIS_QUESTIONS,
    ):
        self.base_url = base_url.rstrip("/")
        self.openai_base_url = openai_base_url.rstrip("/")
        self.openai_api_key = openai_api_key
        self.openai_model = openai_model.strip()
        self.openai_timeout_seconds = openai_timeout_seconds
        self.openai_available = bool(enable_openai and self.openai_api_key and self.openai_model)
        configured_models = list(models or (OLLAMA_MODEL_A, OLLAMA_MODEL_B))
        if self.openai_available and self.openai_model not in configured_models:
            configured_models.append(self.openai_model)
        self.models = [model for model in configured_models if model]
        self.openai_models = {self.openai_model} if self.openai_available else set()
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
        self.max_questions = max_questions
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
            "providers": {model: self._provider_for_model(model) for model in self.models},
            "fallbacks": self.fallbacks,
            "openai": {
                "enabled": self.openai_available,
                "model": self.openai_model,
                "base_url": self.openai_base_url,
                "api_key_present": bool(self.openai_api_key),
                "timeout_seconds": self.openai_timeout_seconds,
            },
            "timeout_seconds": self.timeout_seconds,
            "response_types": sorted(COMPARISON_RESPONSE_TYPES),
            "num_predict_by_response_type": self.num_predict_by_response_type,
            "learning_log_path": str(self.learning_log_path),
            "max_questions": self.max_questions,
            "mode": "multi_llm_anamnesis_first_structured_json",
        }

    def generate_assessment(
        self,
        *,
        session_id: str,
        user_message: str,
        response_type: str,
        conversation_history: list[dict[str, str]],
        retrieved_context: list[RetrievedContext],
        anamnesis_summary: dict[str, object],
        red_flags: list[str],
        question_count: int,
        force_final: bool = False,
    ) -> ModelComparison:
        if not self.enabled:
            return ModelComparison(enabled=False, note="LLM comparison disabled or less than two models configured.")
        if not self._should_compare(response_type):
            return ModelComparison(
                enabled=False,
                note=f"LLM comparison dilewati untuk response_type '{response_type}'.",
            )

        prompt = build_medical_prompt(
            user_message=user_message,
            response_type=response_type,
            conversation_history=conversation_history,
            retrieved_context=retrieved_context,
            anamnesis_summary=anamnesis_summary,
            red_flags=red_flags,
            question_count=question_count,
            max_questions=self.max_questions,
            force_final=force_final,
        )
        num_predict = self._num_predict_for(response_type)

        ordered_candidates: dict[str, ModelCandidate] = {}
        run_in_parallel = response_type != "recommendation"
        if run_in_parallel:
            futures = {}
            with ThreadPoolExecutor(max_workers=min(len(self.models), 4)) as executor:
                for model in self.models:
                    futures[
                        executor.submit(
                            self._generate_assessment_for_model,
                            model=model,
                            prompt=prompt,
                            response_type=response_type,
                            retrieved_context=retrieved_context,
                            anamnesis_summary=anamnesis_summary,
                            user_message=user_message,
                            question_count=question_count,
                            force_final=force_final,
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
                            provider=self._provider_for_model(model),
                            status="error",
                            score=0.0,
                            error=str(error),
                        )
        else:
            for model in self.models:
                try:
                    ordered_candidates[model] = self._generate_assessment_for_model(
                        model=model,
                        prompt=prompt,
                        response_type=response_type,
                        retrieved_context=retrieved_context,
                        anamnesis_summary=anamnesis_summary,
                        user_message=user_message,
                        question_count=question_count,
                        force_final=force_final,
                        num_predict=num_predict,
                    )
                except Exception as error:  # pragma: no cover - defensive safety net.
                    ordered_candidates[model] = ModelCandidate(
                        model=model,
                        provider=self._provider_for_model(model),
                        status="error",
                        score=0.0,
                        error=str(error),
                    )

        candidates = [
            ordered_candidates.get(model)
            or ModelCandidate(
                model=model,
                provider=self._provider_for_model(model),
                status="error",
                score=0.0,
                error="Candidate generation did not complete.",
            )
            for model in self.models
        ]
        usable = [
            candidate
            for candidate in candidates
            if candidate.status == "ok" and candidate.assessment is not None
        ]
        selected = max(usable, key=lambda item: item.score) if usable else None
        log_id = self._write_learning_log(
            {
                "id": str(uuid.uuid4()),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "session_id": session_id,
                "user_message": user_message,
                "response_type": response_type,
                "prompt": prompt,
                "question_count": question_count,
                "force_final": force_final,
                "execution_mode": "parallel" if run_in_parallel else "sequential",
                "selected_model": selected.model if selected else None,
                "selected_reply": selected.reply if selected else None,
                "selected_assessment": selected.assessment.model_dump() if selected and selected.assessment else None,
                "candidates": [candidate.model_dump() for candidate in candidates],
            }
        )

        note = None
        if not selected:
            note = (
                "Belum ada model yang menghasilkan struktur anamnesis valid. "
                "Periksa model Ollama/OpenAI atau turunkan kompleksitas prompt."
            )

        return ModelComparison(
            enabled=True,
            selected_model=selected.model if selected else None,
            selected_reply=selected.reply if selected else None,
            selected_assessment=selected.assessment if selected else None,
            learning_log_id=log_id,
            candidates=candidates,
            note=note,
        )

    def _generate_assessment_for_model(
        self,
        *,
        model: str,
        prompt: str,
        response_type: str,
        retrieved_context: list[RetrievedContext],
        anamnesis_summary: dict[str, object],
        user_message: str,
        question_count: int,
        force_final: bool,
        num_predict: int,
    ) -> ModelCandidate:
        started = time.perf_counter()
        attempted_models = [model, *self.fallbacks.get(model, [])]
        errors: list[str] = []

        for candidate_model in attempted_models:
            try:
                raw_reply, inference_metrics = self._chat(candidate_model, prompt, num_predict=num_predict)
                assessment = parse_model_assessment(raw_reply)
                latency_ms = int((time.perf_counter() - started) * 1000)
                score, breakdown = score_assessment(
                    assessment=assessment,
                    response_type=response_type,
                    retrieved_context=retrieved_context,
                    anamnesis_summary=anamnesis_summary,
                    user_message=user_message,
                    question_count=question_count,
                    max_questions=self.max_questions,
                    force_final=force_final,
                )
                return ModelCandidate(
                    model=candidate_model,
                    provider=self._provider_for_model(candidate_model),
                    status="ok",
                    reply=render_candidate_reply(assessment, response_type),
                    score=score,
                    latency_ms=latency_ms,
                    scoring_breakdown=breakdown,
                    inference_metrics=inference_metrics,
                    assessment=assessment,
                )
            except Exception as error:  # pragma: no cover - depends on local Ollama state.
                errors.append(f"{candidate_model}: {error}")
                if not is_missing_model_error(error):
                    break

        latency_ms = int((time.perf_counter() - started) * 1000)
        return ModelCandidate(
            model=model,
            provider=self._provider_for_model(model),
            status="error",
            score=0.0,
            latency_ms=latency_ms,
            error=" | ".join(errors),
        )

    def _chat(self, model: str, prompt: str, *, num_predict: int) -> tuple[str, dict[str, object]]:
        if self._provider_for_model(model) == "openai":
            return self._chat_openai(model, prompt, num_predict=num_predict)
        return self._chat_ollama(model, prompt, num_predict=num_predict)

    def _chat_ollama(self, model: str, prompt: str, *, num_predict: int) -> tuple[str, dict[str, object]]:
        payload = {
            "model": model,
            "stream": False,
            "think": False,
            "keep_alive": "15m",
            "format": "json",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "options": {
                "temperature": 0.15,
                "top_p": 0.8,
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

        inference_metrics = extract_ollama_inference_metrics(data)
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
        return strip_thinking(content), inference_metrics

    def _chat_openai(self, model: str, prompt: str, *, num_predict: int) -> tuple[str, dict[str, object]]:
        if not self.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY belum diset, kandidat GPT-4 tidak dapat dipanggil.")

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.15,
            "top_p": 0.8,
            "max_tokens": num_predict,
            "response_format": {"type": "json_object"},
        }
        request = Request(
            f"{self.openai_base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.openai_timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI HTTP {error.code}: {body}") from error
        except URLError as error:
            raise RuntimeError(f"OpenAI tidak dapat dihubungi di {self.openai_base_url}: {error}") from error

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"OpenAI model {model} tidak mengembalikan choices.")
        inference_metrics = extract_openai_inference_metrics(data)
        message = choices[0].get("message") or {}
        content = _content_to_text(message.get("content")).strip()
        if not content:
            raise RuntimeError(f"OpenAI model {model} tidak mengembalikan konten.")
        return strip_thinking(content), inference_metrics

    def _provider_for_model(self, model: str) -> str:
        return "openai" if model in self.openai_models else "ollama"

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


def build_medical_prompt(
    *,
    user_message: str,
    response_type: str,
    conversation_history: list[dict[str, str]],
    retrieved_context: list[RetrievedContext],
    anamnesis_summary: dict[str, object],
    red_flags: list[str],
    question_count: int,
    max_questions: int,
    force_final: bool,
) -> str:
    conversation = "\n".join(
        f"{turn.get('role', 'user')}: {turn.get('content', '').strip()}"
        for turn in conversation_history[-4:]
        if turn.get("content")
    ) or f"user: {user_message}"
    contexts = "\n".join(
        f"- {item.type}:{item.title} | ev={item.evidence_level or '-'}"
        for item in retrieved_context[:4]
    ) or "-"
    mode = "final_answer" if response_type == "recommendation" or force_final else "anamnesis_follow_up"
    answered_context = _answered_context_for_prompt(anamnesis_summary)
    asked_questions = _summary_list(anamnesis_summary, "asked_follow_up_questions")

    if mode == "anamnesis_follow_up":
        schema = {
            "scope": "supported|internal|critical|unsupported",
            "why": "alasan singkat",
            "suspected": ["maksimal 3 dugaan kondisi"],
            "enough": False,
            "question": "isi tepat satu pertanyaan lanjutan yang paling penting",
            "question_why": "mengapa pertanyaan itu dipilih",
            "red_flags": ["daftar red flag bila ada"],
            "refer": False,
        }
    else:
        schema = {
            "scope": "supported|internal|critical|unsupported",
            "why": "alasan singkat",
            "suspected": ["maksimal 3 dugaan kondisi"],
            "red_flags": ["daftar red flag bila ada"],
            "refer": False,
            "final": "jawaban akhir singkat dan aman untuk user",
            "herbal": "nama ramuan dari konteks retrieval bila ada, selain itu string kosong",
            "ingredients": ["bahan ramuan"],
            "prep": "cara pengolahan singkat",
            "dose": "dosis/kisaran penggunaan",
            "warn": "kewaspadaan utama",
            "source": "judul sumber/context utama yang paling relevan",
        }

    return (
        "Analisis percakapan medis ringan dengan metode humanizing anamnesis.\n\n"
        f"Mode: {mode}\n"
        f"Pertanyaan anamnesis yang sudah ditanyakan: {question_count} dari maksimum {max_questions}\n"
        f"Paksa jawaban final: {'ya' if force_final else 'tidak'}\n"
        f"Red flag heuristik backend: {', '.join(red_flags) if red_flags else '-'}\n"
        f"Informasi yang sudah dijawab user: {answered_context}\n"
        f"Pertanyaan yang sudah pernah ditanyakan: {', '.join(asked_questions) if asked_questions else '-'}\n"
        f"Ringkasan anamnesis backend: {json.dumps(anamnesis_summary, ensure_ascii=False)}\n\n"
        f"Riwayat percakapan:\n{conversation}\n\n"
        f"Konteks retrieval:\n{contexts}\n\n"
        "Aturan penting:\n"
        "- Fokus pada penyakit/keluhan non-kritis dan non-penyakit dalam.\n"
        "- scope harus tepat satu label saja: supported, internal, critical, atau unsupported.\n"
        "- Jika kasus mengarah ke penyakit dalam, pakai scope=internal. Jika kritis/red flag berat, pakai scope=critical.\n"
        "- Jika mode anamnesis_follow_up dan informasi belum cukup, berikan tepat satu pertanyaan lanjutan yang paling penting.\n"
        "- Jangan menanyakan ulang gejala, durasi, atau tanda bahaya yang sudah ada pada informasi yang sudah dijawab user.\n"
        "- Jangan mengulang pertanyaan follow-up yang sebelumnya sudah pernah ditanyakan, kecuali jawaban user benar-benar kosong atau ambigu.\n"
        "- Pertanyaan lanjutan harus memperdalam: tanyakan intensitas, pola, tanda bahaya yang belum dijawab, atau gejala pembeda.\n"
        "- Jika user sudah menyebut demam, jangan bertanya apakah ada demam; tanyakan suhu tertinggi atau pola demam.\n"
        "- Jika user sudah menyebut durasi, jangan bertanya sejak kapan; tanyakan apakah memburuk atau tanda bahaya lain.\n"
        "- Contoh: bila user berkata sakit kepala dan demam sudah 3 hari, tanyakan suhu tertinggi, nyeri belakang mata, ruam, muntah, kaku leher, atau lemas berat.\n"
        "- Jika mode final_answer, isi field final dan bila aman sertakan ramuan hanya dari konteks retrieval.\n"
        "- Jangan mengarang bahan, dosis, atau cara pengolahan di luar konteks retrieval.\n"
        "- suspected maksimal 2 item.\n"
        "- why maksimal 25 kata.\n"
        "- question maksimal 35 kata dan tetap berupa satu kalimat tanya.\n"
        "- question_why maksimal 20 kata.\n"
        "- final maksimal 90 kata.\n"
        "- ingredients maksimal 4 item.\n"
        f"- Disclaimer harus tercermin di field final: {DISCLAIMER}\n"
        "- Gunakan kalimat singkat dan hemat token.\n\n"
        f"Output JSON yang diharapkan:\n{json.dumps(schema, ensure_ascii=False, indent=2)}"
    )


def parse_model_assessment(raw_reply: str) -> ModelAssessment:
    normalized = strip_thinking(raw_reply).strip()
    normalized = re.sub(r"^```(?:json)?\s*", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\s*```$", "", normalized)
    json_blob = extract_json_blob(normalized)
    data = json.loads(json_blob)
    if not isinstance(data, dict):
        raise ValueError("Structured response harus berupa object JSON.")

    raw_scope = str(data.get("scope") or "unsupported").strip().lower()
    scope_tokens = [token.strip() for token in re.split(r"[|,/]+", raw_scope) if token.strip()] or [raw_scope]
    scope = "unsupported"
    for token in scope_tokens:
        mapped = {
            "supported": "supported",
            "non-critical": "supported",
            "non_critical": "supported",
            "internal": "internal_medicine",
            "internal_medicine": "internal_medicine",
            "critical": "critical",
            "unsupported": "unsupported",
        }.get(token)
        if mapped:
            scope = mapped
            break

    return ModelAssessment(
        scope=scope,
        scope_reason=str(data.get("scope_reason") or data.get("why") or "").strip(),
        suspected_conditions=ensure_list(data.get("suspected_conditions") or data.get("suspected")),
        reasoning=str(data.get("reasoning") or data.get("why") or "").strip(),
        enough_information=bool(data.get("enough_information") if "enough_information" in data else data.get("enough")),
        follow_up_question=clean_optional_text(data.get("follow_up_question") or data.get("question")),
        follow_up_rationale=str(data.get("follow_up_rationale") or data.get("question_why") or "").strip(),
        red_flags=ensure_list(data.get("red_flags")),
        need_medical_referral=bool(
            data.get("need_medical_referral") if "need_medical_referral" in data else data.get("refer")
        ),
        final_answer=str(data.get("final_answer") or data.get("final") or "").strip(),
        recommended_herbal_name=str(data.get("recommended_herbal_name") or data.get("herbal") or "").strip(),
        ingredients=ensure_list(data.get("ingredients")),
        preparation=str(data.get("preparation") or data.get("prep") or "").strip(),
        dosage=str(data.get("dosage") or data.get("dose") or "").strip(),
        warning_notes=str(data.get("warning_notes") or data.get("warn") or "").strip(),
        source_hint=str(data.get("source_hint") or data.get("source") or "").strip(),
    )


def extract_json_blob(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Tidak menemukan object JSON pada output model.")
    return text[start : end + 1]


def render_candidate_reply(assessment: ModelAssessment, response_type: str) -> str:
    if response_type == "follow_up" and not assessment.enough_information and assessment.follow_up_question:
        return assessment.follow_up_question
    if assessment.final_answer:
        return assessment.final_answer
    if assessment.follow_up_question:
        return assessment.follow_up_question
    if assessment.scope_reason:
        return assessment.scope_reason
    if assessment.reasoning:
        return assessment.reasoning
    return "Model belum menghasilkan jawaban yang dapat dipakai."


def score_assessment(
    *,
    assessment: ModelAssessment,
    response_type: str,
    retrieved_context: list[RetrievedContext],
    anamnesis_summary: dict[str, object],
    user_message: str,
    question_count: int,
    max_questions: int,
    force_final: bool,
) -> tuple[float, dict[str, float]]:
    reply_text = render_candidate_reply(assessment, response_type)
    normalized_reply = normalize(reply_text)
    breakdown = {
        "safety": assessment_safety_score(assessment, normalized_reply),
        "grounding": assessment_grounding_score(assessment, retrieved_context),
        "completeness": assessment_completeness_score(assessment, response_type, force_final),
        "relevance": assessment_relevance_score(assessment, user_message, retrieved_context, anamnesis_summary),
        "language": language_score(reply_text),
    }
    if response_type == "follow_up":
        breakdown["question_quality"] = question_quality_score(assessment, question_count, max_questions, force_final)
        breakdown["non_repetition"] = question_non_repetition_score(assessment, anamnesis_summary)
        total = (
            0.25 * breakdown["safety"]
            + 0.2 * breakdown["grounding"]
            + 0.16 * breakdown["completeness"]
            + 0.16 * breakdown["relevance"]
            + 0.08 * breakdown["language"]
            + 0.05 * breakdown["question_quality"]
            + 0.1 * breakdown["non_repetition"]
        )
    else:
        breakdown["herbal_fit"] = herbal_fit_score(assessment, retrieved_context)
        total = (
            0.28 * breakdown["safety"]
            + 0.22 * breakdown["grounding"]
            + 0.2 * breakdown["completeness"]
            + 0.14 * breakdown["relevance"]
            + 0.08 * breakdown["language"]
            + 0.08 * breakdown["herbal_fit"]
        )
    return round(max(0.0, min(1.0, total)), 4), {key: round(value, 4) for key, value in breakdown.items()}


def assessment_safety_score(assessment: ModelAssessment, normalized_reply: str) -> float:
    score = 0.25
    if assessment.scope in {"critical", "internal_medicine"}:
        score += 0.18
    if assessment.need_medical_referral:
        score += 0.18
    if assessment.red_flags:
        score += 0.14
    if "tenaga kesehatan" in normalized_reply or "fasilitas kesehatan" in normalized_reply:
        score += 0.14
    if "bukan diagnosis" in normalized_reply or "bukan pengganti" in normalized_reply:
        score += 0.11
    return min(1.0, score)


def assessment_grounding_score(assessment: ModelAssessment, retrieved_context: list[RetrievedContext]) -> float:
    expected_terms: set[str] = set()
    for item in retrieved_context[:8]:
        expected_terms.update(tokenize(item.title))
    for condition in assessment.suspected_conditions:
        expected_terms.update(tokenize(condition))
    expected_terms.update(tokenize(assessment.recommended_herbal_name))
    for ingredient in assessment.ingredients:
        expected_terms.update(tokenize(ingredient))
    if not expected_terms:
        return 0.5
    matched = [term for term in expected_terms if term in normalize(" ".join([assessment.final_answer, assessment.reasoning, assessment.source_hint]))]
    return min(1.0, len(matched) / max(5, len(expected_terms) * 0.6))


def assessment_completeness_score(assessment: ModelAssessment, response_type: str, force_final: bool) -> float:
    score = 0.0
    if assessment.suspected_conditions:
        score += 0.22
    if assessment.reasoning:
        score += 0.16
    if response_type == "follow_up" and not force_final:
        if assessment.follow_up_question:
            score += 0.32
        if not assessment.enough_information:
            score += 0.15
        if assessment.follow_up_rationale:
            score += 0.15
        return min(1.0, score)

    if assessment.final_answer:
        score += 0.22
    if assessment.warning_notes or assessment.need_medical_referral:
        score += 0.12
    if assessment.recommended_herbal_name:
        score += 0.08
    if assessment.preparation:
        score += 0.08
    if assessment.dosage:
        score += 0.07
    return min(1.0, score)


def assessment_relevance_score(
    assessment: ModelAssessment,
    user_message: str,
    retrieved_context: list[RetrievedContext],
    anamnesis_summary: dict[str, object],
) -> float:
    score = 0.2
    query_terms = set(tokenize(user_message))
    symptom_terms = set()
    for symptom in anamnesis_summary.get("detected_symptoms", []) if isinstance(anamnesis_summary.get("detected_symptoms"), list) else []:
        symptom_terms.update(tokenize(str(symptom)))
    condition_terms = set()
    for item in assessment.suspected_conditions:
        condition_terms.update(tokenize(item))
    question_terms = set(tokenize(assessment.follow_up_question or assessment.final_answer))
    context_terms = set()
    for item in retrieved_context[:5]:
        context_terms.update(tokenize(item.title))

    if query_terms and question_terms:
        score += min(0.25, len(query_terms & question_terms) / max(3, len(query_terms)))
    if symptom_terms and question_terms:
        score += min(0.2, len(symptom_terms & question_terms) / max(2, len(symptom_terms)))
    if condition_terms and context_terms:
        score += min(0.2, len(condition_terms & context_terms) / max(2, len(condition_terms)))
    if assessment.follow_up_question and assessment.follow_up_question.endswith("?"):
        score += 0.1
    return min(1.0, score)


def question_quality_score(
    assessment: ModelAssessment,
    question_count: int,
    max_questions: int,
    force_final: bool,
) -> float:
    if force_final or assessment.enough_information:
        return 0.5
    question = (assessment.follow_up_question or "").strip()
    if not question:
        return 0.0
    score = 0.35
    if question.endswith("?"):
        score += 0.18
    if 15 <= len(question) <= 180:
        score += 0.18
    if question.count("?") <= 1:
        score += 0.12
    if question_count < max_questions:
        score += 0.1
    return min(1.0, score)


def question_non_repetition_score(assessment: ModelAssessment, anamnesis_summary: dict[str, object]) -> float:
    question = normalize(assessment.follow_up_question or "")
    if not question:
        return 0.0

    answered_symptoms = set(_summary_list(anamnesis_summary, "present_symptoms"))
    answered_symptoms.update(_summary_list(anamnesis_summary, "absent_symptoms"))
    mentioned_symptoms = _symptoms_mentioned_in_question(question)
    new_symptoms = mentioned_symptoms - answered_symptoms

    has_duration_answer = bool(anamnesis_summary.get("duration_text") or anamnesis_summary.get("has_duration_signal"))
    if has_duration_answer and any(term in question for term in FOLLOW_UP_DURATION_TERMS):
        return 0.15
    if _is_duplicate_question_history(question, _summary_list(anamnesis_summary, "asked_follow_up_questions")):
        return 0.05
    if any(term in question for term in FOLLOW_UP_GENERIC_SYMPTOM_TERMS) and not new_symptoms:
        return 0.15

    asks_existence = any(
        term in question
        for term in ("apakah ada", "apa ada", "ada ", "terdapat", "disertai", "mengalami")
    )
    asks_detail = any(term in question for term in FOLLOW_UP_DETAIL_TERMS)
    if asks_existence and bool(mentioned_symptoms & answered_symptoms) and not new_symptoms and not asks_detail:
        return 0.15
    return 1.0


def herbal_fit_score(assessment: ModelAssessment, retrieved_context: list[RetrievedContext]) -> float:
    if not assessment.recommended_herbal_name and not assessment.ingredients:
        return 0.45 if assessment.need_medical_referral or assessment.scope != "supported" else 0.25
    context_text = normalize(" ".join(item.title for item in retrieved_context[:8]))
    score = 0.3
    if assessment.recommended_herbal_name and normalize(assessment.recommended_herbal_name) in context_text:
        score += 0.3
    ingredient_hits = sum(1 for ingredient in assessment.ingredients if normalize(ingredient) in context_text)
    if assessment.ingredients:
        score += min(0.3, ingredient_hits / max(1, len(assessment.ingredients)) * 0.3)
    if assessment.preparation and assessment.dosage:
        score += 0.1
    return min(1.0, score)


def ensure_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in re.split(r"[;\n,]+", value) if item.strip()]
    return []


def clean_optional_text(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _answered_context_for_prompt(anamnesis_summary: dict[str, object]) -> str:
    present = _summary_list(anamnesis_summary, "present_symptoms")
    absent = _summary_list(anamnesis_summary, "absent_symptoms")
    duration = str(anamnesis_summary.get("duration_text") or "").strip()
    slots = _summary_list(anamnesis_summary, "answered_slots")
    parts = []
    if present:
        parts.append(f"gejala ada: {', '.join(present)}")
    if absent:
        parts.append(f"gejala tidak ada: {', '.join(absent)}")
    if duration:
        parts.append(f"durasi: {duration}")
    if slots:
        parts.append(f"slot terjawab: {', '.join(slots[:8])}")
    return "; ".join(parts) if parts else "-"


def _symptoms_mentioned_in_question(normalized_question: str) -> set[str]:
    symptoms = set()
    for symptom, aliases in SYMPTOM_CONCEPTS.items():
        if any(contains_phrase(normalized_question, alias) for alias in [symptom, *aliases]):
            symptoms.add(symptom)
    return symptoms


def _is_duplicate_question_history(normalized_question: str, asked_questions: list[str]) -> bool:
    current_terms = set(tokenize(normalized_question))
    for asked_question in asked_questions:
        normalized_asked = normalize(asked_question)
        if not normalized_asked:
            continue
        if normalized_question == normalized_asked:
            return True
        if normalized_question in normalized_asked or normalized_asked in normalized_question:
            return True
        asked_terms = set(tokenize(normalized_asked))
        if current_terms and asked_terms:
            overlap = len(current_terms & asked_terms) / max(1, min(len(current_terms), len(asked_terms)))
            if overlap >= 0.72:
                return True
    return False


def _summary_list(anamnesis_summary: dict[str, object], key: str) -> list[str]:
    values = anamnesis_summary.get(key)
    if not isinstance(values, list):
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def _content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or ""))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content or "")


def extract_ollama_inference_metrics(data: dict[str, object]) -> dict[str, object]:
    metrics: dict[str, object] = {}
    for key in OLLAMA_DURATION_FIELDS + OLLAMA_COUNT_FIELDS:
        value = _safe_int(data.get(key))
        if value is not None:
            metrics[key] = value

    response_model = str(data.get("model") or "").strip()
    if response_model:
        metrics["response_model"] = response_model

    created_at = str(data.get("created_at") or "").strip()
    if created_at:
        metrics["created_at"] = created_at

    done_reason = str(data.get("done_reason") or "").strip()
    if done_reason:
        metrics["done_reason"] = done_reason

    prompt_rate = _token_rate_from_ns(metrics.get("prompt_eval_count"), metrics.get("prompt_eval_duration"))
    if prompt_rate is not None:
        metrics["prompt_eval_rate_tps"] = prompt_rate

    eval_rate = _token_rate_from_ns(metrics.get("eval_count"), metrics.get("eval_duration"))
    if eval_rate is not None:
        metrics["eval_rate_tps"] = eval_rate
    return metrics


def extract_openai_inference_metrics(data: dict[str, object]) -> dict[str, object]:
    metrics: dict[str, object] = {}
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    for key in OPENAI_USAGE_FIELDS:
        value = _safe_int(usage.get(key) if isinstance(usage, dict) else None)
        if value is not None:
            metrics[key] = value

    prompt_details = usage.get("prompt_tokens_details") if isinstance(usage, dict) else {}
    if isinstance(prompt_details, dict):
        cached_prompt_tokens = _safe_int(prompt_details.get("cached_tokens"))
        if cached_prompt_tokens is not None:
            metrics["cached_prompt_tokens"] = cached_prompt_tokens

    completion_details = usage.get("completion_tokens_details") if isinstance(usage, dict) else {}
    if isinstance(completion_details, dict):
        reasoning_tokens = _safe_int(completion_details.get("reasoning_tokens"))
        if reasoning_tokens is not None:
            metrics["reasoning_tokens"] = reasoning_tokens

    response_model = str(data.get("model") or "").strip()
    if response_model:
        metrics["response_model"] = response_model

    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0] if isinstance(choices[0], dict) else {}
        finish_reason = str(first_choice.get("finish_reason") or "").strip()
        if finish_reason:
            metrics["finish_reason"] = finish_reason
    return metrics


def _safe_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _token_rate_from_ns(count_value: object, duration_value: object) -> float | None:
    count = _safe_int(count_value)
    duration_ns = _safe_int(duration_value)
    if count is None or duration_ns is None or duration_ns <= 0:
        return None
    return round(count / (duration_ns / 1_000_000_000), 2)


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
    if 20 <= length <= 1800:
        return 1.0
    if 10 <= length < 20 or 1800 < length <= 2600:
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
        "instruksi jawaban",
        "konteks retrieval",
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
