from __future__ import annotations

from copy import deepcopy
from uuid import uuid4

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import (
    APP_NAME,
    APP_VERSION,
    CHROMA_DB_DIR,
    CORS_ORIGINS,
    DATA_DIR,
    ENABLE_LLM_FOLLOW_UP,
    MAX_ANAMNESIS_QUESTIONS,
)
from app.models import ChatRequest, ChatResponse, ModelAssessment, ModelComparison, Recommendation, SessionSync
from app.services.anamnesis import SYMPTOM_CONCEPTS, analyze_message
from app.services.clinical_nlp import IndonesianClinicalNLPProcessor
from app.services.knowledge_base import CaseEntry, KnowledgeBase
from app.services.learning import LearningCaptureService
from app.services.llm_comparison import DualLLMComparator
from app.services.recommendation import (
    DISCLAIMER,
    build_assessment_follow_up_reply,
    build_assessment_recommendation,
    build_assessment_recommendation_reply,
    build_feedback_reply,
    build_follow_up_reply,
    build_out_of_scope_reply,
    build_preparation_detail_reply,
    build_recommendation,
    build_recommendation_reply,
    build_red_flag_reply,
    build_scope_referral_reply,
    enhance_recommendation_preparation,
    to_context,
)
from app.services.retrieval import RAGRetriever, RetrievedItem
from app.services.safety import evaluate_safety, safety_assessment_to_model_assessment
from app.services.text import contains_phrase, normalize, tokenize
app = FastAPI(title=APP_NAME, version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

kb = KnowledgeBase(DATA_DIR)
retriever = RAGRetriever(kb, chroma_db_dir=CHROMA_DB_DIR)
clinical_nlp = IndonesianClinicalNLPProcessor()
llm_comparator = DualLLMComparator()
learning_capture = LearningCaptureService()
SESSIONS: dict[str, dict[str, object]] = {}

FOLLOW_UP_CLINICAL_TERMS = (
    "sejak",
    "berapa lama",
    "demam",
    "sesak",
    "nyeri",
    "mual",
    "muntah",
    "darah",
    "bengkak",
    "memburuk",
    "durasi",
    "lemas",
    "lepuh",
    "menyebar",
    "perdarahan",
)
FOLLOW_UP_RECOMMENDATION_TERMS = (
    "bahan",
    "dosis",
    "gel",
    "lidah buaya",
    "memakai",
    "menggunakan",
    "minum",
    "oles",
    "ramuan",
)
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
    "terus",
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


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "app": APP_NAME,
        "version": APP_VERSION,
        "knowledge_base": kb.health(),
        "clinical_nlp": clinical_nlp.health(),
        "retriever": retriever.health(),
        "llm_comparison": llm_comparator.health(),
        "fast_anamnesis": {
            "enabled": not ENABLE_LLM_FOLLOW_UP,
            "llm_follow_up_enabled": ENABLE_LLM_FOLLOW_UP,
            "mode": "deterministic_humanizing_slot_based_follow_up",
        },
        "max_anamnesis_questions": MAX_ANAMNESIS_QUESTIONS,
        "pipeline": _pipeline_steps(),
    }


@app.get("/api/system-flow")
def system_flow() -> dict[str, object]:
    return {
        "objective": (
            "Deteksi awal kemungkinan penyakit tropis melalui anamnesis humanis Bahasa Indonesia, "
            "grounding RAG medical knowledge base, safety layer, dan rekomendasi herbal aman sebagai pendamping edukatif."
        ),
        "flow": _pipeline_steps(),
        "components": {
            "input": "User input Bahasa Indonesia",
            "clinical_nlp": clinical_nlp.health(),
            "anamnesis_reasoner": {
                "primary_model": "DeepSeek-R1",
                "runtime_models": llm_comparator.health().get("models", []),
                "task": "humanizing_anamnesis_dan_reasoning_pertanyaan_lanjutan",
                "fast_follow_up": not ENABLE_LLM_FOLLOW_UP,
            },
            "rag_medical_knowledge_base": {
                "sources": [
                    "Kemenkes/ayosehat/keslan/malaria.kemkes.go.id yang tersedia di dataset",
                    "WHO/CDC disease guidance yang tersedia di dataset",
                    "dokumen penyakit tropis lokal yang sudah dikurasi",
                    "referensi formula dan jurnal herbal pada knowledge base",
                ],
                "retriever": retriever.health(),
                "knowledge_base": kb.health(),
            },
            "safety_layer": {
                "checks": ["red_flag", "emergency", "medical_referral", "disclaimer", "batasan herbal"],
                "output_policy": "herbal tidak menjadi terapi utama pada red flag atau dugaan penyakit tropis berat",
            },
            "output": "Kemungkinan penyakit tropis + saran awal + rekomendasi herbal aman bila konteksnya mendukung",
        },
    }


@app.get("/api/knowledge-base")
def knowledge_base() -> dict[str, object]:
    return {
        "cases": [
            {
                "id": case.id,
                "keluhan_ringan": case.keluhan_ringan,
                "ramuan_rekomendasi": case.ramuan_rekomendasi,
                "bahan": case.bahan,
                "sumber_ringkas": case.sumber_ringkas,
            }
            for case in kb.cases
        ],
        "formulas": [
            {
                "id": formula.id,
                "nama_formula": formula.nama_formula,
                "gejala_target": formula.gejala_target,
                "evidence_level": formula.evidence_level,
            }
            for formula in kb.formulas
        ],
        "chunks": [
            {
                "id": chunk.id,
                "type": chunk.type,
                "title": chunk.title,
                "source": chunk.source,
                "evidence_level": chunk.evidence_level,
            }
            for chunk in kb.chunks
        ],
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    session_id = request.session_id or str(uuid4())
    session = SESSIONS.setdefault(session_id, _new_session())
    _sync_session_from_client(session, request.session_sync)
    turns = session.setdefault("turns", [])
    if isinstance(turns, list) and not _last_turn_matches(turns, "user", request.message):
        turns.append({"role": "user", "content": request.message})

    post_recommendation_response = _post_recommendation_response(
        session_id=session_id,
        session=session,
        user_message=request.message,
        pipeline=_pipeline_steps(),
    )
    if post_recommendation_response:
        return _finalize_response(session_id, session, request.message, post_recommendation_response)

    combined_user_message = _combined_user_messages(session)
    nlp_result = clinical_nlp.process(combined_user_message)
    session["last_nlp_summary"] = nlp_result.as_summary()
    anamnesis = analyze_message(nlp_result.normalized_text)
    question_count = int(session.get("question_count", 0))
    pipeline = _pipeline_steps()

    if anamnesis.red_flags:
        session["completed"] = True
        session["conversation_stage"] = "red_flag_guardrail"
        response = ChatResponse(
            session_id=session_id,
            response_type="red_flag",
            reply=build_red_flag_reply(anamnesis.red_flags),
            conversation_stage="red_flag_guardrail",
            suspected_conditions=list(session.get("suspected_conditions", [])),
            red_flags=anamnesis.red_flags,
            quick_replies=_quick_replies("red_flag"),
            anamnesis_summary=_anamnesis_summary(anamnesis, session=session),
            questions_asked=question_count,
            max_questions=MAX_ANAMNESIS_QUESTIONS,
            pipeline=pipeline,
        )
        return _finalize_response(session_id, session, request.message, response)

    retrieval = _retrieve_bundle(combined_user_message, anamnesis, session)
    best_case = retrieval["best_case"]
    retrieved_items = retrieval["items"]
    guidance_items = retrieval["guidance_items"]
    if best_case:
        session["matched_case_id"] = best_case.id

    if not _supports_medical_flow(retrieval["top_cases"], guidance_items, anamnesis):
        session["completed"] = True
        session["conversation_stage"] = "out_of_scope"
        response = ChatResponse(
            session_id=session_id,
            response_type="out_of_scope",
            reply=build_out_of_scope_reply(),
            conversation_stage="out_of_scope",
            quick_replies=_quick_replies("out_of_scope"),
            anamnesis_summary=_anamnesis_summary(anamnesis, session=session),
            questions_asked=question_count,
            max_questions=MAX_ANAMNESIS_QUESTIONS,
            pipeline=pipeline,
        )
        return _finalize_response(session_id, session, request.message, response)

    contexts = to_context(retrieved_items)
    anamnesis_summary = _anamnesis_summary(
        anamnesis,
        keluhan_ringan=best_case.keluhan_ringan if best_case else (guidance_items[0].title if guidance_items else None),
        session=session,
    )
    conversation_history = _conversation_history(session)

    if question_count >= MAX_ANAMNESIS_QUESTIONS:
        response = _final_recommendation_response(
            session_id=session_id,
            session=session,
            user_message=request.message,
            conversation_history=conversation_history,
            contexts=contexts,
            anamnesis_summary=anamnesis_summary,
            red_flags=anamnesis.red_flags,
            best_case=best_case,
            retrieved_items=retrieved_items,
            pipeline=pipeline,
        )
        return _finalize_response(session_id, session, request.message, response)

    if not ENABLE_LLM_FOLLOW_UP and not _is_fast_anamnesis_complete(anamnesis_summary, question_count):
        fast_assessment = _fast_follow_up_assessment(
            anamnesis_summary=anamnesis_summary,
            fallback_case=best_case,
            guidance_items=guidance_items,
            asked_questions=_asked_follow_up_questions(session),
        )
        fast_comparison = ModelComparison(
            enabled=False,
            selected_assessment=fast_assessment,
            selected_reply=fast_assessment.follow_up_question,
            note=(
                "Fast anamnesis mode: pertanyaan lanjutan dibuat dari slot klinis dan RAG agar respons lebih cepat; "
                "LLM ringan dipakai pada tahap final/reasoning."
            ),
        )
        session["question_count"] = question_count + 1
        session["suspected_conditions"] = fast_assessment.suspected_conditions[:3]
        session["conversation_stage"] = f"anamnesis_follow_up_{session['question_count']}"
        response = _follow_up_response(
            session_id=session_id,
            assessment=fast_assessment,
            comparison=fast_comparison,
            anamnesis_summary=anamnesis_summary,
            contexts=contexts,
            question_count=int(session["question_count"]),
            pipeline=pipeline,
            fallback_case=best_case,
            asked_questions=_asked_follow_up_questions(session),
        )
        _remember_follow_up_question(session, response.follow_up_question)
        return _finalize_response(session_id, session, request.message, response)

    follow_up_comparison = llm_comparator.generate_assessment(
        session_id=session_id,
        user_message=request.message,
        response_type="follow_up",
        conversation_history=conversation_history,
        retrieved_context=contexts,
        anamnesis_summary=anamnesis_summary,
        red_flags=anamnesis.red_flags,
        question_count=question_count,
        force_final=False,
    )
    selected_assessment = follow_up_comparison.selected_assessment
    safety_decision = evaluate_safety(
        red_flags=anamnesis.red_flags,
        assessment=selected_assessment,
        retrieved_items=retrieved_items,
        nlp_summary=session.get("last_nlp_summary") if isinstance(session.get("last_nlp_summary"), dict) else {},
    )

    if _needs_referral(selected_assessment) or safety_decision.requires_medical_referral:
        response = _referral_response(
            session_id=session_id,
            assessment=safety_assessment_to_model_assessment(safety_decision, selected_assessment)
            if safety_decision.requires_medical_referral
            else selected_assessment,
            comparison=follow_up_comparison,
            anamnesis_summary=anamnesis_summary,
            contexts=contexts,
            question_count=question_count,
            pipeline=pipeline,
        )
        session["completed"] = True
        session["conversation_stage"] = "medical_referral"
        return _finalize_response(session_id, session, request.message, response)

    if _is_unsupported(selected_assessment):
        session["completed"] = True
        session["conversation_stage"] = "out_of_scope"
        response = ChatResponse(
            session_id=session_id,
            response_type="out_of_scope",
            reply=build_out_of_scope_reply(),
            conversation_stage="out_of_scope",
            suspected_conditions=selected_assessment.suspected_conditions if selected_assessment else [],
            quick_replies=_quick_replies("out_of_scope"),
            anamnesis_summary=anamnesis_summary,
            retrieved_context=contexts,
            model_comparison=follow_up_comparison,
            questions_asked=question_count,
            max_questions=MAX_ANAMNESIS_QUESTIONS,
            pipeline=pipeline,
        )
        return _finalize_response(session_id, session, request.message, response)

    if _should_force_final_recommendation(session, anamnesis_summary, selected_assessment):
        response = _final_recommendation_response(
            session_id=session_id,
            session=session,
            user_message=request.message,
            conversation_history=conversation_history,
            contexts=contexts,
            anamnesis_summary=anamnesis_summary,
            red_flags=anamnesis.red_flags,
            best_case=best_case,
            retrieved_items=retrieved_items,
            pipeline=pipeline,
            initial_comparison=follow_up_comparison,
        )
        return _finalize_response(session_id, session, request.message, response)

    if _should_ask_follow_up(selected_assessment, question_count):
        session["question_count"] = question_count + 1
        session["suspected_conditions"] = (selected_assessment.suspected_conditions if selected_assessment else [])[:3]
        session["conversation_stage"] = f"anamnesis_follow_up_{session['question_count']}"
        response = _follow_up_response(
            session_id=session_id,
            assessment=selected_assessment,
            comparison=follow_up_comparison,
            anamnesis_summary=anamnesis_summary,
            contexts=contexts,
            question_count=int(session["question_count"]),
            pipeline=pipeline,
            fallback_case=best_case,
            asked_questions=_asked_follow_up_questions(session),
        )
        _remember_follow_up_question(session, response.follow_up_question)
        return _finalize_response(session_id, session, request.message, response)

    response = _final_recommendation_response(
        session_id=session_id,
        session=session,
        user_message=request.message,
        conversation_history=conversation_history,
        contexts=contexts,
        anamnesis_summary=anamnesis_summary,
        red_flags=anamnesis.red_flags,
        best_case=best_case,
        retrieved_items=retrieved_items,
        pipeline=pipeline,
        initial_comparison=follow_up_comparison,
    )
    return _finalize_response(session_id, session, request.message, response)


@app.delete("/api/session/{session_id}")
def clear_session(session_id: str) -> dict[str, str]:
    SESSIONS.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}


def _new_session() -> dict[str, object]:
    return {
        "turns": [],
        "matched_case_id": None,
        "question_count": 0,
        "suspected_conditions": [],
        "asked_follow_up_questions": [],
        "conversation_stage": "initial",
        "completed": False,
        "last_recommendation": None,
        "last_nlp_summary": {},
        "feedback": [],
    }


def _pipeline_steps() -> list[str]:
    return [
        "1. User input Bahasa Indonesia diterima dan digabung dengan riwayat sesi",
        f"2. IndoBERT/XLM-R layer melakukan normalisasi bahasa dan ekstraksi gejala ({clinical_nlp.health()['backend']})",
        "3. Fast anamnesis layer menyusun pertanyaan lanjutan humanis tanpa menunggu LLM besar",
        f"4. RAG Medical Knowledge Base mengambil konteks Kemenkes/WHO/dokumen penyakit tropis/jurnal herbal melalui {retriever.health()['backend']}",
        "5. Safety Layer memeriksa red flag, emergency, kebutuhan rujukan dokter, disclaimer, dan batasan herbal",
        "6. DeepSeek-R1 ringan dan Gemma ringan dipakai untuk reasoning/final answer saat informasi sudah cukup",
        f"7. Sistem bertanya maksimal {MAX_ANAMNESIS_QUESTIONS} kali sebelum finalisasi bila informasi belum cukup",
        "8. Output berisi kemungkinan penyakit tropis/kondisi terkait, saran awal, dan rekomendasi herbal aman bila layak",
        "9. Hasil model, konteks RAG, feedback, dan conversation turn disimpan sebagai learning log",
    ]


def _post_recommendation_response(
    *,
    session_id: str,
    session: dict[str, object],
    user_message: str,
    pipeline: list[str],
) -> ChatResponse | None:
    if not _is_post_recommendation_turn(session):
        return None

    recommendation = _last_recommendation(session)
    if _is_preparation_detail_request(user_message):
        if recommendation:
            reply = build_preparation_detail_reply(recommendation)
        else:
            reply = (
                "Saya belum menemukan kartu rekomendasi terakhir untuk dijelaskan lebih detail. "
                "Silakan mulai sesi baru atau ulangi keluhan utama agar saya bisa menyusun rekomendasi ulang."
            )
        return ChatResponse(
            session_id=session_id,
            response_type="preparation_detail",
            reply=reply,
            conversation_stage="preparation_detail_after_recommendation",
            recommendation=recommendation,
            quick_replies=_quick_replies("preparation_detail"),
            questions_asked=int(session.get("question_count", 0)),
            max_questions=MAX_ANAMNESIS_QUESTIONS,
            pipeline=pipeline,
        )

    feedback_label = _classify_recommendation_feedback(user_message)
    if not feedback_label:
        return None

    feedback_log_id = learning_capture.capture_feedback(
        session_id=session_id,
        user_message=user_message,
        feedback_label=feedback_label,
        recommendation=recommendation.model_dump() if recommendation else None,
        session_snapshot=_session_snapshot(session),
    )
    feedback_rows = session.setdefault("feedback", [])
    if isinstance(feedback_rows, list):
        feedback_rows.append(
            {
                "feedback_log_id": feedback_log_id,
                "label": feedback_label,
                "message": user_message,
            }
        )

    return ChatResponse(
        session_id=session_id,
        response_type="feedback",
        reply=f"{build_feedback_reply(feedback_label, recommendation)}\n\nFeedback log: {feedback_log_id}",
        conversation_stage="recommendation_feedback",
        recommendation=recommendation,
        quick_replies=_quick_replies("feedback"),
        questions_asked=int(session.get("question_count", 0)),
        max_questions=MAX_ANAMNESIS_QUESTIONS,
        pipeline=pipeline,
    )


def _is_post_recommendation_turn(session: dict[str, object]) -> bool:
    return bool(session.get("completed")) and str(session.get("conversation_stage") or "") == "final_recommendation"


def _last_recommendation(session: dict[str, object]) -> Recommendation | None:
    raw = session.get("last_recommendation")
    if isinstance(raw, Recommendation):
        return raw
    if isinstance(raw, dict):
        try:
            return Recommendation(**raw)
        except Exception:
            return None
    return None


def _is_preparation_detail_request(message: str) -> bool:
    normalized = message.lower()
    detail_terms = ("detail", "jelaskan", "rinci", "langkah", "step", "cara")
    preparation_terms = ("pengolahan", "membuat", "buat", "rebus", "seduh", "olah", "ramuan")
    return any(term in normalized for term in detail_terms) and any(term in normalized for term in preparation_terms)


def _classify_recommendation_feedback(message: str) -> str | None:
    normalized = message.lower()
    if any(term in normalized for term in ("pengolahan kurang jelas", "cara kurang jelas", "kurang detail", "tidak jelas")):
        return "unclear_preparation"
    if any(
        term in normalized
        for term in (
            "belum membantu",
            "tidak membantu",
            "kurang membantu",
            "kurang cocok",
            "tidak cocok",
            "gak cocok",
            "nggak cocok",
            "belum membaik",
            "tidak membaik",
            "bahan sulit",
            "sulit didapat",
        )
    ):
        return "not_helpful"
    if any(term in normalized for term in ("membantu", "cocok", "lebih baik", "bermanfaat", "terbantu")):
        return "helpful"
    return None


def _combined_user_messages(session: dict[str, object]) -> str:
    turns = session.get("turns", [])
    if not isinstance(turns, list):
        return ""
    user_messages = [str(turn.get("content") or "").strip() for turn in turns if turn.get("role") == "user"]
    return "\n".join(message for message in user_messages if message)


def _conversation_history(session: dict[str, object]) -> list[dict[str, str]]:
    turns = session.get("turns", [])
    if not isinstance(turns, list):
        return []
    return [
        {"role": str(turn.get("role") or "user"), "content": str(turn.get("content") or "")}
        for turn in turns[-10:]
        if turn.get("content")
    ]


def _retrieve_bundle(query: str, anamnesis, session: dict[str, object]) -> dict[str, object]:
    retrieval_query = _rag_query(query, session)
    guidance_items = retriever.retrieve_guidance(retrieval_query, anamnesis, limit=5)
    top_cases = retriever.retrieve_cases(retrieval_query, anamnesis, limit=3)

    best_case: CaseEntry | None = top_cases[0].payload if top_cases else None
    previous_case_id = str(session.get("matched_case_id") or "")
    if not best_case and previous_case_id:
        best_case = kb.case_by_id(previous_case_id)

    supporting_context: list[RetrievedItem] = []
    if best_case:
        supporting_context = retriever.retrieve_context_for_case(retrieval_query, best_case, anamnesis, limit=6)

    items: list[RetrievedItem] = []
    items.extend(guidance_items[:5])
    items.extend(top_cases[:2])
    items.extend(supporting_context[:6])
    return {
        "top_cases": top_cases,
        "guidance_items": guidance_items,
        "best_case": best_case,
        "items": _dedupe_items(items),
    }


def _rag_query(query: str, session: dict[str, object]) -> str:
    nlp_summary = session.get("last_nlp_summary")
    if not isinstance(nlp_summary, dict):
        return query
    parts = [
        query,
        str(nlp_summary.get("normalized_text") or ""),
        " ".join(str(item) for item in nlp_summary.get("extracted_symptoms", []) if str(item).strip())
        if isinstance(nlp_summary.get("extracted_symptoms"), list)
        else "",
        " ".join(str(item) for item in nlp_summary.get("risk_contexts", []) if str(item).strip())
        if isinstance(nlp_summary.get("risk_contexts"), list)
        else "",
    ]
    return " ".join(part.strip() for part in parts if part and part.strip())


def _dedupe_items(items: list[RetrievedItem]) -> list[RetrievedItem]:
    deduped: dict[str, RetrievedItem] = {}
    for item in sorted(items, key=lambda candidate: candidate.score, reverse=True):
        deduped.setdefault(item.id, item)
    return list(deduped.values())


def _supports_medical_flow(top_cases: list[RetrievedItem], guidance_items: list[RetrievedItem], anamnesis) -> bool:
    if not top_cases and not guidance_items:
        return False
    if anamnesis.detected_symptoms or anamnesis.present_symptoms or anamnesis.hinted_case_ids:
        return True
    best_score = max(
        [item.score for item in [*top_cases[:1], *guidance_items[:1]]],
        default=0.0,
    )
    return best_score >= 0.28


def _anamnesis_summary(anamnesis, keluhan_ringan: str | None = None, session: dict[str, object] | None = None) -> dict[str, object]:
    question_count = int((session or {}).get("question_count", 0))
    suspected_conditions = list((session or {}).get("suspected_conditions", []))
    nlp_summary = (session or {}).get("last_nlp_summary")
    if not isinstance(nlp_summary, dict):
        nlp_summary = {}
    return {
        "keluhan_ringan": keluhan_ringan,
        "clinical_nlp": nlp_summary,
        "nlp_extracted_symptoms": nlp_summary.get("extracted_symptoms", []),
        "nlp_negated_symptoms": nlp_summary.get("negated_symptoms", []),
        "nlp_risk_contexts": nlp_summary.get("risk_contexts", []),
        "nlp_backend": nlp_summary.get("backend"),
        "nlp_confidence": nlp_summary.get("confidence"),
        "detected_symptoms": anamnesis.detected_symptoms,
        "present_symptoms": getattr(anamnesis, "present_symptoms", []),
        "absent_symptoms": getattr(anamnesis, "absent_symptoms", []),
        "duration_text": getattr(anamnesis, "duration_text", None),
        "answered_slots": getattr(anamnesis, "answered_slots", []),
        "has_intensity_signal": getattr(anamnesis, "has_intensity_signal", False),
        "has_progression_signal": getattr(anamnesis, "has_progression_signal", False),
        "hinted_case_ids": anamnesis.hinted_case_ids,
        "has_duration_signal": anamnesis.has_duration_signal,
        "has_safety_clearance_signal": anamnesis.has_safety_clearance_signal,
        "asks_for_recommendation": anamnesis.asks_for_recommendation,
        "red_flags": anamnesis.red_flags,
        "question_count": question_count,
        "suspected_conditions": suspected_conditions,
        "asked_follow_up_questions": _asked_follow_up_questions(session or {}),
    }


def _is_fast_anamnesis_complete(anamnesis_summary: dict[str, object], question_count: int) -> bool:
    if question_count <= 0:
        return False
    has_primary_complaint = bool(
        _summary_list(anamnesis_summary, "present_symptoms")
        or _summary_list(anamnesis_summary, "detected_symptoms")
        or anamnesis_summary.get("keluhan_ringan")
    )
    has_duration = bool(anamnesis_summary.get("duration_text") or anamnesis_summary.get("has_duration_signal"))
    has_safety = bool(anamnesis_summary.get("has_safety_clearance_signal"))
    has_detail = bool(anamnesis_summary.get("has_intensity_signal") or anamnesis_summary.get("has_progression_signal"))
    return has_primary_complaint and has_duration and has_safety and (has_detail or question_count >= 2)


def _fast_follow_up_assessment(
    *,
    anamnesis_summary: dict[str, object],
    fallback_case: CaseEntry | None,
    guidance_items: list[RetrievedItem],
    asked_questions: list[str],
) -> ModelAssessment:
    question = _smart_follow_up_question(anamnesis_summary, fallback_case, asked_questions)
    suspected = _fast_suspected_conditions(anamnesis_summary, fallback_case, guidance_items)
    return ModelAssessment(
        scope="supported",
        scope_reason="Informasi awal belum cukup untuk menyimpulkan; perlu anamnesis lanjutan yang singkat dan aman.",
        suspected_conditions=suspected,
        reasoning=(
            "Pertanyaan dipilih dari gejala yang sudah disebut, slot yang belum lengkap, dan konteks RAG paling dekat."
        ),
        enough_information=False,
        follow_up_question=question,
        follow_up_rationale=(
            "Saya perlu memastikan pola gejala dan tanda bahaya sebelum memberi dugaan awal atau herbal pendamping."
        ),
        red_flags=[],
        need_medical_referral=False,
    )


def _fast_suspected_conditions(
    anamnesis_summary: dict[str, object],
    fallback_case: CaseEntry | None,
    guidance_items: list[RetrievedItem],
) -> list[str]:
    candidates: list[str] = []
    if fallback_case:
        candidates.append(fallback_case.keluhan_ringan)
    keluhan = str(anamnesis_summary.get("keluhan_ringan") or "").strip()
    if keluhan:
        candidates.append(keluhan)
    for item in guidance_items[:5]:
        if item.type == "anamnesis" and item.title:
            candidates.append(item.title)
    for symptom in _summary_list(anamnesis_summary, "nlp_extracted_symptoms")[:3]:
        candidates.append(f"gejala {symptom}")
    return list(dict.fromkeys(candidate for candidate in candidates if candidate))[:3]


def _needs_referral(assessment: ModelAssessment | None) -> bool:
    if assessment is None:
        return False
    return assessment.scope in {"critical", "internal_medicine"} or assessment.need_medical_referral


def _is_unsupported(assessment: ModelAssessment | None) -> bool:
    return assessment is not None and assessment.scope == "unsupported"


def _should_ask_follow_up(assessment: ModelAssessment | None, question_count: int) -> bool:
    if assessment is None:
        return question_count < MAX_ANAMNESIS_QUESTIONS
    return (
        assessment.scope == "supported"
        and not assessment.enough_information
        and bool(assessment.follow_up_question)
        and question_count < MAX_ANAMNESIS_QUESTIONS
    )


def _referral_response(
    *,
    session_id: str,
    assessment: ModelAssessment | None,
    comparison: ModelComparison,
    anamnesis_summary: dict[str, object],
    contexts,
    question_count: int,
    pipeline: list[str],
) -> ChatResponse:
    selected = assessment or ModelAssessment(
        scope="critical",
        scope_reason="Kasus ini tidak aman untuk jalur herbal mandiri.",
        need_medical_referral=True,
        warning_notes="Prioritaskan pemeriksaan tenaga kesehatan.",
    )
    return ChatResponse(
        session_id=session_id,
        response_type="medical_referral",
        reply=build_scope_referral_reply(selected, anamnesis_summary),
        conversation_stage="medical_referral",
        suspected_conditions=selected.suspected_conditions,
        red_flags=selected.red_flags,
        quick_replies=_quick_replies("medical_referral"),
        anamnesis_summary=anamnesis_summary,
        retrieved_context=contexts,
        model_comparison=comparison,
        questions_asked=question_count,
        max_questions=MAX_ANAMNESIS_QUESTIONS,
        pipeline=pipeline,
    )


def _follow_up_response(
    *,
    session_id: str,
    assessment: ModelAssessment | None,
    comparison: ModelComparison,
    anamnesis_summary: dict[str, object],
    contexts,
    question_count: int,
    pipeline: list[str],
    fallback_case: CaseEntry | None,
    asked_questions: list[str],
) -> ChatResponse:
    if assessment and assessment.follow_up_question:
        assessment_for_reply = _assessment_with_better_follow_up_question(
            assessment,
            fallback_case,
            anamnesis_summary,
            asked_questions,
        )
        comparison = _comparison_with_selected_follow_up(comparison, assessment_for_reply)
        reply = build_assessment_follow_up_reply(
            assessment_for_reply,
            anamnesis_summary,
            question_number=question_count,
            max_questions=MAX_ANAMNESIS_QUESTIONS,
        )
        follow_up_question = assessment_for_reply.follow_up_question
        suspected_conditions = assessment.suspected_conditions
    else:
        if fallback_case:
            follow_up_question = _smart_follow_up_question(anamnesis_summary, fallback_case, asked_questions)
            reply = build_follow_up_reply(fallback_case, questions=[follow_up_question])
        else:
            reply = (
                "Saya masih perlu satu klarifikasi lagi sebelum menyusun jawaban akhir. "
                "Bisa jelaskan sejak kapan keluhan ini muncul, apakah memburuk, dan apakah ada demam atau sesak?\n\n"
                f"{DISCLAIMER}"
            )
            follow_up_question = "Sejak kapan keluhan ini muncul, apakah memburuk, dan apakah ada demam atau sesak?"
        suspected_conditions = assessment.suspected_conditions if assessment else []

    return ChatResponse(
        session_id=session_id,
        response_type="follow_up",
        reply=reply,
        conversation_stage=f"anamnesis_follow_up_{question_count}",
        suspected_conditions=suspected_conditions,
        follow_up_question=follow_up_question,
        quick_replies=_quick_replies("follow_up"),
        anamnesis_summary=anamnesis_summary,
        retrieved_context=contexts,
        model_comparison=comparison,
        questions_asked=question_count,
        max_questions=MAX_ANAMNESIS_QUESTIONS,
        pipeline=pipeline,
    )


def _assessment_with_better_follow_up_question(
    assessment: ModelAssessment,
    fallback_case: CaseEntry | None,
    anamnesis_summary: dict[str, object],
    asked_questions: list[str],
) -> ModelAssessment:
    question = assessment.follow_up_question or ""
    needs_replacement = _is_low_quality_follow_up_question(question) or _is_repetitive_follow_up_question(
        question,
        anamnesis_summary,
    ) or _is_duplicate_of_previous_follow_up_question(question, asked_questions)
    if not needs_replacement:
        return assessment

    replacement_question = _smart_follow_up_question(anamnesis_summary, fallback_case, asked_questions)
    suspected_conditions = assessment.suspected_conditions[:3]
    if fallback_case:
        suspected_conditions = list(dict.fromkeys([fallback_case.keluhan_ringan, *suspected_conditions]))[:3]

    return assessment.model_copy(
        update={
            "follow_up_question": replacement_question,
            "follow_up_rationale": (
                "Pertanyaan diganti agar tidak mengulang informasi yang sudah disebut user dan tetap menyingkirkan tanda bahaya."
            ),
            "suspected_conditions": suspected_conditions,
        }
    )


def _is_low_quality_follow_up_question(question: str) -> bool:
    normalized = question.lower().strip()
    if not normalized or not normalized.endswith("?"):
        return True

    has_clinical_term = any(term in normalized for term in FOLLOW_UP_CLINICAL_TERMS)
    has_recommendation_term = any(term in normalized for term in FOLLOW_UP_RECOMMENDATION_TERMS)
    return not has_clinical_term or (has_recommendation_term and not has_clinical_term)


def _is_repetitive_follow_up_question(question: str, anamnesis_summary: dict[str, object]) -> bool:
    normalized = normalize(question)
    if not normalized:
        return True

    answered_symptoms = set(_summary_list(anamnesis_summary, "present_symptoms"))
    answered_symptoms.update(_summary_list(anamnesis_summary, "absent_symptoms"))
    mentioned_symptoms = _symptoms_mentioned_in_question(normalized)
    new_symptoms = mentioned_symptoms - answered_symptoms

    has_duration_answer = bool(anamnesis_summary.get("duration_text") or anamnesis_summary.get("has_duration_signal"))
    if has_duration_answer and any(term in normalized for term in FOLLOW_UP_DURATION_TERMS):
        return True

    is_generic_symptom_question = any(term in normalized for term in FOLLOW_UP_GENERIC_SYMPTOM_TERMS)
    if is_generic_symptom_question and not new_symptoms:
        return True

    asks_existence = any(
        term in normalized
        for term in ("apakah ada", "apa ada", "ada ", "terdapat", "disertai", "mengalami")
    )
    asks_detail = any(term in normalized for term in FOLLOW_UP_DETAIL_TERMS)
    return asks_existence and bool(mentioned_symptoms & answered_symptoms) and not new_symptoms and not asks_detail


def _smart_follow_up_question(
    anamnesis_summary: dict[str, object],
    fallback_case: CaseEntry | None,
    asked_questions: list[str] | None = None,
) -> str:
    present_symptoms = set(_summary_list(anamnesis_summary, "present_symptoms"))
    absent_symptoms = set(_summary_list(anamnesis_summary, "absent_symptoms"))
    duration_answered = bool(anamnesis_summary.get("duration_text") or anamnesis_summary.get("has_duration_signal"))
    safety_answered = bool(anamnesis_summary.get("has_safety_clearance_signal"))
    candidates: list[str] = []

    if {"demam", "sakit kepala"} <= present_symptoms:
        candidates.append(
            "Berapa suhu tertinggi, dan apakah ada nyeri belakang mata, ruam/bintik merah, muntah, kaku leher, atau sangat lemas?"
        )
    if {"demam", "ruam"} <= present_symptoms:
        candidates.append("Ruamnya berupa bintik perdarahan, dan apakah ada mimisan, gusi berdarah, muntah, atau lemas berat?")
    if {"demam", "sakit tenggorokan"} <= present_symptoms:
        candidates.extend(
            [
                "Apakah menelan terasa sangat sakit, ada bercak putih di amandel, atau suara serak yang makin berat?",
                "Apakah ada sesak, sulit menelan ludah, bengkak leher, atau demam tinggi yang masih menetap?",
                "Apakah ada batuk, pilek, nyeri badan, atau pembesaran amandel yang terasa jelas?",
            ]
        )
    if "sakit tenggorokan" in present_symptoms and "demam" not in absent_symptoms:
        candidates.extend(
            [
                "Apakah sakit tenggorokan terasa lebih berat saat menelan, dan adakah bercak putih, batuk, atau pilek?",
                "Apakah ada sesak, suara bindeng berat, sulit menelan, atau air liur terasa sulit ditelan?",
            ]
        )
    if "demam" in present_symptoms:
        candidates.append("Berapa suhu tertinggi, apakah demam naik-turun atau menetap, dan apakah ada ruam, muntah, atau sangat lemas?")
    if "sakit kepala" in present_symptoms:
        candidates.append("Nyeri kepala di bagian mana, seberapa berat, dan apakah ada penglihatan kabur, muntah, atau kaku leher?")
    if {"gatal", "ruam"} & present_symptoms:
        if not duration_answered:
            candidates.append("Sejak kapan ruam muncul, apakah menyebar, melepuh, nyeri, atau ada bengkak wajah/bibir?")
        candidates.append("Apakah ruam melebar, terasa nyeri/melepuh, atau ada bengkak wajah, bibir, lidah, atau sesak?")
    if "batuk" in present_symptoms:
        candidates.append("Batuknya kering atau berdahak, dan apakah ada sesak, nyeri dada, darah, atau demam tinggi?")
    if not duration_answered:
        candidates.append("Sejak kapan keluhan muncul, apakah makin berat, dan aktivitas apa yang membuatnya memburuk?")
    if not safety_answered:
        candidates.append("Apakah ada tanda bahaya seperti sesak, nyeri hebat, perdarahan, dehidrasi, pingsan, atau keluhan cepat memburuk?")
    if fallback_case:
        candidates.append(_first_non_repetitive_case_question(fallback_case, anamnesis_summary))
    candidates.append("Gejala apa yang paling mengganggu sekarang, dan apakah ada tanda bahaya yang belum kamu sebutkan?")

    return _pick_best_follow_up_candidate(candidates, anamnesis_summary, asked_questions or [])


def _first_non_repetitive_case_question(
    fallback_case: CaseEntry,
    anamnesis_summary: dict[str, object],
) -> str:
    questions = fallback_case.pertanyaan_anamnesis
    if isinstance(questions, str):
        questions = [questions]
    for question in questions:
        if not _is_repetitive_follow_up_question(question, anamnesis_summary):
            return question
    return "Apakah keluhan makin berat, menyebar, atau muncul tanda bahaya yang belum kamu sebutkan?"


def _pick_best_follow_up_candidate(
    candidates: list[str],
    anamnesis_summary: dict[str, object],
    asked_questions: list[str],
) -> str:
    for candidate in candidates:
        if not candidate:
            continue
        if _is_repetitive_follow_up_question(candidate, anamnesis_summary):
            continue
        if _is_duplicate_of_previous_follow_up_question(candidate, asked_questions):
            continue
        return candidate
    return "Gejala apa yang paling mengganggu sekarang, dan apakah ada tanda bahaya yang belum kamu sebutkan?"


def _symptoms_mentioned_in_question(normalized_question: str) -> set[str]:
    symptoms = set()
    for symptom, aliases in SYMPTOM_CONCEPTS.items():
        if any(contains_phrase(normalized_question, alias) for alias in [symptom, *aliases]):
            symptoms.add(symptom)
    return symptoms


def _summary_list(anamnesis_summary: dict[str, object], key: str) -> list[str]:
    values = anamnesis_summary.get(key)
    if not isinstance(values, list):
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def _asked_follow_up_questions(session: dict[str, object]) -> list[str]:
    values = session.get("asked_follow_up_questions")
    if not isinstance(values, list):
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def _remember_follow_up_question(session: dict[str, object], question: str | None) -> None:
    normalized_question = str(question or "").strip()
    if not normalized_question:
        return
    questions = _asked_follow_up_questions(session)
    if _is_duplicate_of_previous_follow_up_question(normalized_question, questions):
        session["asked_follow_up_questions"] = questions
        return
    session["asked_follow_up_questions"] = [*questions[-7:], normalized_question]


def _is_duplicate_of_previous_follow_up_question(question: str, asked_questions: list[str]) -> bool:
    normalized_question = normalize(question)
    if not normalized_question:
        return True

    current_terms = set(tokenize(normalized_question))
    for previous in asked_questions:
        normalized_previous = normalize(previous)
        if not normalized_previous:
            continue
        if normalized_question == normalized_previous:
            return True
        if normalized_question in normalized_previous or normalized_previous in normalized_question:
            return True
        previous_terms = set(tokenize(normalized_previous))
        if current_terms and previous_terms:
            overlap = len(current_terms & previous_terms) / max(1, min(len(current_terms), len(previous_terms)))
            if overlap >= 0.72:
                return True
    return False


def _should_force_final_recommendation(
    session: dict[str, object],
    anamnesis_summary: dict[str, object],
    assessment: ModelAssessment | None,
) -> bool:
    if assessment is None or assessment.scope != "supported":
        return False

    question_count = int(session.get("question_count", 0))
    if question_count <= 0:
        return False

    has_primary_complaint = bool(
        _summary_list(anamnesis_summary, "present_symptoms")
        or _summary_list(anamnesis_summary, "detected_symptoms")
        or anamnesis_summary.get("keluhan_ringan")
    )
    has_duration = bool(anamnesis_summary.get("duration_text") or anamnesis_summary.get("has_duration_signal"))
    has_safety = bool(anamnesis_summary.get("has_safety_clearance_signal"))
    has_detail = bool(anamnesis_summary.get("has_intensity_signal") or anamnesis_summary.get("has_progression_signal"))
    return has_primary_complaint and has_duration and has_safety and (has_detail or question_count >= 2)


def _sync_session_from_client(session: dict[str, object], session_sync: SessionSync | None) -> None:
    if session_sync is None:
        return
    turns = session.get("turns", [])
    if isinstance(turns, list) and turns:
        return

    synced_turns = [
        {"role": "assistant" if turn.role not in {"user", "assistant"} else turn.role, "content": turn.content}
        for turn in session_sync.turns
        if str(turn.content).strip()
    ]
    if synced_turns:
        session["turns"] = synced_turns
    session["question_count"] = max(int(session_sync.question_count), 0)
    session["conversation_stage"] = str(session_sync.conversation_stage or session.get("conversation_stage") or "initial")
    session["completed"] = bool(session_sync.completed)
    session["suspected_conditions"] = list(session_sync.suspected_conditions[:3])
    session["asked_follow_up_questions"] = [question for question in session_sync.asked_follow_up_questions if question.strip()]
    if session_sync.last_recommendation:
        session["last_recommendation"] = dict(session_sync.last_recommendation)


def _last_turn_matches(turns: list[dict[str, str]], role: str, content: str) -> bool:
    if not turns:
        return False
    last_turn = turns[-1]
    return (
        str(last_turn.get("role") or "") == role
        and normalize(str(last_turn.get("content") or "")) == normalize(content)
    )


def _comparison_with_selected_follow_up(
    comparison: ModelComparison,
    assessment: ModelAssessment,
) -> ModelComparison:
    if not comparison.enabled or not comparison.selected_model:
        return comparison

    selected_reply = assessment.follow_up_question or comparison.selected_reply
    updated_candidates = []
    for candidate in comparison.candidates:
        if candidate.model == comparison.selected_model and candidate.assessment is not None:
            updated_candidates.append(
                candidate.model_copy(
                    update={
                        "reply": selected_reply or candidate.reply,
                        "assessment": assessment,
                    }
                )
            )
        else:
            updated_candidates.append(candidate)

    return comparison.model_copy(
        update={
            "selected_reply": selected_reply,
            "selected_assessment": assessment,
            "candidates": updated_candidates,
        }
    )


def _final_recommendation_response(
    *,
    session_id: str,
    session: dict[str, object],
    user_message: str,
    conversation_history: list[dict[str, str]],
    contexts,
    anamnesis_summary: dict[str, object],
    red_flags: list[str],
    best_case: CaseEntry | None,
    retrieved_items: list[RetrievedItem],
    pipeline: list[str],
    initial_comparison: ModelComparison | None = None,
) -> ChatResponse:
    question_count = int(session.get("question_count", 0))
    comparison = llm_comparator.generate_assessment(
        session_id=session_id,
        user_message=user_message,
        response_type="recommendation",
        conversation_history=conversation_history,
        retrieved_context=contexts,
        anamnesis_summary=anamnesis_summary,
        red_flags=red_flags,
        question_count=question_count,
        force_final=True,
    )
    selected_assessment = comparison.selected_assessment
    safety_decision = evaluate_safety(
        red_flags=red_flags,
        assessment=selected_assessment,
        retrieved_items=retrieved_items,
        nlp_summary=session.get("last_nlp_summary") if isinstance(session.get("last_nlp_summary"), dict) else {},
    )

    if _needs_referral(selected_assessment) or safety_decision.requires_emergency_referral:
        session["completed"] = True
        session["conversation_stage"] = "medical_referral"
        return _referral_response(
            session_id=session_id,
            assessment=safety_assessment_to_model_assessment(safety_decision, selected_assessment)
            if safety_decision.requires_medical_referral
            else selected_assessment,
            comparison=comparison,
            anamnesis_summary=anamnesis_summary,
            contexts=contexts,
            question_count=question_count,
            pipeline=pipeline,
        )

    if selected_assessment and selected_assessment.suspected_conditions:
        session["suspected_conditions"] = selected_assessment.suspected_conditions[:3]

    recommendation = build_assessment_recommendation(selected_assessment, best_case) if selected_assessment else None
    recommendation = enhance_recommendation_preparation(recommendation, retrieved_items)
    if selected_assessment:
        reply = build_assessment_recommendation_reply(
            selected_assessment,
            recommendation,
            retrieved_items,
            anamnesis_summary,
        )
    elif best_case:
        fallback_recommendation = build_recommendation(best_case)
        recommendation = enhance_recommendation_preparation(fallback_recommendation, retrieved_items)
        reply = build_recommendation_reply(recommendation or fallback_recommendation, retrieved_items, anamnesis_summary)
        comparison = initial_comparison or comparison
    else:
        reply = build_out_of_scope_reply()

    session["completed"] = True
    session["conversation_stage"] = "final_recommendation"
    session["last_recommendation"] = recommendation.model_dump() if recommendation else None
    return ChatResponse(
        session_id=session_id,
        response_type="recommendation",
        reply=reply,
        conversation_stage="final_recommendation",
        suspected_conditions=session.get("suspected_conditions", []),
        recommendation=recommendation,
        quick_replies=_quick_replies("recommendation"),
        anamnesis_summary=anamnesis_summary,
        retrieved_context=contexts,
        model_comparison=comparison,
        questions_asked=question_count,
        max_questions=MAX_ANAMNESIS_QUESTIONS,
        pipeline=pipeline,
        feedback_prompt="Apakah rekomendasi ini membantu keluhanmu?",
        feedback_options=[
            "Rekomendasi ini membantu",
            "Belum membantu / kurang cocok",
            "Jelaskan cara pengolahan ramuan ini lebih detail",
        ],
    )


def _quick_replies(response_type: str) -> list[str]:
    options = {
        "follow_up": [
            "Tidak ada demam, tidak sesak, baru 1 hari.",
            "Keluhan makin berat dan ada tanda bahaya.",
            "Saya sudah jawab, lanjutkan anamnesis.",
        ],
        "recommendation": [
            "Rekomendasi ini membantu",
            "Belum membantu / kurang cocok",
            "Jelaskan cara pengolahan ramuan ini lebih detail",
        ],
        "feedback": [
            "Cara pengolahan kurang jelas",
            "Bahan sulit didapat",
            "Mulai sesi baru",
        ],
        "preparation_detail": [
            "Rekomendasi ini membantu",
            "Cara pengolahan kurang jelas",
            "Mulai sesi baru",
        ],
        "medical_referral": [
            "Tolong rangkum alasan saya harus diperiksa.",
            "Apa tanda bahaya yang harus saya waspadai?",
            "Mulai sesi baru",
        ],
        "out_of_scope": [
            "Saya mau cek keluhan lain.",
            "Keluhan saya mual ringan sejak tadi pagi.",
            "Mulai sesi baru",
        ],
        "red_flag": [
            "Tolong rangkum tanda bahaya yang terdeteksi.",
            "Saya akan mulai sesi baru.",
            "Mulai sesi baru",
        ],
    }
    return options.get(response_type, [])


def _finalize_response(
    session_id: str,
    session: dict[str, object],
    user_message: str,
    response: ChatResponse,
) -> ChatResponse:
    turns = session.setdefault("turns", [])
    if isinstance(turns, list):
        turns.append({"role": "assistant", "content": response.reply})
    learning_capture.capture_turn(
        session_id=session_id,
        user_message=user_message,
        response=response,
        session_snapshot=_session_snapshot(session),
    )
    return response


def _session_snapshot(session: dict[str, object]) -> dict[str, object]:
    snapshot = {
        "matched_case_id": session.get("matched_case_id"),
        "question_count": int(session.get("question_count", 0)),
        "suspected_conditions": list(session.get("suspected_conditions", [])),
        "asked_follow_up_questions": _asked_follow_up_questions(session),
        "conversation_stage": str(session.get("conversation_stage") or ""),
        "completed": bool(session.get("completed")),
        "last_recommendation": session.get("last_recommendation"),
        "last_nlp_summary": session.get("last_nlp_summary") if isinstance(session.get("last_nlp_summary"), dict) else {},
        "feedback": list(session.get("feedback", [])) if isinstance(session.get("feedback"), list) else [],
        "turn_count": len(session.get("turns", [])) if isinstance(session.get("turns", []), list) else 0,
        "turns": deepcopy(session.get("turns", [])),
    }
    return snapshot
