from __future__ import annotations

from uuid import uuid4

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import APP_NAME, APP_VERSION, CHROMA_DB_DIR, CORS_ORIGINS, DATA_DIR
from app.models import ChatRequest, ChatResponse
from app.services.anamnesis import analyze_message
from app.services.knowledge_base import AnamnesisEntry, CaseEntry, KnowledgeBase
from app.services.llm_comparison import DualLLMComparator
from app.services.recommendation import (
    DISCLAIMER,
    build_follow_up_reply,
    build_out_of_scope_reply,
    build_recommendation,
    build_recommendation_reply,
    build_red_flag_reply,
    to_context,
)
from app.services.retrieval import RAGRetriever, RetrievedItem
from app.services.text import contains_phrase, normalize


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
llm_comparator = DualLLMComparator()
SESSIONS: dict[str, dict[str, object]] = {}


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "app": APP_NAME,
        "version": APP_VERSION,
        "knowledge_base": kb.health(),
        "retriever": retriever.health(),
        "llm_comparison": llm_comparator.health(),
        "pipeline": [
            "humanizing_anamnesis",
            "chunked_knowledge_base",
            retriever.health()["backend"],
            "metadata_reranking",
            "grounded_recommendation_generation",
            "dual_ollama_candidate_generation",
            "deterministic_candidate_scoring",
            "red_flag_guardrail",
        ],
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
    session = SESSIONS.setdefault(session_id, {"turns": [], "matched_case_id": None, "asked_anamnesis": False})
    session["turns"].append({"role": "user", "content": request.message})

    anamnesis = analyze_message(request.message)
    pipeline = [
        "1. Humanizing anamnesis: ekstraksi gejala, durasi, permintaan rekomendasi, dan red flag",
        f"2. Retrieval: query ke chunk knowledge base dengan {retriever.health()['backend']}",
        "3. Re-ranking: boost berdasarkan sinyal gejala, formula, bahan, dan evidence level",
        "4. Grounded generation: respons hanya memakai case dan konteks retrieval terpilih",
        "5. Dual LLM comparison: prompt dikirim ke DeepSeek-R1 dan model Gemma/Gemma4 via Ollama",
        "6. Candidate scoring: penilaian keamanan, grounding RAG, kelengkapan, dan bahasa",
        "7. Guardrail: disclaimer dan penolakan untuk red flag/out-of-scope",
    ]
    if anamnesis.red_flags:
        reply = build_red_flag_reply(anamnesis.red_flags)
        response = _with_model_comparison(
            ChatResponse(
                session_id=session_id,
                response_type="red_flag",
                reply=reply,
                red_flags=anamnesis.red_flags,
                quick_replies=_quick_replies("red_flag"),
                anamnesis_summary=_anamnesis_summary(anamnesis),
                pipeline=pipeline,
            ),
            request.message,
        )
        session["turns"].append({"role": "assistant", "content": response.reply})
        return response

    previous_case_id = session.get("matched_case_id")
    top_cases = retriever.retrieve_cases(request.message, anamnesis, limit=3)
    if previous_case_id and _should_keep_previous_case(session, str(previous_case_id), request.message):
        previous_case = kb.case_by_id(str(previous_case_id))
        if previous_case:
            top_cases = [
                RetrievedItem(
                    id=previous_case.id,
                    type="case",
                    title=previous_case.keluhan_ringan,
                    score=1.25,
                    source=previous_case.sumber_ringkas,
                    payload=previous_case,
                    evidence_level="conversation_context",
                )
            ] + [item for item in top_cases if item.id != previous_case.id]
    if not top_cases and previous_case_id:
        previous_case = kb.case_by_id(str(previous_case_id))
        if previous_case:
            top_cases = [
                RetrievedItem(
                    id=previous_case.id,
                    type="case",
                    title=previous_case.keluhan_ringan,
                    score=1.0,
                    source=previous_case.sumber_ringkas,
                    payload=previous_case,
                )
            ]

    if not top_cases:
        reply = build_out_of_scope_reply()
        response = _with_model_comparison(
            ChatResponse(
                session_id=session_id,
                response_type="out_of_scope",
                reply=reply,
                quick_replies=_quick_replies("out_of_scope"),
                anamnesis_summary=_anamnesis_summary(anamnesis),
                pipeline=pipeline,
            ),
            request.message,
        )
        session["turns"].append({"role": "assistant", "content": response.reply})
        return response

    best_case = top_cases[0].payload
    session["matched_case_id"] = best_case.id
    supporting_context = retriever.retrieve_context_for_case(request.message, best_case, anamnesis, limit=6)
    grounded_items = [top_cases[0]] + supporting_context
    contexts = to_context(grounded_items)
    anamnesis_summary = _anamnesis_summary(anamnesis, best_case.keluhan_ringan)

    should_ask_follow_up = _needs_more_follow_up(str(previous_case_id or ""), request.message, anamnesis) or (
        not _is_detail_request(request.message)
        and not session.get("asked_anamnesis")
        and not (
            anamnesis.has_duration_signal and anamnesis.asks_for_recommendation
        )
    )
    if should_ask_follow_up:
        session["asked_anamnesis"] = True
        anamnesis_record = _select_anamnesis_record(best_case, request.message)
        follow_up_questions = _follow_up_questions(best_case, anamnesis_record)
        reply = build_follow_up_reply(
            best_case,
            questions=follow_up_questions,
            source_title=anamnesis_record.suspected_condition if anamnesis_record else None,
        )
        response = _with_model_comparison(
            ChatResponse(
                session_id=session_id,
                response_type="follow_up",
                reply=reply,
                retrieved_context=contexts,
                follow_up_question="\n".join(follow_up_questions),
                quick_replies=_quick_replies("follow_up"),
                anamnesis_summary=anamnesis_summary,
                pipeline=pipeline,
            ),
            request.message,
        )
        session["turns"].append({"role": "assistant", "content": response.reply})
        return response

    recommendation = build_recommendation(best_case)
    reply = build_recommendation_reply(recommendation, grounded_items, anamnesis_summary)
    session["asked_anamnesis"] = False
    response = _with_model_comparison(
        ChatResponse(
            session_id=session_id,
            response_type="recommendation",
            reply=reply,
            recommendation=recommendation,
            retrieved_context=contexts,
            quick_replies=_quick_replies("recommendation"),
            anamnesis_summary=anamnesis_summary,
            pipeline=pipeline,
        ),
        request.message,
    )
    session["turns"].append({"role": "assistant", "content": response.reply})
    return response


@app.delete("/api/session/{session_id}")
def clear_session(session_id: str) -> dict[str, str]:
    SESSIONS.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}


def _anamnesis_summary(anamnesis, keluhan_ringan: str | None = None) -> dict[str, object]:
    return {
        "keluhan_ringan": keluhan_ringan,
        "detected_symptoms": anamnesis.detected_symptoms,
        "hinted_case_ids": anamnesis.hinted_case_ids,
        "has_duration_signal": anamnesis.has_duration_signal,
        "has_safety_clearance_signal": anamnesis.has_safety_clearance_signal,
        "asks_for_recommendation": anamnesis.asks_for_recommendation,
        "red_flags": anamnesis.red_flags,
    }


def _with_model_comparison(response: ChatResponse, user_message: str) -> ChatResponse:
    comparison = llm_comparator.compare(
        session_id=response.session_id,
        user_message=user_message,
        response_type=response.response_type,
        baseline_reply=response.reply,
        recommendation=response.recommendation,
        retrieved_context=response.retrieved_context,
        anamnesis_summary=response.anamnesis_summary,
        red_flags=response.red_flags,
        follow_up_question=response.follow_up_question,
    )
    response.model_comparison = comparison
    if comparison.selected_reply:
        response.reply = _ensure_disclaimer(comparison.selected_reply)
    return response


def _ensure_disclaimer(reply: str) -> str:
    normalized = normalize(reply)
    if "bukan diagnosis" in normalized and "bukan pengganti" in normalized:
        return reply
    return f"{reply.rstrip()}\n\n{DISCLAIMER}"


def _quick_replies(response_type: str) -> list[str]:
    options = {
        "follow_up": [
            "Keluhan masih ringan, tidak ada tanda bahaya, sudah kurang dari 3 hari.",
            "Ada demam tinggi, sesak, nyeri berat, atau gejala memburuk.",
            "Saya ingin rekomendasi ramuan herbal dan dosisnya.",
        ],
        "recommendation": [
            "Bagaimana cara membuat ramuan ini?",
            "Apa catatan kewaspadaannya?",
            "Mulai sesi baru",
        ],
        "out_of_scope": [
            "Saya mual ringan sejak tadi pagi.",
            "Tenggorokan saya tidak nyaman sejak kemarin.",
            "Saya diare ringan tanpa darah sejak tadi pagi.",
        ],
        "red_flag": [
            "Mulai sesi baru",
            "Saya ingin melihat contoh keluhan ringan yang didukung.",
        ],
    }
    return options.get(response_type, [])


def _is_detail_request(message: str) -> bool:
    lowered = message.lower()
    return any(
        keyword in lowered
        for keyword in [
            "cara membuat",
            "cara mengolah",
            "dosis",
            "kewaspadaan",
            "catatan",
            "bagaimana cara",
        ]
    )


def _should_keep_previous_case(session: dict[str, object], previous_case_id: str, message: str) -> bool:
    if not session.get("asked_anamnesis"):
        return False
    normalized = normalize(message)
    if previous_case_id == "case_009":
        return any(
            contains_phrase(normalized, phrase)
            for phrase in [
                "iya",
                "ya",
                "demam",
                "tidak demam",
                "ruam",
                "gatal",
                "bengkak",
                "sesak",
                "lepuh",
                "menyebar",
            ]
        )
    return False


def _needs_more_follow_up(previous_case_id: str, message: str, anamnesis) -> bool:
    normalized = normalize(message)
    if previous_case_id == "case_009":
        if contains_phrase(normalized, "demam") and not (
            contains_phrase(normalized, "demam tinggi")
            or contains_phrase(normalized, "demam ringan")
            or contains_phrase(normalized, "tidak demam")
            or contains_phrase(normalized, "tanpa demam")
            or contains_phrase(normalized, "tidak tinggi")
        ):
            return True
        if any(
            contains_phrase(normalized, phrase)
            for phrase in ["iya demam", "ya demam", "ada demam"]
        ):
            return True
    return False


def _select_anamnesis_record(case: CaseEntry, message: str) -> AnamnesisEntry | None:
    candidates = [record for record in kb.anamnesis_records if case.id in record.applicable_case_ids]
    if not candidates:
        return None

    normalized = normalize(message)

    def score(record: AnamnesisEntry) -> int:
        value = 0
        for symptom in record.primary_symptoms:
            if contains_phrase(normalized, symptom):
                value += 3
            elif any(contains_phrase(normalized, token) for token in symptom.split()):
                value += 1
        if contains_phrase(normalized, record.condition_group):
            value += 2
        if contains_phrase(normalized, record.suspected_condition):
            value += 2
        if case.id in record.applicable_case_ids:
            value += 1
        return value

    return max(candidates, key=score)


def _follow_up_questions(case: CaseEntry, record: AnamnesisEntry | None) -> list[str]:
    questions: list[str] = []
    if record:
        questions.extend(record.required_questions[:4])
        questions.extend(record.red_flag_questions[:2])
    else:
        questions.append(case.pertanyaan_anamnesis)

    deduped: list[str] = []
    for question in questions:
        if question and question not in deduped:
            deduped.append(question)
    return deduped[:6]
