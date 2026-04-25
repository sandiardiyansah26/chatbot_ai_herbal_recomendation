from __future__ import annotations

from pydantic import BaseModel, Field


class ConversationTurn(BaseModel):
    role: str = Field(..., min_length=1, max_length=32)
    content: str = Field(..., min_length=1)


class SessionSync(BaseModel):
    turns: list[ConversationTurn] = Field(default_factory=list)
    question_count: int = 0
    conversation_stage: str = ""
    completed: bool = False
    suspected_conditions: list[str] = Field(default_factory=list)
    asked_follow_up_questions: list[str] = Field(default_factory=list)
    last_recommendation: dict[str, object] | None = None


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, max_length=80)
    session_sync: SessionSync | None = None


class RetrievedContext(BaseModel):
    id: str
    type: str
    title: str
    score: float
    source: str | None = None
    evidence_level: str | None = None
    matched_terms: list[str] = Field(default_factory=list)


class Recommendation(BaseModel):
    keluhan_ringan: str
    ramuan: str
    bahan: list[str]
    cara_pengolahan: str
    dosis_penggunaan: str
    catatan_kewaspadaan: str
    sumber_ringkas: str
    disclaimer: str


class ModelAssessment(BaseModel):
    scope: str = "supported"
    scope_reason: str = ""
    suspected_conditions: list[str] = Field(default_factory=list)
    reasoning: str = ""
    enough_information: bool = False
    follow_up_question: str | None = None
    follow_up_rationale: str = ""
    red_flags: list[str] = Field(default_factory=list)
    need_medical_referral: bool = False
    final_answer: str = ""
    recommended_herbal_name: str = ""
    ingredients: list[str] = Field(default_factory=list)
    preparation: str = ""
    dosage: str = ""
    warning_notes: str = ""
    source_hint: str = ""


class ModelCandidate(BaseModel):
    model: str
    status: str
    provider: str | None = None
    reply: str = ""
    score: float = 0.0
    latency_ms: int | None = None
    error: str | None = None
    scoring_breakdown: dict[str, float] = Field(default_factory=dict)
    inference_metrics: dict[str, object] = Field(default_factory=dict)
    assessment: ModelAssessment | None = None


class ModelComparison(BaseModel):
    enabled: bool = False
    selected_model: str | None = None
    selected_reply: str | None = None
    selected_assessment: ModelAssessment | None = None
    scoring_method: str = "deterministic_multi_llm_medical_ranking"
    learning_log_id: str | None = None
    candidates: list[ModelCandidate] = Field(default_factory=list)
    note: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    response_type: str
    reply: str
    conversation_stage: str = ""
    suspected_conditions: list[str] = Field(default_factory=list)
    recommendation: Recommendation | None = None
    retrieved_context: list[RetrievedContext] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    follow_up_question: str | None = None
    quick_replies: list[str] = Field(default_factory=list)
    anamnesis_summary: dict[str, object] = Field(default_factory=dict)
    questions_asked: int = 0
    max_questions: int = 3
    pipeline: list[str] = Field(default_factory=list)
    model_comparison: ModelComparison | None = None
    feedback_prompt: str | None = None
    feedback_options: list[str] = Field(default_factory=list)
