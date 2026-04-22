from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, max_length=80)


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


class ModelCandidate(BaseModel):
    model: str
    status: str
    reply: str = ""
    score: float = 0.0
    latency_ms: int | None = None
    error: str | None = None
    scoring_breakdown: dict[str, float] = Field(default_factory=dict)


class ModelComparison(BaseModel):
    enabled: bool = False
    selected_model: str | None = None
    selected_reply: str | None = None
    scoring_method: str = "deterministic_rag_safety_rubric"
    learning_log_id: str | None = None
    candidates: list[ModelCandidate] = Field(default_factory=list)
    note: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    response_type: str
    reply: str
    recommendation: Recommendation | None = None
    retrieved_context: list[RetrievedContext] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    follow_up_question: str | None = None
    quick_replies: list[str] = Field(default_factory=list)
    anamnesis_summary: dict[str, object] = Field(default_factory=dict)
    pipeline: list[str] = Field(default_factory=list)
    model_comparison: ModelComparison | None = None
