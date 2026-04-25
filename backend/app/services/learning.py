from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import CONVERSATION_LOG_PATH, KB_ENRICHMENT_LOG_PATH, RECOMMENDATION_FEEDBACK_LOG_PATH
from app.models import ChatResponse


class LearningCaptureService:
    def __init__(
        self,
        conversation_log_path: Path = CONVERSATION_LOG_PATH,
        enrichment_log_path: Path = KB_ENRICHMENT_LOG_PATH,
        feedback_log_path: Path = RECOMMENDATION_FEEDBACK_LOG_PATH,
    ):
        self.conversation_log_path = conversation_log_path
        self.enrichment_log_path = enrichment_log_path
        self.feedback_log_path = feedback_log_path

    def capture_turn(
        self,
        *,
        session_id: str,
        user_message: str,
        response: ChatResponse,
        session_snapshot: dict[str, Any],
    ) -> dict[str, str | None]:
        turn_log_id = str(uuid.uuid4())
        turn_row = {
            "id": turn_log_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "user_message": user_message,
            "response_type": response.response_type,
            "conversation_stage": response.conversation_stage,
            "suspected_conditions": response.suspected_conditions,
            "red_flags": response.red_flags,
            "follow_up_question": response.follow_up_question,
            "assistant_reply": response.reply,
            "questions_asked": response.questions_asked,
            "max_questions": response.max_questions,
            "anamnesis_summary": response.anamnesis_summary,
            "retrieved_context": [item.model_dump() for item in response.retrieved_context],
            "recommendation": response.recommendation.model_dump() if response.recommendation else None,
            "model_comparison": response.model_comparison.model_dump() if response.model_comparison else None,
            "session_snapshot": session_snapshot,
        }
        self._append_jsonl(self.conversation_log_path, turn_row)

        enrichment_log_id: str | None = None
        if response.model_comparison and response.model_comparison.selected_assessment:
            assessment = response.model_comparison.selected_assessment
            enrichment_log_id = str(uuid.uuid4())
            enrichment_row = {
                "id": enrichment_log_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_turn_log_id": turn_log_id,
                "session_id": session_id,
                "status": "pending_curation",
                "source_type": "conversation_learning",
                "scope": assessment.scope,
                "scope_reason": assessment.scope_reason,
                "suspected_conditions": assessment.suspected_conditions,
                "reasoning": assessment.reasoning,
                "red_flags": assessment.red_flags,
                "follow_up_question": assessment.follow_up_question,
                "need_medical_referral": assessment.need_medical_referral,
                "recommended_herbal_name": assessment.recommended_herbal_name,
                "ingredients": assessment.ingredients,
                "preparation": assessment.preparation,
                "dosage": assessment.dosage,
                "warning_notes": assessment.warning_notes,
                "final_answer": assessment.final_answer,
                "retrieved_context_titles": [item.title for item in response.retrieved_context[:8]],
                "model": response.model_comparison.selected_model,
            }
            self._append_jsonl(self.enrichment_log_path, enrichment_row)

        return {
            "turn_log_id": turn_log_id,
            "enrichment_log_id": enrichment_log_id,
        }

    def capture_feedback(
        self,
        *,
        session_id: str,
        user_message: str,
        feedback_label: str,
        recommendation: dict[str, Any] | None,
        session_snapshot: dict[str, Any],
    ) -> str:
        feedback_log_id = str(uuid.uuid4())
        feedback_row = {
            "id": feedback_log_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "feedback_label": feedback_label,
            "user_message": user_message,
            "recommendation": recommendation,
            "session_snapshot": session_snapshot,
            "status": "pending_review",
            "usage": "model_evaluation_and_kb_curation",
        }
        self._append_jsonl(self.feedback_log_path, feedback_row)
        return feedback_log_id

    @staticmethod
    def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
