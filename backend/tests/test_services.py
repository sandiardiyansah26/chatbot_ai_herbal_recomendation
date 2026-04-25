from __future__ import annotations

import os
import unittest
from pathlib import Path

from app.main import (
    _anamnesis_summary,
    _is_duplicate_of_previous_follow_up_question,
    _new_session,
    _should_force_final_recommendation,
    _sync_session_from_client,
)
from app.services.anamnesis import analyze_message
from app.models import ConversationTurn, ModelAssessment, Recommendation, SessionSync
from app.services.knowledge_base import KnowledgeBase, TrainingRecord
from app.services.llm_comparison import (
    DualLLMComparator,
    build_medical_prompt,
    extract_ollama_inference_metrics,
    extract_openai_inference_metrics,
    question_non_repetition_score,
    score_reply,
    strip_thinking,
)
from app.services.recommendation import build_recommendation, enhance_recommendation_preparation
from app.services.retrieval import RetrievedItem
from app.services.retrieval import RAGRetriever



def _default_data_dir() -> Path:
    current = Path(__file__).resolve()
    candidates = [
        current.parents[1] / "data" / "referensi",
        current.parents[3] / "data" / "referensi" if len(current.parents) > 3 else None,
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    return next(candidate for candidate in candidates if candidate is not None)


DATA_DIR = Path(os.getenv("HERBAL_DATA_DIR") or _default_data_dir()).resolve()


class ServiceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.kb = KnowledgeBase(DATA_DIR)
        cls.retriever = RAGRetriever(cls.kb)

    def test_loads_focused_cases(self) -> None:
        self.assertGreaterEqual(len(self.kb.cases), 6)

    def test_retrieves_ginger_for_mild_nausea(self) -> None:
        anamnesis = analyze_message("saya mual ringan sejak tadi pagi")
        results = self.retriever.retrieve_cases("saya mual ringan sejak tadi pagi", anamnesis)
        self.assertTrue(results)
        self.assertEqual(results[0].id, "case_001")

    def test_reranks_serai_cinnamon_honey_for_throat(self) -> None:
        anamnesis = analyze_message("tenggorokan tidak nyaman sejak kemarin tanpa sesak")
        cases = self.retriever.retrieve_cases("tenggorokan tidak nyaman sejak kemarin tanpa sesak", anamnesis)
        self.assertTrue(cases)
        self.assertEqual(cases[0].id, "case_002")
        contexts = self.retriever.retrieve_context_for_case("tenggorokan tidak nyaman", cases[0].payload, anamnesis)
        self.assertTrue(contexts)
        self.assertIn("Serai-Kayu Manis-Madu", [context.title for context in contexts[:2]])

    def test_retrieves_wedang_sinden_for_light_cough_pollution(self) -> None:
        message = "saya batuk ringan dan suara serak karena polusi sejak kemarin"
        anamnesis = analyze_message(message)
        cases = self.retriever.retrieve_cases(message, anamnesis)
        self.assertTrue(cases)
        self.assertEqual(cases[0].id, "case_007")

    def test_light_itchy_rash_is_follow_up_not_red_flag(self) -> None:
        message = "saya gatal gatal dan timbul ruam merah"
        anamnesis = analyze_message(message)
        self.assertEqual(anamnesis.red_flags, [])
        cases = self.retriever.retrieve_cases(message, anamnesis)
        self.assertTrue(cases)
        self.assertEqual(cases[0].id, "case_009")

    def test_anamnesis_extracts_answered_symptom_slots_and_duration(self) -> None:
        anamnesis = analyze_message("saya sedang sakit kepala dan disertai demam sudah 3 hari")
        self.assertIn("demam", anamnesis.present_symptoms)
        self.assertIn("sakit kepala", anamnesis.present_symptoms)
        self.assertEqual(anamnesis.duration_text, "sudah 3 hari")
        self.assertIn("durasi sudah disebut: sudah 3 hari", anamnesis.answered_slots)
        self.assertTrue(anamnesis.has_progression_signal or anamnesis.has_intensity_signal)

    def test_follow_up_scoring_penalizes_repeated_answered_symptom_question(self) -> None:
        summary = {
            "present_symptoms": ["demam", "sakit kepala"],
            "absent_symptoms": [],
            "duration_text": "sudah 3 hari",
            "has_duration_signal": True,
        }
        repeated = ModelAssessment(follow_up_question="Apakah ada demam dan sakit kepala?")
        specific = ModelAssessment(
            follow_up_question=(
                "Berapa suhu tertinggi, dan apakah ada nyeri belakang mata, ruam, muntah, atau sangat lemas?"
            )
        )
        self.assertLess(question_non_repetition_score(repeated, summary), 0.5)
        self.assertEqual(question_non_repetition_score(specific, summary), 1.0)

    def test_medical_prompt_warns_models_not_to_repeat_answered_slots(self) -> None:
        summary = {
            "present_symptoms": ["demam", "sakit kepala"],
            "absent_symptoms": [],
            "duration_text": "sudah 3 hari",
            "answered_slots": [
                "gejala disebut ada: demam",
                "gejala disebut ada: sakit kepala",
                "durasi sudah disebut: sudah 3 hari",
            ],
            "has_duration_signal": True,
        }
        prompt = build_medical_prompt(
            user_message="saya sakit kepala dan demam sudah 3 hari",
            response_type="follow_up",
            conversation_history=[],
            retrieved_context=[],
            anamnesis_summary=summary,
            red_flags=[],
            question_count=0,
            max_questions=3,
            force_final=False,
        )
        self.assertIn("Informasi yang sudah dijawab user", prompt)
        self.assertIn("demam", prompt)
        self.assertIn("sudah 3 hari", prompt)
        self.assertIn("Jangan menanyakan ulang", prompt)

    def test_duplicate_follow_up_question_is_detected_from_history(self) -> None:
        asked_questions = [
            "Bagaimana intensitas gejala demam dan sakit tenggorokan Anda saat ini?",
        ]
        self.assertTrue(
            _is_duplicate_of_previous_follow_up_question(
                "Bagaimana intensitas gejala demam dan sakit tenggorokan Anda saat ini?",
                asked_questions,
            )
        )
        self.assertTrue(
            _is_duplicate_of_previous_follow_up_question(
                "Bagaimana intensitas demam dan sakit tenggorokan Anda sekarang?",
                asked_questions,
            )
        )
        self.assertFalse(
            _is_duplicate_of_previous_follow_up_question(
                "Apakah menelan terasa sangat sakit atau ada bercak putih di amandel?",
                asked_questions,
            )
        )

    def test_session_sync_restores_follow_up_progress_after_restart(self) -> None:
        session = _new_session()
        sync = SessionSync(
            turns=[
                ConversationTurn(role="user", content="saya sakit tenggorokan dan demam sudah 3 hari"),
                ConversationTurn(
                    role="assistant",
                    content="Pertanyaan anamnesis yang perlu kamu jawab: Bagaimana intensitas gejala demam dan sakit tenggorokan Anda saat ini?",
                ),
                ConversationTurn(role="user", content="lebih sakit saat bangun tidur dan saat menelan"),
            ],
            question_count=1,
            conversation_stage="anamnesis_follow_up_1",
            completed=False,
            suspected_conditions=["tenggorokan tidak nyaman"],
            asked_follow_up_questions=["Bagaimana intensitas gejala demam dan sakit tenggorokan Anda saat ini?"],
        )
        _sync_session_from_client(session, sync)
        self.assertEqual(session["question_count"], 1)
        self.assertEqual(session["conversation_stage"], "anamnesis_follow_up_1")
        self.assertEqual(len(session["turns"]), 3)
        self.assertEqual(
            session["asked_follow_up_questions"],
            ["Bagaimana intensitas gejala demam dan sakit tenggorokan Anda saat ini?"],
        )

    def test_force_final_recommendation_when_duration_safety_and_detail_are_answered(self) -> None:
        session = _new_session()
        session["question_count"] = 2
        anamnesis = analyze_message(
            "saya sakit tenggorokan dan demam sudah 3 hari. lebih sakit saat bangun tidur dan saat menelan. tidak sesak dan tidak muntah terus"
        )
        summary = _anamnesis_summary(anamnesis, keluhan_ringan="tenggorokan tidak nyaman", session=session)
        assessment = ModelAssessment(scope="supported", enough_information=False, follow_up_question="pertanyaan lama")
        self.assertTrue(_should_force_final_recommendation(session, summary, assessment))

    def test_loads_herbal_preparation_training_records(self) -> None:
        record_ids = {record.id for record in self.kb.training_records}
        self.assertIn("prep_wedang_sinden_detail", record_ids)

    def test_recommendation_preparation_can_be_enhanced_from_training_record(self) -> None:
        recommendation = Recommendation(
            keluhan_ringan="batuk ringan",
            ramuan="Wedang Sinden",
            bahan=["kencur", "jahe", "sereh"],
            cara_pengolahan="Cuci bahan, seduh, lalu minum hangat.",
            dosis_penggunaan="250 ml, 1 kali sehari.",
            catatan_kewaspadaan="Tidak untuk sesak atau batuk darah.",
            sumber_ringkas="dataset awal",
            disclaimer="bukan diagnosis medis final",
        )
        record = TrainingRecord(
            id="prep_wedang_sinden_detail",
            topic="batuk ringan",
            formula_name="Wedang Sinden",
            ingredients=["kencur", "jahe", "sereh"],
            symptoms=["batuk ringan"],
            preparation="Cuci kencur, jahe, dan sereh. Rajang tipis, seduh dengan air mendidih, tutup 10 menit, lalu sajikan hangat.",
            dosage="250 ml, 1 kali sehari.",
            safety_notes="Tidak untuk sesak.",
            evidence_level="official_preparation_detail",
            source_title="Kencur untuk Atasi Dampak Polusi",
            source_url="https://ayosehat.kemkes.go.id/kencur-untuk-atasi-dampak-polusi",
            curation_method="test",
        )
        item = RetrievedItem(
            id="training:prep_wedang_sinden_detail",
            type="training",
            title="Wedang Sinden",
            score=1.0,
            source=record.source_url,
            payload=record,
            evidence_level=record.evidence_level,
        )
        enhanced = enhance_recommendation_preparation(recommendation, [item])
        self.assertIsNotNone(enhanced)
        self.assertIn("Rajang tipis", enhanced.cara_pengolahan)

    def test_rash_with_high_fever_is_red_flag(self) -> None:
        anamnesis = analyze_message("ruam merah disertai demam tinggi")
        self.assertIn("demam tinggi", anamnesis.red_flags)

    def test_negated_red_flag_is_not_triggered(self) -> None:
        anamnesis = analyze_message("tidak ada demam tinggi dan tidak sesak")
        self.assertEqual(anamnesis.red_flags, [])

    def test_real_red_flag_is_triggered(self) -> None:
        anamnesis = analyze_message("saya demam tinggi dan sesak")
        self.assertIn("demam tinggi", anamnesis.red_flags)
        self.assertIn("sesak napas", anamnesis.red_flags)

    def test_llm_scoring_rewards_safe_grounded_recommendation(self) -> None:
        case = self.kb.case_by_id("case_002")
        recommendation_text = (
            "Ramuan Serai-Kayu Manis-Madu. Bahan serai, kayu manis, dan madu. "
            "Cara pengolahan direbus lalu diminum hangat. Dosis secukupnya sesuai konteks. "
            "Perhatikan kewaspadaan alergi dan ini bukan diagnosis medis final atau pengganti tenaga kesehatan. "
            "Sumber: data referensi."
        )
        score, breakdown = score_reply(
            reply=recommendation_text,
            response_type="recommendation",
            recommendation=build_recommendation(case) if case else None,
            retrieved_context=[],
            red_flags=[],
        )
        self.assertGreater(score, 0.7)
        self.assertGreater(breakdown["safety"], 0.6)

    def test_strip_thinking_removes_model_private_reasoning(self) -> None:
        self.assertEqual(strip_thinking("<think>rahasia</think>Jawaban final").strip(), "Jawaban final")
        self.assertEqual(strip_thinking("<|channel>thought\nrahasia<channel|>Jawaban final").strip(), "Jawaban final")

    def test_llm_num_predict_uses_response_specific_budget(self) -> None:
        comparator = DualLLMComparator(
            enabled=False,
            num_predict_by_response_type={
                "default": 210,
                "recommendation": 280,
                "follow_up": 120,
                "red_flag": 90,
                "out_of_scope": 100,
            },
        )
        self.assertEqual(comparator._num_predict_for("recommendation"), 280)
        self.assertEqual(comparator._num_predict_for("follow_up"), 120)
        self.assertEqual(comparator._num_predict_for("unknown"), 210)

    def test_openai_comparison_model_is_optional(self) -> None:
        without_key = DualLLMComparator(enabled=True, openai_api_key="", enable_openai=True)
        self.assertNotIn("gpt-4o", without_key.models)
        self.assertFalse(without_key.health()["openai"]["enabled"])

        with_key = DualLLMComparator(
            enabled=True,
            openai_api_key="test-key",
            openai_model="gpt-4o",
            enable_openai=True,
        )
        self.assertIn("gpt-4o", with_key.models)
        self.assertEqual(with_key.health()["providers"]["gpt-4o"], "openai")

    def test_extract_ollama_inference_metrics_includes_token_rates(self) -> None:
        metrics = extract_ollama_inference_metrics(
            {
                "model": "deepseek-r1:7b",
                "done_reason": "stop",
                "total_duration": 126_338_287_630,
                "load_duration": 8_614_980_706,
                "prompt_eval_count": 2_589,
                "prompt_eval_duration": 61_369_769_851,
                "eval_count": 598,
                "eval_duration": 55_898_318_824,
            }
        )
        self.assertEqual(metrics["prompt_eval_count"], 2589)
        self.assertEqual(metrics["eval_count"], 598)
        self.assertEqual(metrics["done_reason"], "stop")
        self.assertAlmostEqual(float(metrics["prompt_eval_rate_tps"]), 42.19, places=2)
        self.assertAlmostEqual(float(metrics["eval_rate_tps"]), 10.70, places=2)

    def test_extract_openai_inference_metrics_reads_usage_fields(self) -> None:
        metrics = extract_openai_inference_metrics(
            {
                "model": "gpt-4o",
                "choices": [{"finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 110,
                    "completion_tokens": 84,
                    "total_tokens": 194,
                    "prompt_tokens_details": {"cached_tokens": 12},
                    "completion_tokens_details": {"reasoning_tokens": 9},
                },
            }
        )
        self.assertEqual(metrics["prompt_tokens"], 110)
        self.assertEqual(metrics["completion_tokens"], 84)
        self.assertEqual(metrics["total_tokens"], 194)
        self.assertEqual(metrics["cached_prompt_tokens"], 12)
        self.assertEqual(metrics["reasoning_tokens"], 9)
        self.assertEqual(metrics["finish_reason"], "stop")

    def test_llm_comparison_is_skipped_for_guardrail_responses(self) -> None:
        self.assertTrue(DualLLMComparator._should_compare("recommendation"))
        self.assertTrue(DualLLMComparator._should_compare("follow_up"))
        self.assertFalse(DualLLMComparator._should_compare("red_flag"))
        self.assertFalse(DualLLMComparator._should_compare("out_of_scope"))


if __name__ == "__main__":
    unittest.main()
