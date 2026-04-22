from __future__ import annotations

import os
import unittest
from pathlib import Path

from app.services.anamnesis import analyze_message
from app.services.knowledge_base import KnowledgeBase
from app.services.llm_comparison import DualLLMComparator, score_reply, strip_thinking
from app.services.recommendation import build_recommendation
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

    def test_llm_comparison_is_skipped_for_guardrail_responses(self) -> None:
        self.assertTrue(DualLLMComparator._should_compare("recommendation"))
        self.assertTrue(DualLLMComparator._should_compare("follow_up"))
        self.assertFalse(DualLLMComparator._should_compare("red_flag"))
        self.assertFalse(DualLLMComparator._should_compare("out_of_scope"))


if __name__ == "__main__":
    unittest.main()
