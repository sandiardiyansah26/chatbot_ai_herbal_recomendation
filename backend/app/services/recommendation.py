from __future__ import annotations

from app.models import Recommendation, RetrievedContext
from app.services.knowledge_base import CaseEntry
from app.services.retrieval import RetrievedItem


DISCLAIMER = (
    "Informasi ini bersifat rekomendasi awal dan edukasi, bukan diagnosis medis "
    "final dan bukan pengganti konsultasi tenaga kesehatan. Hasil bisa saja tidak "
    "cocok untuk semua orang."
)


def build_recommendation(case: CaseEntry) -> Recommendation:
    return Recommendation(
        keluhan_ringan=case.keluhan_ringan,
        ramuan=case.ramuan_rekomendasi,
        bahan=case.bahan,
        cara_pengolahan=case.cara_pengolahan,
        dosis_penggunaan=case.dosis_penggunaan,
        catatan_kewaspadaan=case.catatan_kewaspadaan,
        sumber_ringkas=case.sumber_ringkas,
        disclaimer=DISCLAIMER,
    )


def build_recommendation_reply(
    recommendation: Recommendation,
    contexts: list[RetrievedItem] | None = None,
    anamnesis_summary: dict[str, object] | None = None,
) -> str:
    bahan = ", ".join(recommendation.bahan)
    source_lines = _format_context_sources(contexts or [])
    summary_lines = _format_anamnesis_summary(anamnesis_summary or {})
    return (
        "Ringkasan anamnesis:\n"
        f"{summary_lines}\n\n"
        f"Berdasarkan knowledge base dan konteks retrieval, keluhan masih saya posisikan sebagai "
        f"'{recommendation.keluhan_ringan}' selama tidak ada tanda bahaya. "
        "Rekomendasi awal ramuan herbal yang relevan:\n\n"
        f"Ramuan: {recommendation.ramuan}\n"
        f"Bahan: {bahan}\n"
        f"Cara pengolahan: {recommendation.cara_pengolahan}\n"
        f"Dosis/kisaran penggunaan: {recommendation.dosis_penggunaan}\n"
        f"Catatan kewaspadaan: {recommendation.catatan_kewaspadaan}\n"
        f"Sumber ringkas: {recommendation.sumber_ringkas}\n\n"
        f"Konteks RAG yang dipakai:\n{source_lines}\n\n"
        f"{recommendation.disclaimer}"
    )


def build_follow_up_reply(case: CaseEntry, questions: list[str] | None = None, source_title: str | None = None) -> str:
    selected_questions = questions or [case.pertanyaan_anamnesis]
    question_lines = "\n".join(f"{index}. {question}" for index, question in enumerate(selected_questions[:6], start=1))
    source_line = f"\n\nPertanyaan ini memakai rujukan anamnesis: {source_title}." if source_title else ""
    return (
        f"Saya menangkap keluhan kamu mengarah ke '{case.keluhan_ringan}'. "
        "Sebelum memberi rekomendasi ramuan, saya perlu memastikan keluhan ini masih ringan dan tidak ada tanda bahaya.\n\n"
        "Pertanyaan anamnesis yang perlu dijawab:\n"
        f"{question_lines}"
        f"{source_line}\n\n"
        "Jawab singkat saja. Kalau ada tanda berat seperti demam tinggi, sesak, darah, dehidrasi, "
        "nyeri hebat, atau keluhan pada bayi/kehamilan, sebaiknya prioritaskan pemeriksaan medis."
    )


def build_red_flag_reply(red_flags: list[str]) -> str:
    flags = ", ".join(red_flags)
    return (
        f"Saya mendeteksi tanda yang perlu diwaspadai: {flags}. "
        "Untuk keamanan, sistem ini tidak akan memposisikan ramuan herbal sebagai penanganan utama pada kondisi tersebut. "
        "Sebaiknya segera konsultasi ke tenaga kesehatan atau fasilitas kesehatan terdekat, terutama bila gejala berat, menetap, atau memburuk.\n\n"
        f"{DISCLAIMER}"
    )


def build_out_of_scope_reply() -> str:
    return (
        "Saya belum menemukan keluhan ringan yang sesuai dengan knowledge base ramuan herbal saat ini. "
        "Coba jelaskan keluhan utama dengan singkat, misalnya mual ringan, tenggorokan tidak nyaman, "
        "nafsu makan menurun, pegal ringan, diare ringan tanpa darah, atau badan kurang fit.\n\n"
        f"{DISCLAIMER}"
    )


def to_context(items: list[RetrievedItem]) -> list[RetrievedContext]:
    return [
        RetrievedContext(
            id=item.id,
            type=item.type,
            title=item.title,
            score=item.score,
            source=item.source,
            evidence_level=item.evidence_level,
            matched_terms=item.matched_terms,
        )
        for item in items
    ]


def _format_context_sources(items: list[RetrievedItem]) -> str:
    if not items:
        return "- Tidak ada konteks tambahan."
    lines = []
    for item in items[:5]:
        evidence = f", evidence={item.evidence_level}" if item.evidence_level else ""
        lines.append(f"- {item.type}:{item.title} (score={item.score}{evidence})")
    return "\n".join(lines)


def _format_anamnesis_summary(summary: dict[str, object]) -> str:
    if not summary:
        return "- Keluhan ringan terdeteksi dari percakapan."
    keluhan = summary.get("keluhan_ringan") or "-"
    durasi = "ada" if summary.get("has_duration_signal") else "belum jelas"
    safety = "ada klarifikasi tanda bahaya" if summary.get("has_safety_clearance_signal") else "belum ada klarifikasi tanda bahaya eksplisit"
    symptoms = summary.get("detected_symptoms") or []
    symptoms_text = ", ".join(symptoms) if isinstance(symptoms, list) and symptoms else "-"
    return f"- Keluhan: {keluhan}\n- Gejala terdeteksi: {symptoms_text}\n- Durasi: {durasi}\n- Safety check: {safety}"
