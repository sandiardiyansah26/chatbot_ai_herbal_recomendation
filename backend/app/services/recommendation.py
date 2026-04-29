from __future__ import annotations

from app.models import ModelAssessment, Recommendation, RetrievedContext
from app.services.knowledge_base import AnamnesisEntry, CaseEntry, KnowledgeChunk, TrainingRecord
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


def build_assessment_recommendation(
    assessment: ModelAssessment,
    fallback_case: CaseEntry | None = None,
) -> Recommendation | None:
    herbal_name = assessment.recommended_herbal_name or (fallback_case.ramuan_rekomendasi if fallback_case else "")
    ingredients = assessment.ingredients or (fallback_case.bahan if fallback_case else [])
    preparation = assessment.preparation or (fallback_case.cara_pengolahan if fallback_case else "")
    dosage = assessment.dosage or (fallback_case.dosis_penggunaan if fallback_case else "")
    warning_notes = assessment.warning_notes or (fallback_case.catatan_kewaspadaan if fallback_case else "")
    source_hint = assessment.source_hint or (fallback_case.sumber_ringkas if fallback_case else "")
    target_condition = (
        (assessment.suspected_conditions[0] if assessment.suspected_conditions else "")
        or (fallback_case.keluhan_ringan if fallback_case else "")
    )

    if not herbal_name:
        return None

    return Recommendation(
        keluhan_ringan=target_condition or "dugaan kondisi berbasis anamnesis",
        ramuan=herbal_name,
        bahan=ingredients,
        cara_pengolahan=preparation or "Gunakan sesuai sumber yang paling relevan dan sesuaikan dengan kondisi pengguna.",
        dosis_penggunaan=dosage or "Gunakan seperlunya dan hentikan bila tidak cocok.",
        catatan_kewaspadaan=warning_notes or "Hentikan bila keluhan memburuk atau muncul tanda bahaya.",
        sumber_ringkas=source_hint or "Knowledge base terkurasi",
        disclaimer=DISCLAIMER,
    )


def enhance_recommendation_preparation(
    recommendation: Recommendation | None,
    contexts: list[RetrievedItem] | None = None,
) -> Recommendation | None:
    if recommendation is None:
        return None

    best_preparation = recommendation.cara_pengolahan
    best_source = recommendation.sumber_ringkas
    for item in contexts or []:
        payload = item.payload
        if isinstance(payload, KnowledgeChunk):
            payload = payload.payload
        if not isinstance(payload, TrainingRecord):
            continue
        if not _formula_matches(recommendation.ramuan, payload.formula_name):
            continue
        preparation = payload.preparation.strip()
        if len(preparation) > len(best_preparation) + 35:
            best_preparation = preparation
            best_source = payload.source_title or payload.source_url or best_source

    if best_preparation == recommendation.cara_pengolahan:
        return recommendation
    return recommendation.model_copy(
        update={
            "cara_pengolahan": best_preparation,
            "sumber_ringkas": best_source,
        }
    )


def build_preparation_detail_reply(recommendation: Recommendation) -> str:
    steps = _split_preparation_steps(recommendation.cara_pengolahan)
    bahan = ", ".join(recommendation.bahan) or "bahan sesuai kartu rekomendasi"
    return (
        f"Detail cara pengolahan {recommendation.ramuan}:\n"
        f"Bahan yang dipakai: {bahan}.\n\n"
        "Langkah pengolahan:\n"
        + "\n".join(f"{index}. {step}" for index, step in enumerate(steps, start=1))
        + "\n\n"
        f"Dosis/kisaran penggunaan: {recommendation.dosis_penggunaan}\n"
        f"Catatan kewaspadaan: {recommendation.catatan_kewaspadaan}\n\n"
        "Gunakan bahan bersih, alat bersih, dan jangan menyimpan ramuan terlalu lama. "
        "Hentikan penggunaan bila muncul gatal, sesak, nyeri perut, mual berat, ruam memburuk, atau keluhan tidak cocok.\n\n"
        f"{DISCLAIMER}"
    )


def _formula_matches(left: str, right: str) -> bool:
    left_terms = {term for term in _tokenize_formula_name(left) if len(term) > 2}
    right_terms = {term for term in _tokenize_formula_name(right) if len(term) > 2}
    if not left_terms or not right_terms:
        return False
    return len(left_terms & right_terms) >= max(1, min(len(left_terms), len(right_terms)) // 2)


def _tokenize_formula_name(value: str) -> list[str]:
    normalized = value.lower().replace("/", " ").replace("-", " ")
    return [part.strip() for part in normalized.split() if part.strip()]


def build_feedback_reply(feedback_label: str, recommendation: Recommendation | None = None) -> str:
    ramuan = recommendation.ramuan if recommendation else "rekomendasi tadi"
    if feedback_label == "helpful":
        return (
            f"Terima kasih, saya catat bahwa rekomendasi {ramuan} membantu. "
            "Masukan ini akan disimpan sebagai data evaluasi kualitas model dan knowledge base.\n\n"
            "Tetap pantau gejala. Bila keluhan memburuk, muncul demam tinggi, sesak, nyeri berat, perdarahan, "
            "dehidrasi, atau reaksi alergi, prioritaskan pemeriksaan tenaga kesehatan."
        )
    if feedback_label == "not_helpful":
        return (
            f"Terima kasih, saya catat bahwa rekomendasi {ramuan} belum membantu atau kurang cocok. "
            "Masukan ini penting untuk evaluasi model dan kurasi knowledge base berikutnya.\n\n"
            "Boleh jelaskan singkat bagian yang kurang cocok: gejala tidak membaik, cara pengolahan kurang jelas, "
            "bahan sulit didapat, atau muncul efek tidak nyaman? Jika ada tanda bahaya, sebaiknya segera periksa."
        )
    if feedback_label == "unclear_preparation":
        return (
            "Terima kasih, saya catat bahwa bagian cara pengolahan masih kurang jelas. "
            "Saya akan gunakan masukan ini untuk memperbaiki data training dan format jawaban berikutnya.\n\n"
            "Kamu juga bisa klik atau ketik: Jelaskan cara pengolahan ramuan ini lebih detail."
        )
    return (
        "Terima kasih, masukan kamu sudah saya simpan sebagai data evaluasi. "
        "Jika ingin, beri tahu bagian mana yang perlu diperbaiki: hasil, rasa, bahan, dosis, atau cara pengolahan."
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


def build_assessment_follow_up_reply(
    assessment: ModelAssessment,
    anamnesis_summary: dict[str, object] | None = None,
    *,
    question_number: int,
    max_questions: int,
) -> str:
    summary = anamnesis_summary or {}
    suspected = summary.get("keluhan_ringan") or ", ".join(assessment.suspected_conditions[:3]) or "beberapa dugaan kondisi ringan"
    patient_summary = _format_doctor_style_patient_summary(summary, suspected)
    rationale = assessment.follow_up_rationale or "Pertanyaan ini dipilih untuk memperjelas dugaan kondisi dan menyingkirkan tanda bahaya."
    question = assessment.follow_up_question or "Bisa ceritakan gejala yang paling mengganggu saat ini?"
    red_flags = (
        f"Tanda yang tetap perlu diwaspadai: {', '.join(assessment.red_flags[:4])}."
        if assessment.red_flags
        else "Jika ada demam tinggi, sesak, nyeri hebat, perdarahan, dehidrasi, atau kondisi memburuk, segera periksa ke tenaga kesehatan."
    )
    return (
        f"Baik, saya tangkap dulu ya. {patient_summary}\n\n"
        "Sebelum saya menyarankan langkah awal atau ramuan pendamping, saya perlu bertanya seperti dokter saat anamnesis: "
        "satu hal yang paling menentukan keamanan dulu.\n\n"
        f"{question}\n\n"
        "Jawab singkat saja sesuai yang kamu rasakan. Kalau tidak ada, cukup tulis 'tidak ada'.\n\n"
        f"Kenapa saya tanya ini: {rationale} {red_flags}\n\n"
        f"Tahap anamnesis: {question_number}/{max_questions}. "
        f"{DISCLAIMER}"
    )


def build_follow_up_reply(case: CaseEntry, questions: list[str] | None = None, source_title: str | None = None) -> str:
    selected_questions = questions or [case.pertanyaan_anamnesis]
    question_lines = "\n".join(f"{index}. {question}" for index, question in enumerate(selected_questions[:6], start=1))
    source_line = f"\n\nPertanyaan ini memakai rujukan anamnesis: {source_title}." if source_title else ""
    return (
        f"Baik, saya tangkap keluhan kamu sementara mengarah ke '{case.keluhan_ringan}'. "
        "Sebelum membahas ramuan, saya perlu memastikan satu hal seperti dokter saat anamnesis agar sarannya tidak terburu-buru.\n\n"
        f"{question_lines}"
        f"{source_line}\n\n"
        "Jawab singkat saja sesuai yang kamu rasakan. Kalau tidak ada, cukup tulis 'tidak ada'.\n\n"
        "Kalau ada tanda berat seperti demam tinggi, sesak, darah, dehidrasi, nyeri hebat, "
        "atau keluhan pada bayi/kehamilan, sebaiknya prioritaskan pemeriksaan medis.\n\n"
        f"{DISCLAIMER}"
    )


def build_assessment_recommendation_reply(
    assessment: ModelAssessment,
    recommendation: Recommendation | None,
    contexts: list[RetrievedItem] | None = None,
    anamnesis_summary: dict[str, object] | None = None,
) -> str:
    summary_lines = _format_anamnesis_summary(anamnesis_summary or {})
    suspected = ", ".join(assessment.suspected_conditions[:3]) or "dugaan kondisi berbasis anamnesis"
    reasoning = assessment.reasoning or "Dugaan kondisi disusun dari gejala yang muncul selama percakapan dan konteks retrieval."
    warning = assessment.warning_notes or "Segera periksa ke tenaga kesehatan bila keluhan memburuk atau muncul tanda bahaya."
    source_lines = _format_context_sources(contexts or [])
    final_answer = assessment.final_answer or (
        f"Untuk saat ini keluhan lebih mengarah ke {suspected}, tetapi tetap ini belum bisa dianggap sebagai diagnosis medis final."
    )

    blocks = [
        "Ringkasan anamnesis:\n" f"{summary_lines}",
        f"Dugaan kondisi yang paling mungkin saat ini: {suspected}.",
        f"Pertimbangan utama: {reasoning}",
        final_answer,
    ]

    if recommendation:
        blocks.append(
            "Rekomendasi ramuan herbal awal:\n"
            f"Ramuan: {recommendation.ramuan}\n"
            f"Bahan: {', '.join(recommendation.bahan)}\n"
            f"Cara pengolahan: {recommendation.cara_pengolahan}\n"
            f"Dosis/kisaran penggunaan: {recommendation.dosis_penggunaan}\n"
            f"Catatan kewaspadaan: {recommendation.catatan_kewaspadaan}\n"
            f"Sumber ringkas: {recommendation.sumber_ringkas}"
        )
    else:
        blocks.append(
            "Saya belum menemukan ramuan herbal yang cukup aman dan cukup ter-ground ke knowledge base untuk langsung direkomendasikan pada kondisi ini."
        )

    blocks.append(f"Konteks RAG yang dipakai:\n{source_lines}")
    blocks.append(f"Catatan keselamatan: {warning}\n\n{DISCLAIMER}")
    return "\n\n".join(blocks)


def _split_preparation_steps(preparation: str) -> list[str]:
    normalized = preparation.replace(";", ".").strip()
    pieces = [
        piece.strip(" .")
        for piece in normalized.split(".")
        if piece.strip(" .")
    ]
    if len(pieces) >= 2:
        return pieces[:8]
    comma_pieces = [
        piece.strip(" ,")
        for piece in normalized.split(",")
        if piece.strip(" ,")
    ]
    return comma_pieces[:8] or [preparation or "Ikuti cara pengolahan pada kartu rekomendasi."]


def build_red_flag_reply(red_flags: list[str]) -> str:
    flags = ", ".join(red_flags)
    return (
        f"Saya mendeteksi tanda yang perlu diwaspadai: {flags}. "
        "Untuk kondisi seperti ini, saya tidak akan menyarankan ramuan herbal sebagai penanganan utama. "
        "Sebaiknya segera konsultasi ke tenaga kesehatan atau fasilitas kesehatan terdekat, "
        "terutama bila gejala berat, menetap, atau memburuk.\n\n"
        f"{DISCLAIMER}"
    )


def build_scope_referral_reply(
    assessment: ModelAssessment,
    anamnesis_summary: dict[str, object] | None = None,
) -> str:
    summary = anamnesis_summary or {}
    suspected = _patient_friendly_suspected_conditions(assessment.suspected_conditions, summary)
    referral_reason = _patient_friendly_referral_reason(assessment.scope_reason or assessment.reasoning, summary)
    warning = _patient_friendly_warning(
        assessment.warning_notes or "Prioritaskan pemeriksaan tenaga kesehatan dan jangan menunda bila keluhan memburuk."
    )
    return (
        f"Baik, dari cerita kamu sejauh ini, keluhan lebih aman diposisikan sebagai {suspected}.\n\n"
        f"Alasannya: {referral_reason}\n\n"
        f"Yang sebaiknya kamu lakukan sekarang: {warning}\n\n"
        "Saya belum akan memaksakan ramuan herbal sebagai solusi utama, karena pada demam atau keluhan yang masih mungkin infeksi, "
        "yang paling penting adalah memastikan tidak ada tanda bahaya dan memantau perkembangan gejala.\n\n"
        f"{DISCLAIMER}"
    )


def build_out_of_scope_reply() -> str:
    return (
        "Saya belum menemukan keluhan ringan yang sesuai dengan knowledge base ramuan herbal saat ini. "
        "Coba jelaskan keluhan utama dengan singkat, misalnya mual ringan, tenggorokan tidak nyaman, "
        "nafsu makan menurun, pegal ringan, diare ringan tanpa darah, atau badan kurang fit.\n\n"
        f"{DISCLAIMER}"
    )


def build_medical_guidance_reply(
    items: list[RetrievedItem],
    anamnesis_summary: dict[str, object] | None = None,
) -> str:
    disease_record = _primary_disease_record(items)
    anamnesis_record = _primary_anamnesis_record(items)
    title = (
        (disease_record.formula_name if disease_record else "")
        or (anamnesis_record.suspected_condition if anamnesis_record else "")
        or "kondisi yang perlu evaluasi medis"
    )
    symptoms = disease_record.symptoms[:5] if disease_record else []
    questions = (
        disease_record.screening_questions[:4]
        if disease_record and disease_record.screening_questions
        else (anamnesis_record.required_questions[:4] if anamnesis_record else [])
    )
    warning_signs = (
        disease_record.warning_signs[:4]
        if disease_record and disease_record.warning_signs
        else (anamnesis_record.red_flag_questions[:3] if anamnesis_record else [])
    )
    prevention_steps = disease_record.prevention_steps[:4] if disease_record else []
    care_recommendation = (
        (disease_record.care_recommendation if disease_record else "")
        or (anamnesis_record.triage_action if anamnesis_record else "")
        or "Untuk keamanan, sebaiknya periksa ke tenaga kesehatan dan jangan menjadikan herbal sebagai terapi utama."
    )
    diagnosis_summary = disease_record.diagnosis_summary if disease_record else ""
    overview = disease_record.overview if disease_record else ""
    summary_lines = _format_anamnesis_summary(anamnesis_summary or {})

    blocks = [
        "Ringkasan anamnesis:\n" f"{summary_lines}",
        (
            f"Keluhan kamu belum saya posisikan sebagai keluhan ringan yang aman ditangani dengan ramuan herbal saja. "
            f"Dari knowledge base, ada konteks yang mengarah ke '{title}'."
        ),
    ]
    if symptoms:
        blocks.append(f"Gejala kunci yang sering terkait: {', '.join(symptoms)}.")
    if overview:
        blocks.append(f"Ringkasan singkat: {overview}")
    if questions:
        blocks.append(
            "Hal yang perlu dipastikan lebih lanjut:\n"
            + "\n".join(f"- {question}" for question in questions)
        )
    if warning_signs:
        blocks.append(
            "Tanda bahaya atau alasan untuk segera diperiksa:\n"
            + "\n".join(f"- {item}" for item in warning_signs)
        )
    if diagnosis_summary:
        blocks.append(f"Pemeriksaan/penanganan medis yang lazim: {diagnosis_summary}")
    if prevention_steps:
        blocks.append(
            "Edukasi pencegahan yang relevan:\n"
            + "\n".join(f"- {step}" for step in prevention_steps)
        )
    blocks.append(
        f"Arahan awal: {care_recommendation}\n\n"
        "Bila keluhan mengarah ke infeksi tropis, penyakit menular kronis, atau infeksi jamur yang luas/berulang, "
        "prioritas utamanya adalah pemeriksaan medis. Herbal bila ada hanya boleh diposisikan sebagai pendamping edukatif, "
        "bukan pengganti terapi utama."
    )
    blocks.append(DISCLAIMER)
    return "\n\n".join(blocks)


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


def _format_doctor_style_patient_summary(summary: dict[str, object], suspected: str) -> str:
    symptoms = summary.get("present_symptoms") or summary.get("detected_symptoms") or []
    symptom_text = ", ".join(symptoms[:4]) if isinstance(symptoms, list) and symptoms else suspected
    duration = summary.get("duration_text") or ("durasi sudah disebut" if summary.get("has_duration_signal") else "")
    absent = summary.get("absent_symptoms") or []
    parts = [f"Keluhan yang sudah saya catat: {symptom_text}."]
    if duration:
        parts.append(f"Durasi: {duration}.")
    if isinstance(absent, list) and absent:
        parts.append(f"Gejala yang sudah kamu sangkal: {', '.join(absent[:4])}.")
    if summary.get("has_safety_clearance_signal"):
        parts.append("Sebagian tanda bahaya sudah kamu jawab tidak ada.")
    else:
        parts.append("Tanda bahaya belum lengkap tersaring.")
    return " ".join(parts)


def _patient_friendly_suspected_conditions(
    suspected_conditions: list[str],
    summary: dict[str, object],
) -> str:
    suspected_text = ", ".join(item for item in suspected_conditions[:3] if item)
    if suspected_text:
        lowered = suspected_text.lower()
        if "internal_medicine" in lowered or "scope model" in lowered:
            return "keluhan yang perlu pemeriksaan langsung"
        return suspected_text
    symptoms = summary.get("present_symptoms") or summary.get("detected_symptoms") or []
    if isinstance(symptoms, list) and any(str(symptom).lower() == "demam" for symptom in symptoms):
        return "demam atau infeksi yang masih perlu dipantau dan belum bisa dipastikan jenisnya"
    return "keluhan yang belum bisa dipastikan aman hanya dengan perawatan mandiri"


def _patient_friendly_referral_reason(reason: str | None, summary: dict[str, object]) -> str:
    normalized = (reason or "").strip()
    lowered = normalized.lower()
    if not normalized or "scope model" in lowered or "internal_medicine" in lowered or "safety layer" in lowered:
        symptoms = summary.get("present_symptoms") or summary.get("detected_symptoms") or []
        symptom_text = ", ".join(symptoms[:3]) if isinstance(symptoms, list) and symptoms else "keluhan yang kamu sebutkan"
        return (
            f"gejala seperti {symptom_text} perlu dilihat polanya dulu. Jika menetap, memburuk, "
            "atau muncul tanda bahaya, pemeriksaan langsung lebih aman daripada hanya mengandalkan ramuan."
        )
    cleaned = normalized.replace("scope model: internal_medicine", "perlu evaluasi medis")
    cleaned = cleaned.replace("scope model: critical", "ada tanda yang perlu perhatian segera")
    return cleaned[:260]


def _patient_friendly_warning(warning: str) -> str:
    normalized = warning.strip()
    lowered = normalized.lower()
    if not normalized:
        return "Pantau suhu dan perkembangan keluhan. Periksa ke tenaga kesehatan bila demam bertahan lebih dari 3 hari, suhu makin tinggi, keluhan memburuk, atau muncul tanda bahaya."
    if "herbal tidak diposisikan" in lowered and "tanda bahaya" in lowered:
        return "Pantau suhu dan perkembangan keluhan. Periksa ke tenaga kesehatan bila demam bertahan lebih dari 3 hari, suhu makin tinggi, keluhan memburuk, atau muncul tanda bahaya."
    return normalized


def _format_anamnesis_summary(summary: dict[str, object]) -> str:
    if not summary:
        return "- Keluhan ringan terdeteksi dari percakapan."
    keluhan = summary.get("keluhan_ringan") or "-"
    durasi = summary.get("duration_text") or ("ada" if summary.get("has_duration_signal") else "belum jelas")
    safety = "ada klarifikasi tanda bahaya" if summary.get("has_safety_clearance_signal") else "belum ada klarifikasi tanda bahaya eksplisit"
    symptoms = summary.get("present_symptoms") or summary.get("detected_symptoms") or []
    symptoms_text = ", ".join(symptoms) if isinstance(symptoms, list) and symptoms else "-"
    negative_symptoms = summary.get("absent_symptoms") or []
    negative_text = ", ".join(negative_symptoms) if isinstance(negative_symptoms, list) and negative_symptoms else "-"
    detail_answered = "sudah ada" if summary.get("has_intensity_signal") or summary.get("has_progression_signal") else "belum jelas"
    return (
        f"- Keluhan: {keluhan}\n"
        f"- Gejala terdeteksi: {symptoms_text}\n"
        f"- Gejala yang sudah disangkal: {negative_text}\n"
        f"- Durasi: {durasi}\n"
        f"- Detail intensitas/perburukan: {detail_answered}\n"
        f"- Safety check: {safety}"
    )


def _primary_disease_record(items: list[RetrievedItem]) -> TrainingRecord | None:
    for item in items:
        if item.type != "training":
            continue
        payload = item.payload
        if not isinstance(payload, KnowledgeChunk):
            continue
        if not isinstance(payload.payload, TrainingRecord):
            continue
        record = payload.payload
        if record.content_type == "disease_guidance":
            return record
    return None


def _primary_anamnesis_record(items: list[RetrievedItem]) -> AnamnesisEntry | None:
    for item in items:
        if item.type != "anamnesis":
            continue
        payload = item.payload
        if isinstance(payload, KnowledgeChunk) and isinstance(payload.payload, AnamnesisEntry):
            return payload.payload
    return None
