from __future__ import annotations

import re
from dataclasses import dataclass

from app.services.text import contains_phrase, normalize


RED_FLAG_PHRASES = {
    "demam tinggi": "demam tinggi",
    "sesak": "sesak napas",
    "susah napas": "sesak napas",
    "sulit napas": "sesak napas",
    "sulit menelan": "sulit menelan berat",
    "menelan berat": "sulit menelan berat",
    "diare berdarah": "diare berdarah",
    "bab berdarah": "diare berdarah",
    "darah": "perdarahan atau darah pada keluhan",
    "dehidrasi": "tanda dehidrasi",
    "lemas sekali": "lemas berat",
    "muntah terus": "muntah terus-menerus",
    "muntah terus-menerus": "muntah terus-menerus",
    "nyeri hebat": "nyeri hebat",
    "nyeri perut hebat": "nyeri perut hebat",
    "nyeri dada": "nyeri dada",
    "kejang": "kejang",
    "pingsan": "pingsan",
    "penurunan kesadaran": "penurunan kesadaran",
    "kaku leher": "kaku leher",
    "gangguan penglihatan": "gangguan penglihatan",
    "penglihatan kabur": "gangguan penglihatan",
    "pandangan kabur": "gangguan penglihatan",
    "bicara pelo": "kesulitan berbicara",
    "kesulitan berbicara": "kesulitan berbicara",
    "sulit berbicara": "kesulitan berbicara",
    "mati rasa": "mati rasa",
    "kelemahan satu sisi": "kelemahan satu sisi tubuh",
    "lemah satu sisi": "kelemahan satu sisi tubuh",
    "muntah darah": "muntah darah",
    "tinja hitam": "tinja hitam",
    "bab hitam": "tinja hitam",
    "batuk darah": "batuk darah",
    "berat badan turun": "penurunan berat badan",
    "turun berat badan": "penurunan berat badan",
    "keringat malam": "keringat malam",
    "batuk 3 minggu": "batuk lebih dari 3 minggu",
    "batuk lebih dari 3 minggu": "batuk lebih dari 3 minggu",
    "batuk lebih 3 minggu": "batuk lebih dari 3 minggu",
    "perdarahan rektum": "perdarahan rektum",
    "ruam berdarah": "ruam/perdarahan kulit",
    "bintik merah tidak hilang": "ruam/perdarahan kulit",
    "kulit melepuh": "ruam melepuh",
    "ruam melepuh": "ruam melepuh",
    "lepuh": "ruam melepuh",
    "bengkak wajah": "reaksi alergi berat",
    "wajah bengkak": "reaksi alergi berat",
    "bibir bengkak": "reaksi alergi berat",
    "lidah bengkak": "reaksi alergi berat",
    "tenggorokan bengkak": "reaksi alergi berat",
    "urin gelap": "urin gelap",
    "urine gelap": "urin gelap",
    "urine berdarah": "urine berdarah",
    "kencing berdarah": "urine berdarah",
    "air kencing berdarah": "urine berdarah",
    "mata kuning": "mata/kulit kuning",
    "kulit kuning": "mata/kulit kuning",
    "bercak mati rasa": "bercak kulit mati rasa",
    "kulit mati rasa": "bercak kulit mati rasa",
    "luka tidak nyeri": "luka tanpa nyeri",
    "kaki membengkak": "pembengkakan tungkai",
    "tungkai membengkak": "pembengkakan tungkai",
    "pembengkakan kaki": "pembengkakan tungkai",
    "hamil": "kehamilan perlu kehati-hatian khusus",
    "bayi": "keluhan pada bayi perlu evaluasi tenaga kesehatan",
}


CASE_HINTS = {
    "case_001": ["mual", "muntah", "enek", "eneg", "lambung", "maag", "ulu hati", "nyeri ulu hati", "perih lambung", "asam lambung", "sendawa"],
    "case_002": ["tenggorokan", "gatal tenggorokan", "sakit tenggorokan"],
    "case_003": ["nafsu makan", "malas makan", "tidak nafsu", "selera makan"],
    "case_004": ["pegal", "kurang fit", "capek", "lelah", "badan pegal"],
    "case_005": ["diare", "mencret", "bab cair", "perut mulas"],
    "case_006": ["panas badan", "demam", "demam ringan", "tidak nyaman tubuh", "nyeri ringan", "meriang", "meriang ringan", "pusing", "pusing ringan", "sakit kepala", "sakit kepala ringan"],
    "case_007": ["batuk ringan", "suara serak", "serak", "polusi"],
    "case_008": ["batuk pilek", "pilek ringan", "hidung meler", "batuk"],
    "case_009": ["gatal", "gatal-gatal", "gatal gatal", "ruam", "ruam merah", "bentol", "biduran", "alergi kulit"],
}


SYMPTOM_CONCEPTS = {
    "demam": ["demam", "panas badan", "meriang", "badan panas"],
    "sakit kepala": ["sakit kepala", "kepala sakit", "pusing", "kepala pusing", "kunang kunang"],
    "batuk": ["batuk"],
    "pilek": ["pilek", "hidung meler", "ingusan"],
    "sakit tenggorokan": ["sakit tenggorokan", "tenggorokan", "gatal tenggorokan"],
    "sesak napas": ["sesak", "sesak napas", "susah napas", "sulit napas"],
    "mual": ["mual", "enek", "eneg"],
    "muntah": ["muntah"],
    "diare": ["diare", "mencret", "bab cair"],
    "nyeri perut": ["nyeri perut", "perut sakit", "mulas", "nyeri ulu hati", "ulu hati"],
    "ruam": ["ruam", "ruam merah", "bintik merah", "bentol", "biduran"],
    "gatal": ["gatal", "gatal gatal", "gatal-gatal"],
    "nyeri otot/sendi": ["nyeri otot", "nyeri sendi", "pegal", "badan pegal"],
    "lemas": ["lemas", "lemas sekali", "lesu", "lelah", "capek"],
    "bengkak": ["bengkak", "bengkak wajah", "bibir bengkak", "lidah bengkak"],
    "perdarahan": ["darah", "perdarahan", "mimisan", "gusi berdarah", "bab hitam", "tinja hitam"],
}


@dataclass(frozen=True)
class AnamnesisResult:
    normalized_message: str
    red_flags: list[str]
    hinted_case_ids: list[str]
    detected_symptoms: list[str]
    present_symptoms: list[str]
    absent_symptoms: list[str]
    duration_text: str | None
    answered_slots: list[str]
    has_intensity_signal: bool
    has_progression_signal: bool
    has_duration_signal: bool
    has_safety_clearance_signal: bool
    asks_for_recommendation: bool


def _is_negated(normalized: str, phrase: str) -> bool:
    return any(
        contains_phrase(normalized, f"{prefix} {phrase}")
        for prefix in ["tidak ada", "tidak", "tanpa", "nggak", "gak", "ga", "bukan"]
    )


def _extract_duration_text(normalized: str) -> str | None:
    patterns = [
        r"\b(?:sudah|selama|sejak|baru|sekitar|kurang lebih)\s+(?:[a-z]+\s+){0,3}?\d+\s*(?:jam|hari|minggu|bulan)\b",
        r"\b(?:sudah|selama|sejak|baru)\s+(?:tadi\s+(?:pagi|siang|sore|malam)|kemarin|semalam|tadi)\b",
        r"\b(?:sejak|dari)\s+(?:kemarin|semalam|tadi|tadi\s+(?:pagi|siang|sore|malam))\b",
        r"\b\d+\s*(?:jam|hari|minggu|bulan)\b",
        r"\bhari\s+ke\s+\d+\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            return match.group(0).strip()
    return None


def _extract_symptom_slots(normalized: str) -> tuple[list[str], list[str]]:
    present: list[str] = []
    absent: list[str] = []
    for symptom, aliases in SYMPTOM_CONCEPTS.items():
        matched_aliases = [alias for alias in aliases if contains_phrase(normalized, alias)]
        if not matched_aliases:
            continue
        if any(_is_negated(normalized, alias) for alias in matched_aliases):
            absent.append(symptom)
        else:
            present.append(symptom)
    return sorted(set(present)), sorted(set(absent))


def _has_intensity_signal(normalized: str) -> bool:
    return any(
        contains_phrase(normalized, phrase)
        for phrase in [
            "ringan",
            "sedang",
            "berat",
            "parah",
            "sangat sakit",
            "lebih sakit",
            "nyeri saat menelan",
            "sakit saat menelan",
            "kalau menelan sakit",
            "terasa sakit saat menelan",
            "nyeri menelan",
            "suhu",
            "derajat",
        ]
    )


def _has_progression_signal(normalized: str) -> bool:
    return any(
        contains_phrase(normalized, phrase)
        for phrase in [
            "memburuk",
            "makin",
            "semakin",
            "bertambah",
            "lebih sakit",
            "setiap hari",
            "menetap",
            "naik turun",
            "ketika bangun",
            "bangun tidur",
        ]
    )


def _build_answered_slots(
    *,
    present_symptoms: list[str],
    absent_symptoms: list[str],
    duration_text: str | None,
    red_flags: list[str],
    has_safety_clearance_signal: bool,
) -> list[str]:
    slots: list[str] = []
    slots.extend(f"gejala disebut ada: {symptom}" for symptom in present_symptoms)
    slots.extend(f"gejala disebut tidak ada: {symptom}" for symptom in absent_symptoms)
    if duration_text:
        slots.append(f"durasi sudah disebut: {duration_text}")
    slots.extend(f"tanda bahaya terdeteksi: {flag}" for flag in red_flags)
    if has_safety_clearance_signal:
        slots.append("sebagian tanda bahaya sudah disangkal user")
    return slots


def analyze_message(message: str) -> AnamnesisResult:
    normalized = normalize(message)
    red_flags = []
    for phrase, label in RED_FLAG_PHRASES.items():
        if contains_phrase(normalized, phrase) and not _is_negated(normalized, phrase) and label not in red_flags:
            red_flags.append(label)

    hinted_case_ids = []
    detected_symptoms = []
    for case_id, hints in CASE_HINTS.items():
        matched_hints = [
            hint
            for hint in hints
            if contains_phrase(normalized, hint) and not _is_negated(normalized, hint)
        ]
        if matched_hints:
            hinted_case_ids.append(case_id)
            detected_symptoms.extend(matched_hints)

    present_symptoms, absent_symptoms = _extract_symptom_slots(normalized)
    duration_text = _extract_duration_text(normalized)
    has_intensity_signal = _has_intensity_signal(normalized)
    has_progression_signal = _has_progression_signal(normalized)
    has_duration_signal = bool(duration_text) or any(
        contains_phrase(normalized, phrase)
        for phrase in [
            "hari",
            "jam",
            "minggu",
            "sejak",
            "tadi",
            "kemarin",
            "baru",
            "sudah",
            "lama",
        ]
    )
    asks_for_recommendation = any(
        contains_phrase(normalized, phrase)
        for phrase in ["rekomendasi", "ramuan", "herbal", "obat tradisional", "lanjut", "boleh"]
    )
    has_safety_clearance_signal = any(
        contains_phrase(normalized, f"{prefix} {phrase}")
        for prefix in ["tidak ada", "tanpa", "tidak", "nggak", "gak", "ga"]
        for phrase in RED_FLAG_PHRASES
    )
    answered_slots = _build_answered_slots(
        present_symptoms=present_symptoms,
        absent_symptoms=absent_symptoms,
        duration_text=duration_text,
        red_flags=red_flags,
        has_safety_clearance_signal=has_safety_clearance_signal,
    )

    return AnamnesisResult(
        normalized_message=normalized,
        red_flags=red_flags,
        hinted_case_ids=hinted_case_ids,
        detected_symptoms=sorted(set(detected_symptoms)),
        present_symptoms=present_symptoms,
        absent_symptoms=absent_symptoms,
        duration_text=duration_text,
        answered_slots=answered_slots,
        has_intensity_signal=has_intensity_signal,
        has_progression_signal=has_progression_signal,
        has_duration_signal=has_duration_signal,
        has_safety_clearance_signal=has_safety_clearance_signal,
        asks_for_recommendation=asks_for_recommendation,
    )
