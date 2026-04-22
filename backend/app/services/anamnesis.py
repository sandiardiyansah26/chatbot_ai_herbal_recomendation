from __future__ import annotations

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
    "kejang": "kejang",
    "pingsan": "pingsan",
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
    "hamil": "kehamilan perlu kehati-hatian khusus",
    "bayi": "keluhan pada bayi perlu evaluasi tenaga kesehatan",
}


CASE_HINTS = {
    "case_001": ["mual", "muntah", "enek", "eneg", "lambung"],
    "case_002": ["tenggorokan", "gatal tenggorokan", "sakit tenggorokan"],
    "case_003": ["nafsu makan", "malas makan", "tidak nafsu", "selera makan"],
    "case_004": ["pegal", "kurang fit", "capek", "lelah", "badan pegal"],
    "case_005": ["diare", "mencret", "bab cair", "perut mulas"],
    "case_006": ["panas badan", "demam", "demam ringan", "tidak nyaman tubuh", "nyeri ringan", "meriang", "meriang ringan", "pusing", "pusing ringan", "sakit kepala", "sakit kepala ringan"],
    "case_007": ["batuk ringan", "suara serak", "serak", "polusi"],
    "case_008": ["batuk pilek", "pilek ringan", "hidung meler", "batuk"],
    "case_009": ["gatal", "gatal-gatal", "gatal gatal", "ruam", "ruam merah", "bentol", "biduran", "alergi kulit"],
}


@dataclass(frozen=True)
class AnamnesisResult:
    normalized_message: str
    red_flags: list[str]
    hinted_case_ids: list[str]
    detected_symptoms: list[str]
    has_duration_signal: bool
    has_safety_clearance_signal: bool
    asks_for_recommendation: bool


def _is_negated(normalized: str, phrase: str) -> bool:
    return any(
        contains_phrase(normalized, f"{prefix} {phrase}")
        for prefix in ["tidak ada", "tidak", "tanpa", "nggak", "gak", "ga", "bukan"]
    )


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

    has_duration_signal = any(
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

    return AnamnesisResult(
        normalized_message=normalized,
        red_flags=red_flags,
        hinted_case_ids=hinted_case_ids,
        detected_symptoms=sorted(set(detected_symptoms)),
        has_duration_signal=has_duration_signal,
        has_safety_clearance_signal=has_safety_clearance_signal,
        asks_for_recommendation=asks_for_recommendation,
    )
