from __future__ import annotations

import re
import unicodedata


STOPWORDS = {
    "aku",
    "ada",
    "adalah",
    "agar",
    "akan",
    "atau",
    "badan",
    "bagaimana",
    "bagi",
    "bisa",
    "dan",
    "dengan",
    "di",
    "diderita",
    "ini",
    "itu",
    "jadi",
    "juga",
    "kalau",
    "karena",
    "ke",
    "keluhan",
    "kena",
    "lagi",
    "mau",
    "mengalami",
    "mohon",
    "nya",
    "obat",
    "pada",
    "rekomendasi",
    "saya",
    "sedang",
    "sejak",
    "sudah",
    "tapi",
    "terasa",
    "tidak",
    "tolong",
    "untuk",
    "yang",
}

SYNONYMS = {
    "kerongkongan": "tenggorokan",
    "radang": "nyeri",
    "sakit": "nyeri",
    "gatal": "iritasi",
    "mulas": "diare",
    "mencret": "diare",
    "bab": "diare",
    "cair": "diare",
    "enek": "mual",
    "eneg": "mual",
    "masuk": "kurang",
    "angin": "fit",
    "capek": "lelah",
    "lesu": "lelah",
    "malas": "nafsu",
    "selera": "nafsu",
    "makan": "makan",
    "panas": "demam",
    "meriang": "demam",
}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    normalized = normalize(text)
    tokens = []
    for token in normalized.split():
        if len(token) <= 2 or token in STOPWORDS:
            continue
        tokens.append(SYNONYMS.get(token, token))
    return tokens


def contains_phrase(text: str, phrase: str) -> bool:
    return normalize(phrase) in normalize(text)
