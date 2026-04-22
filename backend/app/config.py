from __future__ import annotations

import os
from pathlib import Path


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


def _default_chroma_db_dir() -> Path:
    current = Path(__file__).resolve()
    candidates = [
        current.parents[1] / ".chroma",
        current.parents[2] / ".chroma" if len(current.parents) > 2 else None,
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    return next(candidate for candidate in candidates if candidate is not None)


def _split_env_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return max(parsed, 1)


def _env_path(name: str, default: Path) -> Path:
    return Path(os.getenv(name) or default).resolve()


DATA_DIR = _env_path("HERBAL_DATA_DIR", _default_data_dir())
CHROMA_DB_DIR = _env_path("CHROMA_DB_DIR", _default_chroma_db_dir())
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL_A = os.getenv("OLLAMA_MODEL_A", "deepseek-r1:1.5b")
OLLAMA_MODEL_B = os.getenv("OLLAMA_MODEL_B", "gemma4:e2b")
OLLAMA_MODEL_B_FALLBACKS = _split_env_list(os.getenv("OLLAMA_MODEL_B_FALLBACKS", "gemma4:latest,gemma3:1b"))
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "45"))
OLLAMA_NUM_PREDICT_DEFAULT = _env_int("OLLAMA_NUM_PREDICT_DEFAULT", 220)
OLLAMA_NUM_PREDICT_RECOMMENDATION = _env_int("OLLAMA_NUM_PREDICT_RECOMMENDATION", 768)
OLLAMA_NUM_PREDICT_FOLLOW_UP = _env_int("OLLAMA_NUM_PREDICT_FOLLOW_UP", 140)
OLLAMA_NUM_PREDICT_RED_FLAG = _env_int("OLLAMA_NUM_PREDICT_RED_FLAG", 120)
OLLAMA_NUM_PREDICT_OUT_OF_SCOPE = _env_int("OLLAMA_NUM_PREDICT_OUT_OF_SCOPE", 120)
ENABLE_DUAL_LLM_COMPARISON = os.getenv("ENABLE_DUAL_LLM_COMPARISON", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
LEARNING_LOG_PATH = Path(
    os.getenv("LEARNING_LOG_PATH") or (DATA_DIR.parent / "learning" / "dual_llm_interactions.jsonl")
).resolve()
APP_NAME = "AI Chatbot Rekomendasi Ramuan Herbal"
APP_VERSION = "0.1.1"
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://0.0.0.0:5173,http://localhost:8080,http://127.0.0.1:8080,null",
    ).split(",")
    if origin.strip()
]
