#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Single runtime configuration for the Herbal Chatbot system.
#
# Edit values in this block, then run:
#   ./app.config.sh start
#   ./app.config.sh restart
#   ./app.config.sh logs
#
# To change models, edit OLLAMA_MODEL_A / OLLAMA_MODEL_B below,
# then run: ./app.config.sh restart
# ============================================================

# App identity
APP_NAME="${APP_NAME:-AI Chatbot Rekomendasi Ramuan Herbal}"
APP_VERSION="${APP_VERSION:-0.1.1}"

# Ports
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"

# Backend data paths inside Docker container
HERBAL_DATA_DIR="${HERBAL_DATA_DIR:-/app/data/referensi}"
CHROMA_DB_DIR="${CHROMA_DB_DIR:-/app/.chroma}"

# Frontend runtime config
FRONTEND_PUBLIC_API_BASE="${FRONTEND_PUBLIC_API_BASE:-http://127.0.0.1:${BACKEND_PORT}}"

# Ollama model config
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://host.docker.internal:11434}"
OLLAMA_MODEL_A="${OLLAMA_MODEL_A:-deepseek-r1:7b}"
OLLAMA_MODEL_A_FALLBACKS="${OLLAMA_MODEL_A_FALLBACKS:-deepseek-r1:1.5b}"
OLLAMA_MODEL_B="${OLLAMA_MODEL_B:-gemma4:latest}"
OLLAMA_MODEL_B_FALLBACKS="${OLLAMA_MODEL_B_FALLBACKS:-gemma4:e2b,gemma3:1b}"
ENABLE_DUAL_LLM_COMPARISON="${ENABLE_DUAL_LLM_COMPARISON:-true}"

# Inference limits
OLLAMA_TIMEOUT_SECONDS="${OLLAMA_TIMEOUT_SECONDS:-75}"
OLLAMA_NUM_PREDICT_DEFAULT="${OLLAMA_NUM_PREDICT_DEFAULT:-220}"
OLLAMA_NUM_PREDICT_RECOMMENDATION="${OLLAMA_NUM_PREDICT_RECOMMENDATION:-640}"
OLLAMA_NUM_PREDICT_FOLLOW_UP="${OLLAMA_NUM_PREDICT_FOLLOW_UP:-240}"
OLLAMA_NUM_PREDICT_RED_FLAG="${OLLAMA_NUM_PREDICT_RED_FLAG:-120}"
OLLAMA_NUM_PREDICT_OUT_OF_SCOPE="${OLLAMA_NUM_PREDICT_OUT_OF_SCOPE:-120}"
MAX_ANAMNESIS_QUESTIONS="${MAX_ANAMNESIS_QUESTIONS:-3}"

# OpenAI comparison is optional. Leave OPENAI_API_KEY empty to use local Ollama only.
ENABLE_OPENAI_COMPARISON="${ENABLE_OPENAI_COMPARISON:-true}"
OPENAI_API_KEY="${OPENAI_API_KEY:-}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
OPENAI_MODEL="${OPENAI_MODEL:-gpt-4o}"
OPENAI_TIMEOUT_SECONDS="${OPENAI_TIMEOUT_SECONDS:-75}"

# Learning logs inside Docker container
LEARNING_LOG_PATH="${LEARNING_LOG_PATH:-/app/data/learning/dual_llm_interactions.jsonl}"
CONVERSATION_LOG_PATH="${CONVERSATION_LOG_PATH:-/app/data/learning/conversation_turns.jsonl}"
KB_ENRICHMENT_LOG_PATH="${KB_ENRICHMENT_LOG_PATH:-/app/data/learning/kb_enrichment_candidates.jsonl}"
RECOMMENDATION_FEEDBACK_LOG_PATH="${RECOMMENDATION_FEEDBACK_LOG_PATH:-/app/data/learning/recommendation_feedback.jsonl}"

# CORS for browser access
CORS_ORIGINS="${CORS_ORIGINS:-http://localhost:${FRONTEND_PORT},http://127.0.0.1:${FRONTEND_PORT},http://0.0.0.0:${FRONTEND_PORT},null}"

export APP_NAME APP_VERSION
export BACKEND_PORT FRONTEND_PORT
export FRONTEND_PUBLIC_API_BASE
export HERBAL_DATA_DIR CHROMA_DB_DIR
export OLLAMA_BASE_URL OLLAMA_MODEL_A OLLAMA_MODEL_A_FALLBACKS OLLAMA_MODEL_B OLLAMA_MODEL_B_FALLBACKS
export ENABLE_DUAL_LLM_COMPARISON
export OLLAMA_TIMEOUT_SECONDS OLLAMA_NUM_PREDICT_DEFAULT OLLAMA_NUM_PREDICT_RECOMMENDATION
export OLLAMA_NUM_PREDICT_FOLLOW_UP OLLAMA_NUM_PREDICT_RED_FLAG OLLAMA_NUM_PREDICT_OUT_OF_SCOPE
export MAX_ANAMNESIS_QUESTIONS
export ENABLE_OPENAI_COMPARISON OPENAI_API_KEY OPENAI_BASE_URL OPENAI_MODEL OPENAI_TIMEOUT_SECONDS
export LEARNING_LOG_PATH CONVERSATION_LOG_PATH KB_ENRICHMENT_LOG_PATH RECOMMENDATION_FEEDBACK_LOG_PATH
export CORS_ORIGINS

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

generate_frontend_runtime_config() {
  python3 - <<'PY'
import json
import os
from pathlib import Path

config = {
    "apiBase": os.environ["FRONTEND_PUBLIC_API_BASE"],
    "appName": os.environ["APP_NAME"],
    "appVersion": os.environ["APP_VERSION"],
    "backendPort": os.environ["BACKEND_PORT"],
    "frontendPort": os.environ["FRONTEND_PORT"],
}
Path("frontend/runtime-config.js").write_text(
    "window.HERBAL_APP_CONFIG = "
    + json.dumps(config, ensure_ascii=False, indent=2)
    + ";\n",
    encoding="utf-8",
)
PY
}

print_summary() {
  cat <<EOF
Config loaded:
  Frontend: http://127.0.0.1:${FRONTEND_PORT}
  Backend : http://127.0.0.1:${BACKEND_PORT}
  API base: ${FRONTEND_PUBLIC_API_BASE}
  Model A : ${OLLAMA_MODEL_A}
  Model B : ${OLLAMA_MODEL_B}
  OpenAI  : ${OPENAI_MODEL} (enabled=${ENABLE_OPENAI_COMPARISON}, key_set=$([ -n "${OPENAI_API_KEY}" ] && echo yes || echo no))
EOF
}

pull_models() {
  if ! command -v ollama >/dev/null 2>&1; then
    echo "ollama command not found. Install/start Ollama first."
    exit 1
  fi

  IFS=',' read -r -a model_a_fallbacks <<< "${OLLAMA_MODEL_A_FALLBACKS}"
  IFS=',' read -r -a model_b_fallbacks <<< "${OLLAMA_MODEL_B_FALLBACKS}"

  ollama pull "${OLLAMA_MODEL_A}"
  for model in "${model_a_fallbacks[@]}"; do
    model="$(echo "${model}" | xargs)"
    [ -n "${model}" ] && ollama pull "${model}"
  done

  ollama pull "${OLLAMA_MODEL_B}"
  for model in "${model_b_fallbacks[@]}"; do
    model="$(echo "${model}" | xargs)"
    [ -n "${model}" ] && ollama pull "${model}"
  done
}

command="${1:-start}"
case "${command}" in
  start|up)
    generate_frontend_runtime_config
    print_summary
    docker compose up --build -d
    ;;
  restart|reload)
    generate_frontend_runtime_config
    print_summary
    docker compose up --build -d --force-recreate
    ;;
  stop)
    docker compose stop
    ;;
  down)
    docker compose down
    ;;
  logs)
    shift || true
    docker compose logs -f "$@"
    ;;
  ps|status)
    docker compose ps
    ;;
  config)
    generate_frontend_runtime_config
    docker compose config
    ;;
  pull-models)
    pull_models
    ;;
  *)
    cat <<EOF
Usage: ./app.config.sh [start|restart|stop|down|logs|ps|config|pull-models]

Examples:
  ./app.config.sh start
  ./app.config.sh restart
  ./app.config.sh logs backend
  ./app.config.sh pull-models
EOF
    exit 1
    ;;
esac
