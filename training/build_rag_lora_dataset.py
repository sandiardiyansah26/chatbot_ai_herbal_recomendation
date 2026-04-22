from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_BASE_DATASET = Path(__file__).resolve().parents[1] / "data" / "traning" / "combined_training_sft.jsonl"
DEFAULT_LEARNING_LOG = Path(__file__).resolve().parents[1] / "data" / "learning" / "dual_llm_interactions.jsonl"
DEFAULT_RAG_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "traning" / "rag_learning_sft.jsonl"
DEFAULT_MERGED_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "traning" / "combined_training_sft_rag.jsonl"
DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "data" / "traning" / "rag_learning_manifest.json"
DEFAULT_ALLOWED_TYPES = ("recommendation", "follow_up", "red_flag", "out_of_scope")
DEFAULT_SYSTEM_PROMPT = (
    "Anda adalah chatbot edukasi ramuan herbal berbahasa Indonesia. "
    "Lakukan anamnesis ringan, hanya rekomendasikan ramuan untuk keluhan ringan, "
    "jelaskan cara pengolahan dan dosis, dan selalu tekankan bahwa ini bukan diagnosis "
    "medis final serta bukan pengganti konsultasi tenaga kesehatan. Gunakan hanya "
    "jawaban yang grounded pada konteks RAG yang aman."
)
DISCLAIMER_SNIPPET = "bukan diagnosis medis final"


def main() -> int:
    args = parse_args()
    base_rows = load_jsonl(args.base_input)
    rag_rows, manifest = build_rag_rows(
        learning_log=load_jsonl(args.learning_log),
        allowed_response_types=args.allowed_response_types,
        system_prompt=args.system_prompt,
    )

    args.rag_output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.rag_output, rag_rows)
    write_jsonl(args.merged_output, [*base_rows, *rag_rows])
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert RAG learning logs into extra SFT rows for lightweight LoRA retraining.")
    parser.add_argument("--base-input", type=Path, default=DEFAULT_BASE_DATASET)
    parser.add_argument("--learning-log", type=Path, default=DEFAULT_LEARNING_LOG)
    parser.add_argument("--rag-output", type=Path, default=DEFAULT_RAG_OUTPUT)
    parser.add_argument("--merged-output", type=Path, default=DEFAULT_MERGED_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--allowed-response-types",
        nargs="+",
        default=list(DEFAULT_ALLOWED_TYPES),
        help="Response types from the learning log that should become SFT rows.",
    )
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_rag_rows(
    *,
    learning_log: list[dict[str, Any]],
    allowed_response_types: list[str],
    system_prompt: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    response_type_counter: Counter[str] = Counter()
    seen_pairs: set[tuple[str, str]] = set()

    for item in learning_log:
        response_type = str(item.get("response_type") or "").strip()
        if response_type not in allowed_response_types:
            continue

        user_message = clean_message(str(item.get("user_message") or ""))
        assistant_message = choose_assistant_reply(item)
        if not user_message or not assistant_message:
            continue

        signature = (normalize_for_dedupe(user_message), normalize_for_dedupe(assistant_message))
        if signature in seen_pairs:
            continue
        seen_pairs.add(signature)

        rows.append(
            {
                "id": f"rag_{item.get('id') or len(rows) + 1}",
                "source": "learning_log_rag_grounded",
                "response_type": response_type,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": assistant_message},
                ],
            }
        )
        response_type_counter[response_type] += 1

    manifest = {
        "learning_log_rows": len(learning_log),
        "rag_sft_rows": len(rows),
        "response_type_breakdown": dict(response_type_counter),
        "allowed_response_types": allowed_response_types,
        "selection_policy": "Prefer grounded baseline_reply; fallback to selected_reply when baseline is empty.",
        "system_prompt": system_prompt,
    }
    return rows, manifest


def choose_assistant_reply(item: dict[str, Any]) -> str:
    baseline = clean_reply(str(item.get("baseline_reply") or ""))
    if baseline:
        return ensure_disclaimer(baseline)

    selected = clean_reply(str(item.get("selected_reply") or ""))
    if selected:
        return ensure_disclaimer(selected)

    return ""


def clean_message(text: str) -> str:
    return " ".join(text.split()).strip()


def clean_reply(text: str) -> str:
    reply = text.replace("\r\n", "\n").strip()
    if not reply:
        return ""

    leaked_prefixes = (
        "Jawab dalam Bahasa Indonesia yang natural dan interaktif:",
        "Instruksi jawaban:",
    )
    for prefix in leaked_prefixes:
        if reply.startswith(prefix):
            reply = reply[len(prefix) :].strip()

    lines = [line.rstrip() for line in reply.splitlines()]
    compact_lines: list[str] = []
    previous_blank = False
    for line in lines:
        if not line.strip():
            if previous_blank:
                continue
            previous_blank = True
            compact_lines.append("")
            continue
        previous_blank = False
        compact_lines.append(line)

    return "\n".join(compact_lines).strip()


def ensure_disclaimer(reply: str) -> str:
    normalized = normalize_for_dedupe(reply)
    if DISCLAIMER_SNIPPET in normalized:
        return reply
    return (
        f"{reply.rstrip()}\n\n"
        "Informasi ini bersifat rekomendasi awal dan edukasi, bukan diagnosis medis final "
        "dan bukan pengganti konsultasi tenaga kesehatan."
    )


def normalize_for_dedupe(text: str) -> str:
    return " ".join(text.lower().split())


if __name__ == "__main__":
    raise SystemExit(main())
