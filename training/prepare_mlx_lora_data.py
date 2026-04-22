from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path(__file__).resolve().parents[2] / "data" / "traning" / "combined_training_sft.jsonl"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data_mlx" / "herbal_chat"


def main() -> int:
    args = parse_args()
    rows = load_jsonl(args.input)
    rows = [normalize_row(row) for row in rows if is_valid_chat_row(row)]
    if args.limit:
        rows = rows[: args.limit]

    if len(rows) < 3:
        raise SystemExit(f"Dataset terlalu kecil untuk split train/valid/test: {len(rows)} row.")

    random.Random(args.seed).shuffle(rows)
    train_rows, valid_rows, test_rows = split_rows(
        rows,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "train.jsonl", train_rows)
    write_jsonl(args.output / "valid.jsonl", valid_rows)
    write_jsonl(args.output / "test.jsonl", test_rows)

    manifest = {
        "source": str(args.input),
        "output": str(args.output),
        "total_rows": len(rows),
        "train_rows": len(train_rows),
        "valid_rows": len(valid_rows),
        "test_rows": len(test_rows),
        "format": "OpenAI chat JSONL: {'messages': [{'role': ..., 'content': ...}]}",
        "seed": args.seed,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MLX-LM LoRA chat dataset.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--valid-ratio", type=float, default=0.08)
    parser.add_argument("--test-ratio", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def is_valid_chat_row(row: dict[str, Any]) -> bool:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return False
    roles = {message.get("role") for message in messages if isinstance(message, dict)}
    return "user" in roles and "assistant" in roles


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    messages = []
    for message in row["messages"]:
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", "")).strip()
        if role in {"system", "user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    return {"messages": messages}


def split_rows(
    rows: list[dict[str, Any]],
    *,
    valid_ratio: float,
    test_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    total = len(rows)
    valid_count = max(1, int(total * valid_ratio))
    test_count = max(1, int(total * test_ratio))
    train_count = total - valid_count - test_count
    if train_count <= 0:
        raise ValueError("Split ratio membuat train set kosong.")
    return rows[:train_count], rows[train_count : train_count + valid_count], rows[train_count + valid_count :]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
