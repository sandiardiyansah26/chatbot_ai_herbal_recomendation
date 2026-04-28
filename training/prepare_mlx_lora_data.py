from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "data" / "traning" / "combined_training_sft.jsonl"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data_mlx" / "herbal_chat"


def main() -> int:
    args = parse_args()
    rows = [row for row in load_jsonl(args.input) if is_valid_chat_row(row)]
    if args.limit:
        rows = rows[: args.limit]

    if len(rows) < 3:
        raise SystemExit(f"Dataset terlalu kecil untuk split train/valid/test: {len(rows)} row.")

    if args.split_strategy == "group":
        train_rows, valid_rows, test_rows = split_rows_by_group(
            rows,
            valid_ratio=args.valid_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
    else:
        random.Random(args.seed).shuffle(rows)
        train_rows, valid_rows, test_rows = split_rows(
            rows,
            valid_ratio=args.valid_ratio,
            test_ratio=args.test_ratio,
        )

    split_audit = build_split_audit(train_rows, valid_rows, test_rows)
    train_rows = [normalize_row(row) for row in train_rows]
    valid_rows = [normalize_row(row) for row in valid_rows]
    test_rows = [normalize_row(row) for row in test_rows]

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
        "split_strategy": args.split_strategy,
        **split_audit,
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
    parser.add_argument(
        "--split-strategy",
        choices=("group", "random"),
        default="group",
        help=(
            "group menjaga semua variasi dari source_record/topic yang sama tetap berada di split yang sama, "
            "sehingga valid/test tidak bocor dari train. random mempertahankan perilaku lama."
        ),
    )
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


def split_rows_by_group(
    rows: list[dict[str, Any]],
    *,
    valid_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    total = len(rows)
    valid_target = max(1, int(total * valid_ratio))
    test_target = max(1, int(total * test_ratio))

    groups = build_leakage_safe_groups(rows)
    random.Random(seed).shuffle(groups)

    valid_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []

    for group_rows in groups:
        if len(valid_rows) < valid_target:
            valid_rows.extend(group_rows)
        elif len(test_rows) < test_target:
            test_rows.extend(group_rows)
        else:
            train_rows.extend(group_rows)

    if not train_rows or not valid_rows or not test_rows:
        raise ValueError("Split group menghasilkan split kosong; gunakan --split-strategy random atau ubah rasio.")
    return train_rows, valid_rows, test_rows


def build_leakage_safe_groups(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    key_to_index: dict[str, int] = {}
    for index, row in enumerate(rows):
        for key in row_leakage_keys(row, index):
            if key in key_to_index:
                union(key_to_index[key], index)
            else:
                key_to_index[key] = index

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[find(index)].append(row)
    return list(grouped.values())


def row_leakage_keys(row: dict[str, Any], index: int) -> list[str]:
    keys = [row_group_key(row, index)]
    assistant = first_message_content(row, "assistant")
    user = first_message_content(row, "user")
    if assistant:
        keys.append(f"assistant:{stable_hash(assistant)}")
    if user:
        keys.append(f"user:{stable_hash(user)}")
    return keys


def row_group_key(row: dict[str, Any], index: int) -> str:
    source_record_id = str(row.get("source_record_id") or "").strip()
    if source_record_id:
        return f"source_record:{source_record_id}"

    row_id = str(row.get("id") or "").strip()
    medlineplus_match = re.match(r"medlineplus_(\d+)_", row_id)
    if medlineplus_match:
        return f"medlineplus_topic:{medlineplus_match.group(1)}"

    anamnesis_match = re.match(r"sft_anam_(\d+)", row_id)
    if anamnesis_match:
        return f"anamnesis_case:{anamnesis_match.group(1)}"

    if row_id.startswith("rag_"):
        return f"rag_user:{stable_hash(first_message_content(row, 'user'))}"

    return f"row_id:{row_id or index}"


def first_message_content(row: dict[str, Any], role: str) -> str:
    for message in row.get("messages", []):
        if isinstance(message, dict) and message.get("role") == role:
            return str(message.get("content") or "")
    return ""


def stable_hash(value: str) -> str:
    normalized = " ".join(value.split()).strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def build_split_audit(
    train_rows: list[dict[str, Any]],
    valid_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    split_groups = {
        "train": {row_group_key(row, index) for index, row in enumerate(train_rows)},
        "valid": {row_group_key(row, index) for index, row in enumerate(valid_rows)},
        "test": {row_group_key(row, index) for index, row in enumerate(test_rows)},
    }
    leaked_groups = (
        (split_groups["train"] & split_groups["valid"])
        | (split_groups["train"] & split_groups["test"])
        | (split_groups["valid"] & split_groups["test"])
    )
    assistant_leakage = cross_split_content_leakage(train_rows, valid_rows, test_rows, "assistant")
    user_leakage = cross_split_content_leakage(train_rows, valid_rows, test_rows, "user")
    return {
        "total_groups": len(set().union(*split_groups.values())),
        "train_groups": len(split_groups["train"]),
        "valid_groups": len(split_groups["valid"]),
        "test_groups": len(split_groups["test"]),
        "group_leakage_count": len(leaked_groups),
        "assistant_cross_split_duplicate_groups": assistant_leakage,
        "user_cross_split_duplicate_groups": user_leakage,
    }


def cross_split_content_leakage(
    train_rows: list[dict[str, Any]],
    valid_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    role: str,
) -> int:
    locations: dict[str, set[str]] = defaultdict(set)
    for split_name, rows in (("train", train_rows), ("valid", valid_rows), ("test", test_rows)):
        for row in rows:
            content = first_message_content(row, role)
            if content:
                locations[stable_hash(content)].add(split_name)
    return sum(1 for split_names in locations.values() if len(split_names) > 1)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
