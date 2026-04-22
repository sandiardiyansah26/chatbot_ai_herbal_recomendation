from __future__ import annotations

import argparse
import importlib.util
import json
import platform
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = Path(__file__).with_name("qlora_config.example.json")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    config.update({key: value for key, value in vars(args).items() if value is not None and key != "config"})
    run_training(config)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for AI Chatbot Ramuan Herbal.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--base_model")
    parser.add_argument("--dataset_path")
    parser.add_argument("--output_dir")
    parser.add_argument("--max_seq_length", type=int)
    parser.add_argument("--num_train_epochs", type=float)
    parser.add_argument("--per_device_train_batch_size", type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--check_environment", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def run_training(config: dict[str, Any]) -> None:
    if config.get("check_environment"):
        print(json.dumps(inspect_environment(config), ensure_ascii=False, indent=2))
        return

    if config.get("dry_run"):
        print(json.dumps(build_training_plan(config), ensure_ascii=False, indent=2))
        return

    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ImportError as error:  # pragma: no cover - depends on optional GPU stack.
        raise SystemExit(
            "Dependensi QLoRA belum terpasang. Jalankan:\n"
            "  cd program/training\n"
            "  python3 -m venv .venv\n"
            "  source .venv/bin/activate\n"
            "  pip install -r requirements-qlora.txt\n\n"
            f"Detail import error: {error}"
        ) from error

    if not torch.cuda.is_available():
        raise SystemExit(
            "QLoRA 4-bit pada script ini membutuhkan CUDA/NVIDIA GPU karena memakai bitsandbytes. "
            "Environment saat ini tidak memiliki CUDA. Untuk Mac Apple Silicon, gunakan mesin training "
            "cloud/NVIDIA GPU, atau buat pipeline LoRA non-4bit terpisah berbasis MPS/MLX."
        )

    dataset_path = resolve_path(config["dataset_path"])
    output_dir = resolve_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    if bool(config.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(config["lora_r"]),
            lora_alpha=int(config["lora_alpha"]),
            lora_dropout=float(config["lora_dropout"]),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(config["target_modules"]),
        ),
    )

    raw_dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    tokenized_dataset = raw_dataset.map(
        lambda example: tokenize_example(example, tokenizer, int(config["max_seq_length"])),
        remove_columns=raw_dataset.column_names,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(config["num_train_epochs"]),
        per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        learning_rate=float(config["learning_rate"]),
        logging_steps=5,
        save_strategy="epoch",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        optim="paged_adamw_8bit",
        gradient_checkpointing=bool(config.get("gradient_checkpointing", True)),
        warmup_ratio=float(config.get("warmup_ratio", 0.03)),
        max_grad_norm=float(config.get("max_grad_norm", 0.3)),
        save_total_limit=int(config.get("save_total_limit", 2)),
        group_by_length=True,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"QLoRA adapter saved to {output_dir}")


def inspect_environment(config: dict[str, Any]) -> dict[str, Any]:
    modules = {
        name: importlib.util.find_spec(name) is not None
        for name in ["torch", "transformers", "datasets", "peft", "bitsandbytes", "accelerate"]
    }
    torch_info: dict[str, Any] = {"available": modules["torch"]}
    if modules["torch"]:
        try:
            import torch

            torch_info.update(
                {
                    "version": torch.__version__,
                    "cuda_available": torch.cuda.is_available(),
                    "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                    "bf16_supported": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
                    "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
                }
            )
        except Exception as error:  # pragma: no cover - environment dependent.
            torch_info["error"] = str(error)

    plan = build_training_plan(config)
    return {
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "modules": modules,
        "torch": torch_info,
        "qlora_cuda_ready": bool(torch_info.get("cuda_available")) and modules["bitsandbytes"],
        "plan": plan,
        "recommendation": (
            "Jalankan QLoRA di cloud/NVIDIA GPU bila qlora_cuda_ready=false. "
            "Di Mac Apple Silicon, gunakan script ini untuk validasi dataset/config, bukan training 4-bit."
        ),
    }


def build_training_plan(config: dict[str, Any]) -> dict[str, Any]:
    dataset_path = resolve_path(config["dataset_path"])
    return {
        "base_model": config["base_model"],
        "dataset_path": str(dataset_path),
        "dataset_rows": count_jsonl_rows(dataset_path),
        "output_dir": str(resolve_path(config["output_dir"])),
        "max_seq_length": int(config["max_seq_length"]),
        "num_train_epochs": float(config["num_train_epochs"]),
        "effective_batch_size": int(config["per_device_train_batch_size"])
        * int(config["gradient_accumulation_steps"]),
        "learning_rate": float(config["learning_rate"]),
        "lora": {
            "r": int(config["lora_r"]),
            "alpha": int(config["lora_alpha"]),
            "dropout": float(config["lora_dropout"]),
            "target_modules": list(config["target_modules"]),
        },
        "memory_optimizations": {
            "load_in_4bit": True,
            "quant_type": "nf4",
            "double_quant": True,
            "gradient_checkpointing": bool(config.get("gradient_checkpointing", True)),
            "optim": "paged_adamw_8bit",
            "group_by_length": True,
        },
    }


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


def tokenize_example(example: dict[str, Any], tokenizer: Any, max_seq_length: int) -> dict[str, Any]:
    text = format_messages(example["messages"], tokenizer)
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_seq_length,
        padding=False,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def format_messages(messages: list[dict[str, str]], tokenizer: Any) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    lines = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        lines.append(f"<{role}>\n{content}\n</{role}>")
    return "\n".join(lines)


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (Path(__file__).resolve().parent / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
