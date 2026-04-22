# MLX LoRA Fine-Tuning untuk MacBook Apple Silicon

Pipeline ini dipakai untuk fine-tuning lokal di MacBook M-series. Berbeda dari QLoRA `bitsandbytes` yang membutuhkan CUDA/NVIDIA, MLX memakai Apple Metal GPU.

## Environment

```bash
cd program/training
python3 -m venv .venv-mlx
.venv-mlx/bin/python -m pip install --upgrade pip setuptools wheel
.venv-mlx/bin/python -m pip install -r requirements-mlx.txt
```

## Siapkan Dataset

Dataset sumber memakai format chat SFT di `data/traning/combined_training_sft.jsonl`.

```bash
python3 program/training/prepare_mlx_lora_data.py
```

Output:

- `program/training/data_mlx/herbal_chat/train.jsonl`
- `program/training/data_mlx/herbal_chat/valid.jsonl`
- `program/training/data_mlx/herbal_chat/test.jsonl`
- `program/training/data_mlx/herbal_chat/manifest.json`

## Augmentasi dari RAG Learning Log

Jika ingin model kecil lebih patuh pada jawaban grounded dari pipeline RAG aplikasi, buat dataset tambahan dari `data/learning/dual_llm_interactions.jsonl` lalu gabungkan ke dataset SFT utama:

```bash
python3 program/training/build_rag_lora_dataset.py
python3 program/training/prepare_mlx_lora_data.py \
  --input data/traning/combined_training_sft_rag.jsonl \
  --output program/training/data_mlx/herbal_chat_rag_light
```

Output tambahan:

- `data/traning/rag_learning_sft.jsonl`
- `data/traning/combined_training_sft_rag.jsonl`
- `data/traning/rag_learning_manifest.json`
- `program/training/data_mlx/herbal_chat_rag_light/*`

## Smoke Training

Smoke training dipakai untuk memastikan pipeline berjalan tanpa menghabiskan waktu dan storage.

```bash
program/training/.venv-mlx/bin/mlx_lm.lora -c program/training/mlx_lora_smoke.yaml
```

Output adapter:

```text
program/training/outputs/mlx-qwen25-05b-herbal-lora-smoke
```

Untuk smoke retraining model ringan yang sudah ditambah data grounded dari RAG log:

```bash
program/training/.venv-mlx/bin/mlx_lm.lora -c program/training/mlx_lora_rag_light_smoke.yaml
```

Output adapter:

```text
program/training/outputs/mlx-qwen25-05b-herbal-rag-lora-light-smoke
```

## Training Lebih Serius

```bash
program/training/.venv-mlx/bin/mlx_lm.lora -c program/training/mlx_lora_full.yaml
```

Output adapter:

```text
program/training/outputs/mlx-qwen25-05b-herbal-lora
```

Versi penuh untuk dataset gabungan SFT + learning log RAG:

```bash
program/training/.venv-mlx/bin/mlx_lm.lora -c program/training/mlx_lora_rag_light_full.yaml
```

Output adapter:

```text
program/training/outputs/mlx-qwen25-05b-herbal-rag-lora-light
```

## Uji Generate dengan Adapter

```bash
program/training/.venv-mlx/bin/mlx_lm.generate \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --adapter-path program/training/outputs/mlx-qwen25-05b-herbal-rag-lora-light-smoke \
  --prompt "Saya tenggorokan tidak nyaman sejak kemarin, tidak demam tinggi dan tidak sesak. Rekomendasikan ramuan herbal yang aman."
```

## Catatan

- Model default lokal memakai `mlx-community/Qwen2.5-0.5B-Instruct-4bit` agar aman untuk MacBook Air M4 16 GB dan storage terbatas.
- Untuk kualitas lebih baik, ganti model ke `mlx-community/Qwen3-0.6B-4bit` atau model Gemma/DeepSeek MLX yang lebih besar bila storage dan RAM cukup.
- Augmentasi dari learning log membuat LoRA lebih dekat dengan jawaban grounded dari pipeline RAG aplikasi, sambil tetap menjaga model tetap kecil.
- LoRA meningkatkan kepatuhan format dan domain knowledge, tetapi latency aplikasi masih paling banyak dipengaruhi oleh desain inference. Untuk deployment ringan, lebih masuk akal memakai satu model kecil + adapter + RAG daripada dua model besar yang dibandingkan sekaligus.
