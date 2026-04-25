# QLoRA Fine-Tuning Plan

QLoRA memungkinkan untuk proyek ini, tetapi tidak langsung dilakukan pada model `.gguf` atau model yang sudah berada di Ollama. Alur yang benar adalah fine-tuning model dasar dari Hugging Face, menyimpan LoRA adapter, lalu menyiapkan hasilnya untuk inference lokal.

## Tujuan Optimasi

- Membuat model lebih patuh terhadap alur humanizing anamnesis.
- Membuat jawaban lebih konsisten dalam Bahasa Indonesia.
- Mengurangi kecenderungan model mengarang ramuan atau dosis di luar konteks RAG.
- Membuat format jawaban lebih stabil: gejala, pertanyaan lanjutan, ramuan, dosis, kewaspadaan, dan disclaimer.

QLoRA tidak otomatis membuat inference jauh lebih cepat. Latency aplikasi saat ini lebih banyak disebabkan oleh proses komparasi dua model pada flow anamnesis-first. Pada implementasi runtime terbaru, `follow_up` dijalankan paralel, sedangkan `recommendation` dijalankan berurutan agar lebih stabil di mesin lokal saat `deepseek-r1:7b` dan `gemma4:latest` dipakai bersama.

## Kebutuhan Environment

Script `qlora_finetune.py` memakai `bitsandbytes` 4-bit, sehingga paling aman dijalankan pada Linux dengan NVIDIA GPU/CUDA.

Mac Apple Silicon dapat dipakai untuk validasi dataset dan konfigurasi, tetapi bukan target ideal untuk QLoRA 4-bit berbasis `bitsandbytes`.

## Cek Kesiapan

```bash
cd training
python3 qlora_finetune.py --config qlora_config.deepseek-r1.example.json --check_environment
```

Dry-run rencana training:

```bash
cd training
python3 qlora_finetune.py --config qlora_config.deepseek-r1.example.json --dry_run
```

## Instalasi di Mesin Training CUDA

```bash
cd training
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-qlora.txt
```

## Training DeepSeek-R1 Distill

```bash
cd training
source .venv/bin/activate
python qlora_finetune.py --config qlora_config.deepseek-r1.example.json
```

Versi dataset yang sudah ditambah dokumen penyakit tropis:

```bash
cd training
source .venv/bin/activate
python qlora_finetune.py --config qlora_config.deepseek-r1.tropical.json
```

## Training Gemma

```bash
cd training
source .venv/bin/activate
python qlora_finetune.py --config qlora_config.gemma4.example.json
```

Versi dataset yang sudah ditambah dokumen penyakit tropis:

```bash
cd training
source .venv/bin/activate
python qlora_finetune.py --config qlora_config.gemma4.tropical.json
```

## Integrasi ke Ollama

Setelah adapter terbentuk, ada dua opsi:

1. Pakai model Hugging Face + adapter langsung dari backend inference service terpisah.
2. Merge adapter ke base model, convert ke GGUF, lalu buat model Ollama baru.

Contoh target nama model setelah integrasi:

```bash
OLLAMA_MODEL_A=deepseek-r1-herbal-qlora
OLLAMA_MODEL_B=gemma4-herbal-qlora
```

## Catatan Data

Dataset utama saat ini:

- `data/traning/combined_training_sft.jsonl`
- `data/anamnesis/anamnesis_training_sft.jsonl`
- `data/traning/herbal_training_sft.jsonl`
- `data/traning/tropical_disease_training_sft.jsonl`
- `data/learning/conversation_turns.jsonl`
- `data/learning/dual_llm_interactions.jsonl`
- `data/learning/kb_enrichment_candidates.jsonl`

Sebelum training serius, dataset sebaiknya dikurasi lagi untuk memastikan tidak ada dosis berlebihan, klaim menyembuhkan, rekomendasi untuk kondisi red flag, atau enrichment percakapan yang belum lolos review manusia.
