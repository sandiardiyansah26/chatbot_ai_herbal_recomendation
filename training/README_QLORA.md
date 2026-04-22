# QLoRA Fine-Tuning Plan

QLoRA memungkinkan untuk proyek ini, tetapi tidak langsung dilakukan pada model `.gguf` atau model yang sudah berada di Ollama. Alur yang benar adalah fine-tuning model dasar dari Hugging Face, menyimpan LoRA adapter, lalu menyiapkan hasilnya untuk inference lokal.

## Tujuan Optimasi

- Membuat model lebih patuh terhadap alur humanizing anamnesis.
- Membuat jawaban lebih konsisten dalam Bahasa Indonesia.
- Mengurangi kecenderungan model mengarang ramuan atau dosis di luar konteks RAG.
- Membuat format jawaban lebih stabil: gejala, pertanyaan lanjutan, ramuan, dosis, kewaspadaan, dan disclaimer.

QLoRA tidak otomatis membuat inference jauh lebih cepat. Latency aplikasi saat ini lebih banyak disebabkan oleh dua model Ollama yang dipanggil berurutan. Untuk latency, optimasi yang paling berdampak adalah parallel inference, streaming response, model yang lebih kecil, atau menampilkan baseline RAG lebih dulu.

## Kebutuhan Environment

Script `qlora_finetune.py` memakai `bitsandbytes` 4-bit, sehingga paling aman dijalankan pada Linux dengan NVIDIA GPU/CUDA.

Mac Apple Silicon dapat dipakai untuk validasi dataset dan konfigurasi, tetapi bukan target ideal untuk QLoRA 4-bit berbasis `bitsandbytes`.

## Cek Kesiapan

```bash
cd program/training
python3 qlora_finetune.py --config qlora_config.deepseek-r1.example.json --check_environment
```

Dry-run rencana training:

```bash
cd program/training
python3 qlora_finetune.py --config qlora_config.deepseek-r1.example.json --dry_run
```

## Instalasi di Mesin Training CUDA

```bash
cd program/training
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-qlora.txt
```

## Training DeepSeek-R1 Distill

```bash
cd program/training
source .venv/bin/activate
python qlora_finetune.py --config qlora_config.deepseek-r1.example.json
```

## Training Gemma

```bash
cd program/training
source .venv/bin/activate
python qlora_finetune.py --config qlora_config.gemma4.example.json
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
- `data/traning/anamnesis_training_sft.jsonl`
- `data/traning/herbal_training_sft.jsonl`

Sebelum training serius, dataset sebaiknya dikurasi lagi untuk memastikan tidak ada dosis berlebihan, klaim menyembuhkan, atau rekomendasi untuk kondisi red flag.
