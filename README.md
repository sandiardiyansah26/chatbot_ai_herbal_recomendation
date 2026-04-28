# Program AI Chatbot Rekomendasi Ramuan Herbal

Prototype ini adalah chatbot web berbasis FastAPI + Ollama/OpenAI untuk `humanizing anamnesis` dan rekomendasi ramuan herbal. Flow utama sekarang memakai `multi-LLM anamnesis-first`: keluhan user dikirim ke `deepseek-r1:7b`, `gemma4:latest`, dan opsional `gpt-4o` bila `OPENAI_API_KEY` tersedia. Hasil model di-ranking, lalu chatbot bertanya lanjutan maksimal `3` kali sebelum memberi dugaan kondisi, rekomendasi herbal, cara pengolahan, dan catatan kewaspadaan.

## Fitur Utama
- Frontend web Bahasa Indonesia dengan riwayat chat lokal.
- Backend FastAPI dengan session memory percakapan.
- RAG lokal memakai ChromaDB persistent vector store dan fallback TF-IDF.
- Knowledge base gabungan dari data referensi herbal, data anamnesis, dan data training terkurasi.
- Multi-model comparison untuk `follow_up` dan `recommendation`.
- Ranking kandidat berbasis `safety`, `grounding`, `completeness`, `relevance`, `language`, plus `question_quality` atau `herbal_fit`.
- Guardrail red flag, out-of-scope, dan referral untuk kasus `penyakit dalam` atau kondisi kritis.
- Learning capture untuk setiap turn percakapan, kandidat model, feedback rekomendasi, dan kandidat enrichment knowledge base.
- Pipeline training/LoRA tersedia di folder `training/`.

## Model Runtime Saat Ini
- `Model A`: `deepseek-r1:7b`
- `Model B`: `gemma4:latest`
- `Model C opsional`: `gpt-4o` via OpenAI API bila `OPENAI_API_KEY` diset
- Fallback DeepSeek: `deepseek-r1:1.5b`
- Fallback Gemma: `gemma4:e2b`, `gemma3:1b`

Konfigurasi model runtime sekarang dipusatkan di `app.config.sh`. Untuk mengganti model, edit `OLLAMA_MODEL_A`, `OLLAMA_MODEL_B`, atau `OPENAI_MODEL`, lalu jalankan `./app.config.sh restart`.

## Flow Sistem Saat Ini
1. User mengirim keluhan awal ke `/api/chat`.
2. Backend menggabungkan riwayat gejala user dalam satu sesi.
3. Modul anamnesis mendeteksi gejala, durasi, negasi, permintaan rekomendasi, dan red flag.
4. Jika ada red flag, sistem langsung memberi guardrail tanpa memaksa jalur herbal.
5. Jika aman, backend melakukan retrieval ke knowledge base.
6. Pada turn follow-up, DeepSeek, Gemma, dan GPT-4 opsional dipanggil untuk menghasilkan assessment terstruktur JSON.
7. Output model di-ranking, lalu chatbot memilih satu pertanyaan anamnesis terbaik.
8. Sistem mengulang langkah ini sampai informasi cukup atau jumlah pertanyaan mencapai `3`.
9. Setelah itu backend menjalankan final recommendation comparison, memilih jawaban akhir terbaik, lalu menyusun:
   - dugaan kondisi
   - rekomendasi herbal
   - bahan
   - cara pengolahan
   - dosis
   - catatan kewaspadaan
10. Semua turn, skor model, dan snapshot sesi disimpan sebagai data pembelajaran.

Catatan implementasi lokal:
- `follow_up` dijalankan secara paralel untuk kandidat model yang aktif.
- `recommendation` dijalankan berurutan agar lebih stabil pada mesin lokal dan mengurangi timeout karena kontensi memori/model.

## Struktur Data
- `data/referensi/`: dataset case herbal, formula, dan referensi herbal.
- `data/anamnesis/`: pertanyaan anamnesis, referensi anamnesis, dan SFT anamnesis.
- `data/traning/`: training records herbal, tropical disease guidance, combined SFT, dan dataset retraining lain.
- `data/learning/dual_llm_interactions.jsonl`: log komparasi model aktif.
- `data/learning/conversation_turns.jsonl`: log tiap turn user/assistant.
- `data/learning/kb_enrichment_candidates.jsonl`: kandidat enrichment knowledge base dari percakapan.
- `data/learning/recommendation_feedback.jsonl`: umpan balik user apakah rekomendasi membantu, belum membantu, atau cara pengolahan kurang jelas.
- `data/traning/herbal_preparation_training_records.jsonl`: data training tambahan untuk detail cara pengolahan ramuan herbal.
- `data/traning/herbal_preparation_training_sft.jsonl`: contoh SFT khusus pertanyaan cara membuat/mengolah ramuan.

## Menjalankan dengan Docker
```bash
./app.config.sh start
```

Frontend:
```text
http://localhost:5173
```

Backend docs:
```text
http://localhost:8000/docs
```

Health check:
```bash
curl http://127.0.0.1:8000/health
```

Perintah operasional satu file:

```bash
./app.config.sh restart
./app.config.sh logs backend
./app.config.sh ps
./app.config.sh down
```

Script ini berada sejajar dengan `docker-compose.yml`, mengekspor environment backend, membuat `frontend/runtime-config.js`, lalu menjalankan Docker Compose.

## Menjalankan Backend Lokal
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Menyiapkan Ollama
Pull model utama:

```bash
./app.config.sh pull-models
```

Opsional, buat wrapper model herbal dari `ollama/Modelfile.*`:

```bash
ollama create deepseek-r1-herbal -f ollama/Modelfile.deepseek-r1-herbal
ollama create gemma4-herbal -f ollama/Modelfile.gemma4-herbal
```

## Environment Penting
Semua environment runtime utama sekarang bisa diubah dari file root `app.config.sh`:

```bash
OLLAMA_MODEL_A="deepseek-r1:7b"
OLLAMA_MODEL_B="gemma4:latest"
BACKEND_PORT="8000"
FRONTEND_PORT="5173"
MAX_ANAMNESIS_QUESTIONS="3"
```

Komparasi GPT-4/OpenAI opsional:

```bash
ENABLE_OPENAI_COMPARISON="true"
OPENAI_API_KEY=""
OPENAI_MODEL="gpt-4o"
```

Jika `OPENAI_API_KEY` tidak diset, aplikasi tetap berjalan hanya dengan model Ollama lokal.

Log learning:

```bash
export LEARNING_LOG_PATH=data/learning/dual_llm_interactions.jsonl
export CONVERSATION_LOG_PATH=data/learning/conversation_turns.jsonl
export KB_ENRICHMENT_LOG_PATH=data/learning/kb_enrichment_candidates.jsonl
export RECOMMENDATION_FEEDBACK_LOG_PATH=data/learning/recommendation_feedback.jsonl
```

## Benchmark Runtime Terbaru
Angka ini diambil dari runtime aplikasi yang sekarang:

- `follow_up` turn pertama: sekitar `41.34 detik`
  - DeepSeek: `41.09 detik`
  - Gemma: `6.75 detik`
  - model terpilih: `deepseek-r1:7b`
- `final recommendation` turn akhir: sekitar `61.69 detik`
  - DeepSeek: `23.02 detik`
  - Gemma: `38.62 detik`
  - model terpilih: `gemma4:latest`
- `red_flag`: sekitar `0.14 detik`

Interpretasi penting:
- Bottleneck terbesar ada di inference model, bukan di retrieval.
- Final recommendation memang lebih lambat karena kandidat model aktif tetap dievaluasi, lalu hasilnya di-ranking.
- Walaupun lebih lambat, flow ini memenuhi objective penelitian untuk membandingkan performa model pada task anamnesis dan rekomendasi.

## Dataset Builder
Generate ulang dataset anamnesis:

```bash
python3 tools/build_anamnesis_dataset.py
```

Generate dataset disease tropic:

```bash
python3 tools/build_tropical_disease_dataset.py
```

Generate dataset detail cara pengolahan ramuan herbal:

```bash
python3 tools/build_herbal_preparation_dataset.py
```

Scrape/kurasi sumber herbal:

```bash
python3 tools/scrape_herbal_sources.py
```

## Training dan LoRA
Panduan detail ada di `training/README_QLORA.md`.

Contoh:

```bash
cd training
python3 qlora_finetune.py --config qlora_config.deepseek-r1.example.json --check_environment
python3 qlora_finetune.py --config qlora_config.deepseek-r1.example.json --dry_run
```

Catatan:
- QLoRA `bitsandbytes` tetap paling cocok di Linux + NVIDIA CUDA.
- Di Mac Apple Silicon, repo ini lebih cocok untuk validasi data, orchestrator, evaluasi log, dan MLX/eksperimen ringan.
- Learning logs dari aplikasi ini memang disiapkan untuk kurasi dan retraining berikutnya, tetapi tidak langsung mengubah bobot model saat chat berjalan.

## Menjalankan Test
Di container backend:

```bash
docker compose exec -T backend python -m unittest tests.test_services
```

Atau lokal:

```bash
cd backend
PYTHONPATH=. python -m unittest discover -s tests
```

## Batasan Sistem
- Sistem tidak memberi diagnosis medis final.
- Scope diarahkan ke keluhan/penyakit non-kritis dan non-penyakit dalam.
- Jika muncul red flag atau indikasi kasus berat, sistem mengarahkan ke tenaga kesehatan.
- Knowledge base enrichment dari percakapan belum otomatis masuk ke KB final; statusnya tetap `pending_curation`.

## Dokumen Tesis
Diagram dan dokumen tesis ada di:
- `document/tesis/Tesis_AI_Chatbot_Herbal_Doctor_Recommendation_UPH.html`
- `document/tesis/diagrams/`

Diagram workflow runtime terbaru tersimpan di:
- `document/tesis/diagrams/workflow_runtime_respons_model_baru.mmd`
