# Program AI Chatbot Rekomendasi Ramuan Herbal

Prototype ini adalah chatbot web berbasis FastAPI + Ollama/OpenAI untuk deteksi awal kemungkinan penyakit tropis melalui `humanizing anamnesis` berbahasa Indonesia, RAG medical knowledge base, safety layer, dan rekomendasi herbal aman sebagai pendamping edukatif. Flow utama sekarang dimulai dari normalisasi/ekstraksi gejala ala IndoBERT/XLM-R, dilanjutkan DeepSeek-R1 sebagai reasoner anamnesis, retrieval ke knowledge base medis-herbal, lalu guardrail keselamatan sebelum output akhir.

## Fitur Utama
- Frontend web Bahasa Indonesia dengan riwayat chat lokal.
- Backend FastAPI dengan session memory percakapan.
- Lapisan IndoBERT/XLM-R-compatible untuk normalisasi bahasa Indonesia, slang/negasi, ekstraksi gejala, dan konteks risiko tropis.
- RAG lokal memakai ChromaDB persistent vector store dan fallback TF-IDF.
- Knowledge base gabungan dari data Kemenkes/WHO/CDC yang tersedia, dokumen penyakit tropis terkurasi, data anamnesis, referensi herbal, dan data training.
- Multi-model comparison untuk `follow_up` dan `recommendation`.
- Ranking kandidat berbasis `safety`, `grounding`, `completeness`, `relevance`, `language`, plus `question_quality` atau `herbal_fit`.
- Safety layer untuk red flag, emergency, disclaimer, rujukan dokter, dan pembatasan herbal agar tidak menjadi terapi utama pada dugaan penyakit tropis berat.
- Learning capture untuk setiap turn percakapan, kandidat model, feedback rekomendasi, dan kandidat enrichment knowledge base.
- Pipeline training/LoRA tersedia di folder `training/`.

## Model Runtime Saat Ini
- `Model A`: `deepseek-r1:7b`
- `Model B`: `gemma4:latest`
- `Model C opsional`: `gpt-4o` via OpenAI API bila `OPENAI_API_KEY` diset
- Fallback DeepSeek: `deepseek-r1:1.5b`
- Fallback Gemma: `gemma4:e2b`, `gemma3:1b`

## Flow Sistem Saat Ini
1. User mengirim keluhan awal ke `/api/chat`.
2. Backend menggabungkan riwayat gejala user dalam satu sesi.
3. IndoBERT/XLM-R-compatible layer menormalisasi bahasa Indonesia, slang, negasi, dan mengekstrak gejala/konteks risiko.
4. Modul anamnesis mendeteksi slot klinis, durasi, negasi, red flag, dan informasi yang sudah dijawab.
5. RAG mengambil konteks penyakit tropis, anamnesis, formula herbal, cara pengolahan, dan sumber evidensi.
6. DeepSeek-R1 menjalankan humanizing anamnesis dan reasoning pertanyaan lanjutan; Gemma/OpenAI opsional tetap dapat dibandingkan.
7. Safety layer memeriksa red flag, emergency, kebutuhan rujukan dokter, disclaimer, dan batasan herbal.
8. Sistem bertanya maksimal `3` kali bila informasi belum cukup.
9. Output akhir menyusun kemungkinan penyakit tropis/kondisi terkait, saran awal, dan rekomendasi herbal aman bila konteks RAG dan safety layer mengizinkan.
10. Semua turn, skor model, konteks RAG, dan snapshot sesi disimpan sebagai data pembelajaran.

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
docker compose up --build -d
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

Melihat flow runtime:
```bash
curl http://127.0.0.1:8000/api/system-flow
```

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
ollama pull deepseek-r1:7b
ollama pull gemma4:latest
```

Pull fallback:

```bash
ollama pull deepseek-r1:1.5b
ollama pull gemma4:e2b
ollama pull gemma3:1b
```

Opsional, buat wrapper model herbal dari `ollama/Modelfile.*`:

```bash
ollama create deepseek-r1-herbal -f ollama/Modelfile.deepseek-r1-herbal
ollama create gemma4-herbal -f ollama/Modelfile.gemma4-herbal
```

## Environment Penting
```bash
export OLLAMA_MODEL_A=deepseek-r1:1.5b
export OLLAMA_MODEL_B=gemma4:e2b
export OLLAMA_TIMEOUT_SECONDS=35
export OLLAMA_NUM_PREDICT_DEFAULT=160
export OLLAMA_NUM_PREDICT_FOLLOW_UP=120
export OLLAMA_NUM_PREDICT_RECOMMENDATION=420
export MAX_ANAMNESIS_QUESTIONS=3
export ENABLE_LLM_FOLLOW_UP=false
export NLP_MODEL_FAMILY=IndoBERT/XLM-R
export NLP_PRIMARY_MODEL=indobenchmark/indobert-base-p1
export NLP_FALLBACK_MODEL=xlm-roberta-base
export ENABLE_TRANSFORMER_NLP=false
```

Komparasi GPT-4/OpenAI opsional:

```bash
export ENABLE_OPENAI_COMPARISON=true
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4o
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_TIMEOUT_SECONDS=75
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

- `follow_up` turn pertama: sekitar `0.06-0.25 detik`
  - memakai fast anamnesis layer berbasis slot gejala + RAG
  - tidak menunggu inference LLM besar
- `final recommendation` turn akhir: sekitar `61.69 detik`
  - sekarang diarahkan ke model ringan `deepseek-r1:1.5b` dan `gemma4:e2b`
  - timeout default turun menjadi `35 detik`
- `red_flag`: sekitar `0.14 detik`

Interpretasi penting:
- Follow-up anamnesis dibuat cepat agar percakapan terasa hidup dan tidak menggantung.
- Bottleneck inference dipindahkan ke tahap final/reasoning, bukan pertanyaan anamnesis awal.
- Komparasi model tetap tersedia, tetapi memakai varian model yang lebih ringan secara default.

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
