# Program AI Chatbot Rekomendasi Ramuan Herbal

Prototype ini adalah implementasi awal dari tesis/BRD: chatbot web yang melakukan humanizing anamnesis ringan sebelum memberikan rekomendasi ramuan herbal, dosis/kisaran penggunaan, cara pengolahan, catatan kewaspadaan, dan disclaimer.

## Fitur
- Chatbot berbasis web dalam Bahasa Indonesia.
- Backend Python FastAPI.
- Knowledge base dari `../data/referensi/*.csv`, data anamnesis di `../data/anamnesis/*.jsonl`, dan data training tambahan di `../data/traning/*.jsonl`.
- Alur humanizing anamnesis dua tahap: deteksi keluhan, pertanyaan lanjutan, lalu rekomendasi.
- Red flag sederhana untuk kondisi di luar batas keluhan ringan.
- RAG lokal dengan chunked knowledge base, ChromaDB persistent vector store, local hashing embedding ringan, fallback in-memory TF-IDF, dan metadata-aware re-ranking.
- Grounded recommendation generation: respons disusun dari case dan konteks retrieval, bukan dari klaim bebas.
- Dual LLM comparison via Ollama: prompt dikirim ke `deepseek-r1` dan model keluarga Gemma, lalu kandidat jawaban diberi skor dan pemenangnya ditampilkan.
- Learning log: setiap prompt, konteks RAG, kandidat jawaban, skor, dan jawaban terpilih dicatat ke `../data/learning/dual_llm_interactions.jsonl` untuk bahan kurasi/fine-tuning berikutnya.
- Quick replies dinamis agar chatbot lebih interaktif setelah anamnesis, rekomendasi, dan red flag.
- Riwayat percakapan lokal seperti ChatGPT dengan sidebar, tombol `+ Chat Baru`, dan penyimpanan percakapan per sesi di browser.
- Struktur training QLoRA tersedia di `training/` untuk meringankan fine-tuning model instruksional.

## Pipeline Sesuai Metodologi Tesis
1. Humanizing anamnesis mengekstrak gejala, durasi, permintaan rekomendasi, safety clearance, dan red flag.
2. Knowledge base diubah menjadi chunk: case keluhan ringan, formula ramuan, tanaman herbal, record anamnesis, dan record training hasil scraping/kurasi.
3. ChromaDB persistent vector store dipakai sebagai vector database lokal. Bila ChromaDB tidak tersedia, sistem fallback ke in-memory TF-IDF.
4. Retriever mengambil kandidat case dan konteks pendukung.
5. Re-ranker memberi boost berdasarkan kecocokan gejala, nama formula, bahan ramuan, dan evidence level.
6. Grounded generator menyusun jawaban hanya dari case dan konteks retrieval yang terpilih.
7. Prompt RAG dikirim ke dua model Ollama: DeepSeek-R1 dan Gemma/Gemma4.
8. Candidate scorer menilai keamanan, grounding ke RAG, kelengkapan jawaban, dan kualitas bahasa.
9. Jawaban dengan skor tertinggi ditampilkan sebagai respons utama, sedangkan skor kandidat bisa dilihat di panel komparasi.
10. Guardrail menolak kondisi red flag/out-of-scope dan selalu menambahkan disclaimer.

Konsep ini mengikuti prinsip Retrieval-Augmented Generation (RAG): model generatif tidak dibiarkan menjawab hanya dari memori internal, tetapi diberi konteks eksternal yang relevan dari knowledge base sehingga respons lebih spesifik, mudah diperbarui, dan lebih rendah risiko halusinasi.

## Dual LLM Comparison dengan Ollama

Install dan jalankan Ollama, lalu pull model dasar:

```bash
ollama pull deepseek-r1:1.5b
ollama pull gemma4:e2b
```

Jika `gemma4:e2b` belum tersedia lokal atau proses pull gagal, siapkan fallback yang tetap kompatibel:

```bash
ollama pull gemma4:latest
ollama pull gemma3:1b
export OLLAMA_MODEL_B_FALLBACKS=gemma4:latest,gemma3:1b
```

Backend akan mencoba `OLLAMA_MODEL_B` terlebih dahulu. Jika Ollama mengembalikan `404 model not found`, backend otomatis mencoba model fallback agar komparasi tetap berjalan.

Default runtime repo ini sekarang memakai `deepseek-r1:1.5b` dan `gemma4:e2b` agar latency lebih realistis untuk laptop/dev machine. Jika ingin menaikkan kualitas model pada mesin yang lebih kuat, override lewat environment variable:

```bash
export OLLAMA_MODEL_A=deepseek-r1:70b
export OLLAMA_MODEL_B=gemma4:latest
```

Opsional, buat wrapper model herbal berbasis `Modelfile`:

```bash
cd program
ollama create deepseek-r1-herbal -f ollama/Modelfile.deepseek-r1-herbal
ollama create gemma4-herbal -f ollama/Modelfile.gemma4-herbal
export OLLAMA_MODEL_A=deepseek-r1-herbal
export OLLAMA_MODEL_B=gemma4-herbal
```

Untuk menjaga latency tetap masuk akal, backend sekarang mengirim dua request model secara paralel dan memakai budget output yang lebih pendek untuk respons follow-up, red flag, dan out-of-scope. Budget ini bisa diatur lewat:

```bash
export OLLAMA_NUM_PREDICT_DEFAULT=220
export OLLAMA_NUM_PREDICT_RECOMMENDATION=768
export OLLAMA_NUM_PREDICT_FOLLOW_UP=140
export OLLAMA_NUM_PREDICT_RED_FLAG=120
export OLLAMA_NUM_PREDICT_OUT_OF_SCOPE=120
```

Scoring kandidat saat ini bersifat deterministik dengan bobot:

- `safety`: kepatuhan pada disclaimer, tanda bahaya, dan arahan ke tenaga kesehatan.
- `grounding`: kecocokan istilah dengan rekomendasi dan konteks RAG yang terambil.
- `completeness`: kelengkapan anamnesis atau rekomendasi ramuan, dosis, pengolahan, dan kewaspadaan.
- `language`: panjang dan keterbacaan jawaban Bahasa Indonesia.

Catatan penting: proses chat tidak langsung mengubah bobot model secara otomatis. Interaksi disimpan sebagai learning log agar bisa dikurasi, dievaluasi, lalu dipakai untuk QLoRA/SFT pada tahap training berikutnya. Pendekatan ini lebih aman karena perubahan model tetap dapat diaudit.

## Menjalankan dengan Docker
```bash
cd program
cp .env.example .env
docker compose up --build -d
```

Saat memakai Docker di macOS/Windows, backend mengakses Ollama host melalui `http://host.docker.internal:11434`. Pastikan Ollama berjalan di host sebelum mencoba dual LLM comparison.

`docker-compose.yml` sekarang me-mount seluruh folder `../data` ke `/app/data` dan menyimpan ChromaDB persisten di `program/.chroma`, jadi referensi, dataset anamnesis, training records, dan learning log semuanya ikut tersedia di container.

Frontend:
```text
http://localhost:5173
```

Backend API:
```text
http://localhost:8000/docs
```

Health check:
```bash
curl http://localhost:8000/health
```

## Menjalankan Backend Lokal
```bash
cd program/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

ChromaDB secara default disimpan di `program/.chroma`. Lokasinya bisa diganti:
```bash
CHROMA_DB_DIR=.chroma uvicorn app.main:app --reload --port 8000
```

## Mengumpulkan Data Training
Script scraper/kurasi menggabungkan sumber web Kemenkes/AyoSehat/RISTOJA dengan dataset lokal, lalu menulis data ke `../data/traning`.

```bash
cd program
python3 tools/scrape_herbal_sources.py
```

Output utama:
- `../data/traning/scraped_sources.jsonl`: metadata sumber dan excerpt pendek, bukan salinan halaman penuh.
- `../data/traning/herbal_training_records.jsonl`: record terstruktur untuk RAG dan kurasi.
- `../data/traning/herbal_training_sft.jsonl`: contoh percakapan untuk QLoRA/SFT.

## Dataset Anamnesis
Pertanyaan anamnesis terstruktur dapat digenerate dengan:

```bash
python3 program/tools/build_anamnesis_dataset.py
```

Output utama:
- `../data/anamnesis/anamnesis_questions.jsonl`: pertanyaan wajib, pertanyaan red flag, tindakan triase, dan sumber.
- `../data/anamnesis/anamnesis_training_sft.jsonl`: contoh percakapan anamnesis.
- `../data/traning/combined_training_sft.jsonl`: gabungan data herbal + anamnesis untuk QLoRA.

## QLoRA Fine-Tuning
Pipeline QLoRA tersedia sebagai scaffolding. Disarankan dijalankan di Linux GPU dengan VRAM memadai karena `bitsandbytes` 4-bit tidak selalu stabil di Mac/CPU.

```bash
cd program/training
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-qlora.txt
python qlora_finetune.py --config qlora_config.deepseek-r1.example.json
python qlora_finetune.py --config qlora_config.gemma4.example.json
```

Adapter hasil training akan tersimpan di folder `output_dir` pada config. Backend saat ini tetap memakai RAG lokal; adapter QLoRA dapat dipakai pada tahap integrasi model generatif berikutnya.

Learning log dari aplikasi dapat digabungkan ke dataset SFT setelah direview:

```text
../data/learning/dual_llm_interactions.jsonl
```

## Menjalankan Test
```bash
cd program/backend
PYTHONPATH=. .venv/bin/python -m unittest discover -s tests
```

Atau lewat container backend yang sedang aktif:

```bash
cd program
docker exec herbal-chatbot-backend python -m unittest discover -s tests
```

## Contoh Percakapan
Pengguna:
```text
saya mual ringan sejak tadi pagi, tolong rekomendasi ramuan herbal
```

Sistem akan memberikan rekomendasi seperti jahe/jahe-madu bila tidak ada red flag.

Pengguna:
```text
tenggorokan saya tidak nyaman
```

Sistem akan bertanya lanjutan terlebih dahulu sebelum memberi rekomendasi.

## Catatan Batasan
Prototype ini belum melakukan diagnosis medis final. ChromaDB sudah dipakai untuk retrieval lokal, Ollama dipakai untuk komparasi dua kandidat LLM bila model tersedia, sedangkan QLoRA masih berupa pipeline training/adapter untuk tahap berikutnya. Semua rekomendasi tetap dibatasi pada keluhan ringan dan perlu review tenaga kesehatan sebelum produksi.
