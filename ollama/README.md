# Ollama Dual Model Setup

Folder ini berisi contoh `Modelfile` untuk membuat wrapper model lokal yang lebih selaras dengan domain chatbot ramuan herbal.

## Pull Model Dasar

```bash
ollama pull deepseek-r1:1.5b
ollama pull gemma4:e2b
```

Jika `gemma4:e2b` belum berhasil di-pull, gunakan fallback lokal agar backend tetap bisa melakukan komparasi:

```bash
ollama pull gemma4:latest
ollama pull gemma3:1b
export OLLAMA_MODEL_B_FALLBACKS=gemma4:latest,gemma3:1b
```

Backend akan memakai fallback hanya ketika model utama mengembalikan `404 model not found`.

Jika mesin cukup kuat, model dapat diganti melalui environment variable, misalnya `deepseek-r1:8b` atau `gemma4:e4b`.

## Buat Wrapper Herbal

```bash
cd program
ollama create deepseek-r1-herbal -f ollama/Modelfile.deepseek-r1-herbal
ollama create gemma4-herbal -f ollama/Modelfile.gemma4-herbal
```

Setelah itu backend dapat diarahkan ke wrapper:

```bash
OLLAMA_MODEL_A=deepseek-r1-herbal
OLLAMA_MODEL_B=gemma4-herbal
```

Catatan: `Modelfile` ini bukan fine-tuning bobot model. Ia memberi system instruction dan parameter inferensi. Fine-tuning/QLoRA tetap dilakukan melalui pipeline `program/training/`, sedangkan interaksi nyata dicatat ke `data/learning/dual_llm_interactions.jsonl` sebagai bahan kurasi training berikutnya.
