# Data Traning Chatbot Herbal

Folder ini berisi data training/kurasi untuk prototype chatbot rekomendasi ramuan herbal.
Nama folder mengikuti permintaan proyek (`traning`).

## File
- `scraped_sources.jsonl`: metadata sumber web dan excerpt pendek hasil scraping. Excerpt dibatasi agar tidak menyalin halaman penuh.
- `herbal_training_records.jsonl`: record terstruktur ramuan, gejala, cara pengolahan, dosis/kisaran, kewaspadaan, dan sumber.
- `herbal_training_sft.jsonl`: contoh percakapan format messages untuk supervised fine-tuning/QLoRA.
- `herbal_preparation_sources.jsonl`: metadata sumber internet untuk detail cara pengolahan ramuan herbal.
- `herbal_preparation_training_records.jsonl`: record tambahan yang memperinci bahan, langkah pengolahan, dosis, dan batas keamanan.
- `herbal_preparation_training_sft.jsonl`: contoh percakapan SFT khusus pertanyaan "bagaimana cara membuat/mengolah ramuan".
- `medlineplus_guidance_sources.jsonl`: daftar topik MedlinePlus resmi yang dipilih untuk edukasi gejala, triase, follow-up, dan pencegahan.
- `medlineplus_guidance_training_records.jsonl`: record `disease_guidance` skala besar hasil kurasi dari MedlinePlus XML resmi.
- `medlineplus_guidance_training_sft.jsonl`: contoh percakapan SFT berbasis topik gejala/penyakit MedlinePlus.
- `medlineplus_guidance_manifest.json`: manifest jumlah topik, aturan filter, dan tanggal snapshot XML MedlinePlus.
- `anamnesis_training_sft.jsonl`: contoh percakapan anamnesis bila sudah digenerate dari `program/tools/build_anamnesis_dataset.py`.
- `combined_training_sft.jsonl`: gabungan data herbal dan data anamnesis bila `anamnesis_training_sft.jsonl` sudah tersedia.
- `combined_training_sft_rag.jsonl`: gabungan data SFT utama dengan log learning RAG yang sudah dibersihkan.

## Catatan Kurasi
- Jumlah record training herbal dasar: 52.
- Jumlah contoh SFT herbal: 104.
- Jumlah sumber detail pengolahan herbal: 7.
- Jumlah record detail pengolahan herbal: 9.
- Jumlah record disease guidance MedlinePlus: 2660.
- Jumlah total runtime training records yang dibaca knowledge base: 2730.
- Jumlah contoh SFT anamnesis: 1200.
- Jumlah contoh SFT gabungan utama: 3991.
- Jumlah contoh SFT gabungan + learning log RAG: 4071.
- Data anamnesis diperluas dengan variasi synthetic-guided by curated references dari sumber resmi dan paper pendukung. Data ini bukan hasil diagnosis klinis dan tetap perlu review ahli sebelum produksi.
- Dataset MedlinePlus dibangun dari XML resmi snapshot harian dan diekspansi menjadi lima varian record per topik: overview, triage, follow-up, prevention, dan self-care.
- Data ini digunakan untuk edukasi dan rekomendasi awal keluhan ringan, bukan diagnosis medis final.
- Record dari sumber web tetap dikurasi manual agar tidak mengambil klaim mentah tanpa batasan dosis dan kewaspadaan.
- Untuk produksi, setiap record perlu review tenaga kesehatan/herbalis dan uji keamanan lebih lanjut.

## Regenerasi
```bash
python3 tools/scrape_herbal_sources.py
python3 tools/build_herbal_preparation_dataset.py
python3 tools/build_medlineplus_guidance_dataset.py
python3 tools/build_anamnesis_dataset.py
python3 training/build_rag_lora_dataset.py
python3 training/prepare_mlx_lora_data.py --input data/traning/combined_training_sft_rag.jsonl --output training/data_mlx/herbal_chat_rag_light_current
```
