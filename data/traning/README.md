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
- `anamnesis_training_sft.jsonl`: contoh percakapan anamnesis bila sudah digenerate dari `program/tools/build_anamnesis_dataset.py`.
- `combined_training_sft.jsonl`: gabungan data herbal dan data anamnesis bila `anamnesis_training_sft.jsonl` sudah tersedia.

## Catatan Kurasi
- Jumlah sumber web yang diproses: 11.
- Jumlah record training terstruktur: 52.
- Jumlah contoh SFT herbal: 104.
- Jumlah sumber detail pengolahan herbal: 7.
- Jumlah record detail pengolahan herbal: 9.
- Jumlah contoh SFT anamnesis: 1200.
- Jumlah contoh SFT gabungan untuk QLoRA: 1331.
- Data anamnesis diperluas dengan variasi synthetic-guided by curated references dari sumber resmi dan paper pendukung. Data ini bukan hasil diagnosis klinis dan tetap perlu review ahli sebelum produksi.
- Data ini digunakan untuk edukasi dan rekomendasi awal keluhan ringan, bukan diagnosis medis final.
- Record dari sumber web tetap dikurasi manual agar tidak mengambil klaim mentah tanpa batasan dosis dan kewaspadaan.
- Untuk produksi, setiap record perlu review tenaga kesehatan/herbalis dan uji keamanan lebih lanjut.

## Regenerasi
```bash
cd program
python3 tools/scrape_herbal_sources.py
python3 tools/build_herbal_preparation_dataset.py
```
