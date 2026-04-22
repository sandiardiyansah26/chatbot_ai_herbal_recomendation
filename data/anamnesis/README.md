# Dataset Anamnesis Chatbot Herbal

Folder ini berisi pertanyaan anamnesis terstruktur untuk membantu chatbot menggali keluhan, durasi, faktor risiko, dan tanda bahaya sebelum memberi rekomendasi ramuan herbal.

## File
- `anamnesis_questions.jsonl`: dataset utama berisi kelompok kondisi, gejala, pertanyaan wajib, pertanyaan red flag, tindakan triase, dan sumber.
- `anamnesis_questions.csv`: versi spreadsheet dari dataset utama.
- `anamnesis_training_sft.jsonl`: contoh percakapan format messages untuk fine-tuning/SFT.
- `anamnesis_reference_sources.jsonl`: daftar sumber resmi dan paper yang digunakan untuk kurasi dataset anamnesis.

## Integrasi Training
- Salinan `anamnesis_training_sft.jsonl` juga dibuat di `../traning/anamnesis_training_sft.jsonl`.
- File gabungan QLoRA dibuat di `../traning/combined_training_sft.jsonl`.

## Batasan
- Dataset ini untuk deteksi awal dan triase edukatif, bukan diagnosis medis final.
- Jika muncul tanda bahaya, chatbot harus mengarahkan pengguna ke tenaga kesehatan/fasilitas kesehatan.
- Rekomendasi herbal hanya boleh diberikan untuk keluhan ringan yang sesuai batas sistem.

## Ringkasan
- Jumlah record anamnesis: 18.
- Jumlah contoh SFT anamnesis: 1200.
- Target minimum contoh SFT anamnesis: 1200.
- Contoh SFT anamnesis diperluas menggunakan variasi terstruktur atas profil pengguna, durasi, tingkat keluhan, intent, dan urutan pertanyaan. Data ini bersifat synthetic-guided by curated references, sehingga tetap perlu review ahli sebelum produksi.

## Sumber Utama
- [Tanda dan Gejala Demam Berdarah Dengue](https://keslan.kemkes.go.id/view_artikel/10/tanda-dan-gejala-demam-berdarah-dengue)
- [CDC Food Poisoning Symptoms](https://www.cdc.gov/food-safety/signs-symptoms/)
- [Dehidrasi pada Anak](https://ayosehat.kemkes.go.id/topik-penyakit/kesehatan-lainnya/dehidrasi-pada-anak)
- [Batuk dengan Pengobatan Tradisional](https://keslan.kemkes.go.id/view_artikel/653/batuk-dengan-pengobatan-tradisonal)
- [Transforming Health Care Through Chatbots for Medical History-Taking and Future Directions: Comprehensive Systematic Review](https://pmc.ncbi.nlm.nih.gov/articles/PMC11393511/)
- [WHO Dengue and Severe Dengue Fact Sheet](https://www.who.int/en/news-room/fact-sheets/detail/dengue-and-severe-dengue)
- [CDC Symptoms of Dengue and Testing](https://www.cdc.gov/dengue/signs-symptoms/index.html)
- [CDC Symptoms of Malaria](https://www.cdc.gov/malaria/symptoms/index.html)
- [Apa Saja Gejala Malaria?](https://malaria.kemkes.go.id/node/110)
- [Leptospirosis](https://keslan.kemkes.go.id/view_artikel/1952/leptospirosis)
- [CDC About Leptospirosis](https://www.cdc.gov/leptospirosis/index.html)
- [Malaria Bisa Dihentikan, Ayo Bertindak Sekarang](https://keslan.kemkes.go.id/view_artikel/4189/malaria-bisa-dihentikan-ayo-bertindak-sekarang)
- [Tipus/Penyakit Tipes](https://ayosehat.kemkes.go.id/topik-penyakit/infeksi-enterik/tipus-penyakit-tipes)
- [Mari Mengenal Obat Diare](https://keslan.kemkes.go.id/view_artikel/4105/mari-mengenal-obat-diare)
- [Gejala dan Pencegahan Chikungunya](https://ayosehat.kemkes.go.id/gejala-dan-pencegahan-chikungunya)
- [CDC Symptoms, Diagnosis, and Treatment of Chikungunya](https://www.cdc.gov/chikungunya/symptoms-diagnosis-treatment/index.html)
- [Infeksi Saluran Pernapasan Atas (ISPA)](https://keslan.kemkes.go.id/view_artikel/1792/infeksi-saluran-pernapasan-atas-ispa)
- [CDC About Common Cold](https://www.cdc.gov/common-cold/about/index.html)
- [CDC Sore Throat Basics](https://www.cdc.gov/sore-throat/about/index.html)
- [CDC Chest Cold (Acute Bronchitis) Basics](https://www.cdc.gov/acute-bronchitis/about/index.html)
- [CDC About Norovirus](https://www.cdc.gov/norovirus/about/index.html)
- [Aloe vera dermatitis literature](https://pubmed.ncbi.nlm.nih.gov/?term=aloe+vera+dermatitis+review)
- [Anamnesis Questions for Patient Assessment](https://www.scribd.com/document/912361832/Anamnesis-questions-1)
- [Waspada, Ini Tanda Sakit Kepala Berbahaya](https://www.halodoc.com/artikel/waspada-ini-tanda-sakit-kepala-berbahaya-1)
- [Apa itu Sakit Maag? Gejala, Penyebab & Pengobatan - Halodoc](https://www.halodoc.com/kesehatan/sakit-maag/)
- [Apa Itu Konstipasi (Sembelit)? Gejala, Penyebab, dan Pengobatan](https://www.halodoc.com/kesehatan/konstipasi/)
- [Apa Itu Nyeri Haid? Gejala, Penyebab, dan Pengobatan](https://www.halodoc.com/kesehatan/nyeri-haid)

## Paper Pendukung
- [Transforming Health Care Through Chatbots for Medical History-Taking and Future Directions: Comprehensive Systematic Review](https://pmc.ncbi.nlm.nih.gov/articles/PMC11393511/)
- [Retrieval augmented generation for large language models in healthcare: A systematic review](https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0000877)
- [Utilization of traditional and complementary medicine in Indonesia: Results of a national survey in 2014-15](https://pubmed.ncbi.nlm.nih.gov/30396615/)
- [Use of traditional medicines and traditional practitioners by children in Indonesia: findings from a national population survey in 2014-2015](https://pubmed.ncbi.nlm.nih.gov/31114218/)

## Regenerasi
```bash
python3 tools/build_anamnesis_dataset.py
```
