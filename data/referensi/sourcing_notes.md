# Catatan Kurasi Referensi Herbal

## Prinsip kurasi
- Prioritas sumber:
  - artikel resmi Kementerian Kesehatan RI,
  - publikasi ilmiah terindeks PubMed,
  - studi klinis atau systematic review,
  - laporan penelitian kesehatan yang relevan dengan konteks Indonesia.
- Fokus klaim:
  - **membantu meringankan gejala**,
  - bukan **menyembuhkan** atau **mendiagnosis** penyakit tropis.
- Penempatan herbal:
  - sebagai **pendamping gejala**,
  - bukan pengganti penanganan medis.

## Mengapa tidak semua herbal diberi level bukti tinggi
Beberapa tanaman herbal sangat populer di Indonesia, tetapi bukti ilmiah yang langsung mengaitkan tanaman tersebut dengan gejala spesifik penyakit tropis belum selalu kuat. Karena itu data dibagi menjadi tiga level:

- `high`
  Cocok dipakai model sebagai rekomendasi gejala pendamping dengan keyakinan relatif lebih tinggi.
- `medium`
  Cocok dipakai, tetapi sebaiknya dengan bahasa lebih hati-hati.
- `low_to_medium`
  Sebaiknya diposisikan sebagai opsi tradisional tambahan, bukan rekomendasi utama.

## Rekomendasi penggunaan untuk LLM
- Gunakan file ini untuk **retrieval** atau **grounded generation**, bukan sebagai dasar satu-satunya.
- Tambahkan aturan bahwa model:
  - hanya merekomendasikan herbal untuk **gejala ringan**,
  - wajib menolak klaim diagnosis pasti,
  - wajib mengeluarkan disclaimer,
  - wajib melakukan red-flag screening.
- Untuk ramuan kombinasi seperti `madu + kayu manis + serai`, beri label:
  - `traditional_formula_inference: true`
  - artinya manfaat kombinasi merupakan inferensi kurasi dari masing-masing bahan, bukan uji klinis langsung atas formula.

## Red flags yang harus mengalahkan rekomendasi herbal
- demam tinggi lebih dari 3 hari,
- perdarahan,
- muntah terus-menerus,
- diare berat atau dehidrasi,
- sesak napas,
- penurunan kesadaran,
- anak sangat lemah atau tidak mau minum,
- nyeri kepala berat dengan kaku kuduk,
- kejang.

## Saran langkah berikutnya
- Tambahkan referensi resmi BPOM atau farmakope herbal Indonesia bila tersedia.
- Ubah dataset ini menjadi format:
  - `qa_pairs.jsonl`,
  - `symptom_to_herb_rules.yaml`,
  - `red_flag_rules.yaml`,
  - `retrieval_chunks.md`.
- Libatkan reviewer tenaga kesehatan untuk validasi akhir sebelum dipakai di sistem produksi.

## Catatan formula kombinasi
- Formula kombinasi dipisahkan ke:
  - `herbal_formulas.csv`
  - `herbal_formulas.jsonl`
- Ada dua jenis formula:
  - formula yang memang disebut di sumber resmi jamu/tradisional Indonesia, misalnya `kunyit asam`, `beras kencur`, `pahitan`
  - formula inferensi terkurasi dari beberapa bahan yang masing-masing punya manfaat pendukung, misalnya `jahe-madu` atau `serai-kayu manis-madu`
- Untuk formula inferensi, field `traditional_formula_inference` harus dibaca model sebagai sinyal bahwa bukti formula gabungan lebih lemah dibanding bukti masing-masing bahan.

## Catatan pengolahan dan dosis
- Panduan pengolahan dan dosis disimpan terpisah di:
  - `pengolahan_dan_dosis_ramuan.md`
  - `pengolahan_dan_dosis_ramuan.csv`
- Dosis pada file tersebut dibagi dua:
  - dosis yang punya rujukan relatif lebih jelas dari sumber resmi/literatur,
  - `kisaran penggunaan tradisional konservatif` untuk formula yang umum dipakai masyarakat tetapi tidak memiliki dosis rumah tangga baku dari sumber primer.
- Untuk LLM, sebaiknya tambahkan field:
  - `dose_confidence`
  - `only_for_mild_symptoms`
  - `not_a_definitive_medical_prescription`
