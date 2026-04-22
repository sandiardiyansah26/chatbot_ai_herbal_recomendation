# Referensi Herbal untuk Gejala Penyakit Tropis

Folder ini berisi referensi awal tanaman herbal yang **berpotensi membantu meringankan gejala** yang sering muncul pada penyakit tropis di Indonesia, misalnya:

- demam,
- sakit kepala,
- sakit tenggorokan,
- mual,
- gangguan pencernaan,
- diare,
- batuk ringan,
- pilek ringan,
- lemas,
- penurunan nafsu makan.

## Tujuan penggunaan
Data di folder ini disiapkan sebagai **knowledge base awal** untuk:

- penyusunan dataset kurasi,
- retrieval-augmented generation (RAG),
- penyusunan prompt grounding,
- evaluasi jawaban model,
- bahan awal sebelum dibuat dataset training/fine-tuning yang lebih ketat.

## Penting
- Data ini **bukan resep medis** dan **bukan pengganti diagnosis dokter**.
- Fokus referensi adalah **meringankan gejala**, bukan mengklaim menyembuhkan penyakit tropis.
- Untuk kondisi gawat darurat seperti demam tinggi berkepanjangan, perdarahan, sesak napas, penurunan kesadaran, dehidrasi berat, atau nyeri hebat, pengguna harus diarahkan ke fasilitas kesehatan.
- Beberapa herbal di sini memiliki **evidensi klinis lebih kuat** untuk gejala tertentu dibanding yang lain. Karena itu setiap entri diberi `evidence_level`.
- Untuk keperluan training LLM, data ini sebaiknya **dikurasi lagi** menjadi format tanya-jawab, aturan keselamatan, dan fakta terverifikasi. Saya tidak menyarankan memasukkan referensi mentah langsung ke fine-tuning tanpa proses kurasi.

## File
- `herbal_references.csv`
  Ringkasan tabel referensi yang mudah dibuka di spreadsheet.
- `herbal_references.jsonl`
  Format terstruktur yang lebih cocok untuk pipeline data.
- `herbal_formulas.csv`
  Daftar kombinasi ramuan/herbal formula yang umum digunakan.
- `herbal_formulas.jsonl`
  Format formula ramuan yang cocok untuk retrieval/rules engine.
- `penelitian_toga_dan_ramuan_tradisional.md`
  Ringkasan literatur tentang TOGA, ramuan tradisional, dan budidaya rumah tangga.
- `penelitian_toga_dan_ramuan_tradisional.csv`
  Tabel referensi penelitian/kebijakan yang siap dipakai untuk dataset literatur.
- `pengolahan_dan_dosis_ramuan.md`
  Panduan pengolahan yang benar dan kisaran dosis penggunaan tradisional.
- `pengolahan_dan_dosis_ramuan.csv`
  Versi tabel terstruktur untuk pipeline data.
- `sourcing_notes.md`
  Catatan kurasi, batasan, dan prinsip penggunaan medis yang aman.

## Skala evidence_level
- `high`
  Ada systematic review/meta-analysis atau uji klinis yang cukup jelas untuk gejala sasaran.
- `medium`
  Ada dukungan dari uji klinis terbatas, studi klinis lokal, atau gabungan penggunaan tradisional + publikasi ilmiah.
- `low_to_medium`
  Dukungan masih dominan dari penggunaan tradisional, review umum, atau data praklinis.

## Cakupan tanaman herbal awal
- Jahe (`Zingiber officinale`)
- Kunyit (`Curcuma longa`)
- Temulawak (`Curcuma xanthorrhiza`)
- Sambiloto (`Andrographis paniculata`)
- Meniran (`Phyllanthus niruri`)
- Daun/Buah Jambu Biji (`Psidium guajava`)
- Kencur (`Kaempferia galanga`)
- Madu
- Kayu manis (`Cinnamomum spp.`)
- Serai (`Cymbopogon citratus`)
- Jeruk nipis (`Citrus aurantiifolia`)

## Catatan kombinasi ramuan tradisional
Folder ini juga mendukung penggunaan kombinasi tradisional seperti:

- `serai + kayu manis` direbus dengan air,
- `madu` ditambahkan setelah air hangat,
- `kencur + jahe + sereh + madu` sebagai Wedang Sinden,
- `jeruk nipis + madu + air hangat`,

untuk membantu **meredakan rasa tidak nyaman pada tenggorokan ringan**, batuk ringan, suara serak, atau batuk pilek ringan.

Catatan penting:
- Klaim untuk **kombinasi** tersebut sebaiknya diposisikan sebagai **inferensi terkurasi** dari manfaat masing-masing bahan, bukan sebagai bukti klinis langsung bahwa kombinasi tersebut menyembuhkan penyakit tertentu.
- Madu **tidak boleh** diberikan pada bayi usia di bawah 12 bulan.
