from __future__ import annotations

import csv
import html
import json
import re
import ssl
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = ROOT / "data" / "referensi"
TRAINING_DIR = ROOT / "data" / "traning"
SCRAPED_SOURCES_PATH = TRAINING_DIR / "scraped_sources.jsonl"
TRAINING_RECORDS_PATH = TRAINING_DIR / "herbal_training_records.jsonl"
SFT_PATH = TRAINING_DIR / "herbal_training_sft.jsonl"
ANAMNESIS_SFT_PATH = TRAINING_DIR / "anamnesis_training_sft.jsonl"
COMBINED_SFT_PATH = TRAINING_DIR / "combined_training_sft.jsonl"
README_PATH = TRAINING_DIR / "README.md"

SYSTEM_PROMPT = (
    "Anda adalah chatbot edukasi ramuan herbal berbahasa Indonesia. "
    "Lakukan anamnesis ringan, hanya rekomendasikan ramuan untuk keluhan ringan, "
    "jelaskan cara pengolahan dan dosis, serta selalu berikan batasan bahwa ini "
    "bukan diagnosis medis final."
)


@dataclass(frozen=True)
class SourceSeed:
    id: str
    url: str
    expected_title: str
    curated_records: list[dict[str, object]]


class BasicHTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._in_title = False
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        text = normalize_text(data)
        if not text or self._skip_depth:
            return
        if self._in_title:
            self.title_parts.append(text)
            return
        if len(text) > 24:
            self.text_parts.append(text)

    @property
    def title(self) -> str:
        return normalize_text(" ".join(self.title_parts))

    @property
    def text(self) -> str:
        return normalize_text(" ".join(self.text_parts))


SOURCE_SEEDS = [
    SourceSeed(
        id="kemenkes_jahe_mual",
        url="https://keslan.kemkes.go.id/view_artikel/978/takhlukan-mual-pasca-kemoterapi-dengan-jahe",
        expected_title="Takhlukan Mual Pasca Kemoterapi dengan Jahe",
        curated_records=[
            {
                "topic": "mual ringan",
                "formula_name": "Jahe hangat / Jahe-Madu",
                "ingredients": ["jahe", "madu opsional"],
                "symptoms": ["mual ringan", "perut tidak nyaman", "muntah ringan tanpa dehidrasi"],
                "preparation": "Cuci jahe, iris atau memarkan, rebus/seduh 10-15 menit, lalu tambahkan madu saat air sudah hangat bila cocok.",
                "dosage": "150-200 ml, 1-2 kali sehari untuk keluhan ringan.",
                "safety_notes": "Tidak untuk muntah terus-menerus, dehidrasi, nyeri perut hebat, atau demam tinggi. Hati-hati pada penggunaan pengencer darah.",
                "evidence_level": "medium",
            }
        ],
    ),
    SourceSeed(
        id="kemenkes_madu",
        url="https://keslan.kemkes.go.id/view_artikel/72/manfaat-madu-bagi-kesehatan",
        expected_title="Manfaat Madu bagi Kesehatan",
        curated_records=[
            {
                "topic": "tenggorokan tidak nyaman",
                "formula_name": "Madu hangat",
                "ingredients": ["madu", "air hangat"],
                "symptoms": ["iritasi tenggorokan ringan", "batuk ringan"],
                "preparation": "Campurkan madu ke air hangat, jangan ke air mendidih.",
                "dosage": "1-2 sendok teh madu dalam 150-200 ml air hangat, 1-2 kali sehari.",
                "safety_notes": "Madu tidak diberikan pada bayi di bawah 12 bulan. Hati-hati pada diabetes atau alergi produk lebah.",
                "evidence_level": "medium",
            }
        ],
    ),
    SourceSeed(
        id="kemenkes_tujuh_jamu",
        url="https://keslan.kemkes.go.id/view_artikel/2062/7-jamu-herbal-yang-wajib-kamu-tahu",
        expected_title="7 Jamu Herbal yang Wajib Kamu Tahu",
        curated_records=[
            {
                "topic": "badan kurang fit dan pegal ringan",
                "formula_name": "Beras Kencur",
                "ingredients": ["beras", "kencur", "jahe opsional", "kunyit opsional"],
                "symptoms": ["pegal ringan", "kurang fit", "nafsu makan menurun"],
                "preparation": "Rendam beras, haluskan bersama kencur dan rempah, campur air matang/rebusan, lalu saring.",
                "dosage": "150-200 ml, 1 kali sehari untuk pendamping keluhan ringan.",
                "safety_notes": "Tidak untuk nyeri berat, kelemahan anggota tubuh, demam tinggi, atau keluhan menetap.",
                "evidence_level": "low_to_medium",
            },
            {
                "topic": "nafsu makan menurun",
                "formula_name": "Temulawak-Kencur-Asem",
                "ingredients": ["temulawak", "kencur", "asem", "gula aren opsional"],
                "symptoms": ["nafsu makan menurun", "mual ringan", "lemas ringan"],
                "preparation": "Cuci rimpang, iris tipis, rebus bersama asem, lalu saring.",
                "dosage": "150-200 ml, 1 kali sehari.",
                "safety_notes": "Perlu evaluasi medis bila berat badan turun drastis, demam lama, muntah, atau anak sangat lemah.",
                "evidence_level": "medium",
            },
        ],
    ),
    SourceSeed(
        id="ayosehat_kencur_polusi",
        url="https://ayosehat.kemkes.go.id/kencur-untuk-atasi-dampak-polusi",
        expected_title="Kencur untuk Atasi Dampak Polusi",
        curated_records=[
            {
                "topic": "batuk ringan dan suara serak",
                "formula_name": "Wedang Sinden",
                "ingredients": ["kencur", "jahe", "sereh", "madu", "air jeruk opsional"],
                "symptoms": ["batuk ringan", "suara serak", "tenggorokan tidak nyaman", "paparan polusi ringan"],
                "preparation": "Cuci bahan, rajang tipis kencur, jahe, dan sereh, seduh dengan air mendidih, tutup 10 menit, lalu tambahkan madu dan perasan jeruk saat hangat.",
                "dosage": "Sekitar 250 ml, 1 kali sehari untuk pendamping keluhan ringan.",
                "safety_notes": "Tidak untuk sesak, mengi berat, demam tinggi, batuk darah, atau keluhan saluran napas yang memburuk. Madu tidak untuk bayi <12 bulan.",
                "evidence_level": "medium",
            }
        ],
    ),
    SourceSeed(
        id="kemenkes_aman_minum_jamu",
        url="https://keslan.kemkes.go.id/view_artikel/499/amankah-minum-jamu-setiap-hari",
        expected_title="Amankah Minum Jamu Setiap Hari",
        curated_records=[
            {
                "topic": "keamanan penggunaan jamu",
                "formula_name": "Prinsip Aman Minum Jamu",
                "ingredients": ["jamu higienis", "bahan alami sesuai dosis", "produk terdaftar BPOM bila kemasan"],
                "symptoms": ["edukasi keamanan", "penggunaan herbal harian"],
                "preparation": "Gunakan bahan bersih, pengolahan higienis, tidak berlebihan, dan pilih produk yang teruji bila memakai kemasan.",
                "dosage": "Ikuti dosis case/formula terkurasi atau label produk. Hindari penggunaan berlebihan.",
                "safety_notes": "Jamu tetap dapat memiliki efek samping bila berlebihan. Untuk penyakit tertentu atau obat rutin, konsultasikan dengan tenaga kesehatan.",
                "evidence_level": "safety_guideline",
            },
            {
                "topic": "kurang fit dan daya tahan tubuh",
                "formula_name": "Temulawak-Kunyit-Jahe",
                "ingredients": ["temulawak", "kunyit", "jahe"],
                "symptoms": ["kurang fit", "pemulihan ringan", "daya tahan tubuh menurun"],
                "preparation": "Cuci dan iris rimpang, rebus 10-15 menit, lalu saring.",
                "dosage": "150-200 ml, 1 kali sehari untuk penggunaan pendek.",
                "safety_notes": "Hati-hati pada gangguan lambung, empedu, hati, penggunaan antikoagulan, atau gejala akut berat.",
                "evidence_level": "medium",
            },
        ],
    ),
    SourceSeed(
        id="kemenkes_oht_fitofarmaka",
        url="https://keslan.kemkes.go.id/view_artikel/2154/jamu-obat-herbal-terstandar-dan-fitofarmaka",
        expected_title="Jamu, Obat Herbal Terstandar dan Fitofarmaka",
        curated_records=[
            {
                "topic": "klasifikasi evidence herbal",
                "formula_name": "Klasifikasi Jamu-OHT-Fitofarmaka",
                "ingredients": ["jamu", "obat herbal terstandar", "fitofarmaka"],
                "symptoms": ["edukasi evidence level", "validasi rekomendasi herbal"],
                "preparation": "Gunakan sebagai kerangka penilaian evidence, bukan resep ramuan.",
                "dosage": "Tidak berlaku; dosis mengikuti produk/ramuan terkurasi.",
                "safety_notes": "Klaim jamu tidak boleh dilebih-lebihkan dan perlu dibedakan dari OHT/fitofarmaka yang memiliki tingkat pembuktian lebih tinggi.",
                "evidence_level": "evidence_framework",
            }
        ],
    ),
    SourceSeed(
        id="kemenkes_jambu_biji",
        url="https://keslan.kemkes.go.id/view_artikel/162/yuk-ketahui-manfaat-buah-jambu-biji-untuk-kesehatan-tubuh",
        expected_title="Yuk Ketahui Manfaat Buah Jambu Biji untuk Kesehatan Tubuh",
        curated_records=[
            {
                "topic": "diare ringan dan dukungan pencernaan",
                "formula_name": "Daun/Buah Jambu Biji sebagai konteks pencernaan",
                "ingredients": ["daun jambu biji", "buah jambu biji"],
                "symptoms": ["diare ringan", "gangguan saluran cerna ringan", "pemulihan nutrisi"],
                "preparation": "Untuk rekomendasi praktis, gunakan case rebusan daun jambu biji yang sudah terkurasi. Buah jambu biji dapat menjadi dukungan nutrisi bila cocok.",
                "dosage": "Dosis praktis mengikuti case rebusan daun jambu biji; buah dikonsumsi wajar sebagai makanan.",
                "safety_notes": "Diare berdarah, dehidrasi, demam tinggi, atau anak sangat lemah harus diarahkan ke fasilitas kesehatan.",
                "evidence_level": "medium",
            }
        ],
    ),
    SourceSeed(
        id="kemenkes_obat_diare",
        url="https://keslan.kemkes.go.id/view_artikel/4105/mari-mengenal-obat-diare",
        expected_title="Mari Mengenal Obat Diare",
        curated_records=[
            {
                "topic": "triase diare ringan",
                "formula_name": "Oralit dan Cairan sebagai Prioritas Diare",
                "ingredients": ["oralit", "air minum", "cairan cukup"],
                "symptoms": ["diare ringan", "risiko dehidrasi"],
                "preparation": "Gunakan oralit sesuai aturan kemasan dan cukupkan cairan. Ramuan herbal hanya pendamping gejala ringan.",
                "dosage": "Ikuti aturan oralit; minum sedikit-sedikit tetapi sering sesuai kebutuhan cairan.",
                "safety_notes": "Tanda dehidrasi seperti sangat haus, pusing, urine pekat, mulut kering, mata cekung, atau BAK berkurang perlu evaluasi medis.",
                "evidence_level": "clinical_safety_context",
            }
        ],
    ),
    SourceSeed(
        id="kemenkes_batuk_tradisional",
        url="https://keslan.kemkes.go.id/view_artikel/653/batuk-dengan-pengobatan-tradisonal",
        expected_title="Batuk dengan Pengobatan Tradisional",
        curated_records=[
            {
                "topic": "batuk pilek ringan",
                "formula_name": "Jeruk Nipis-Madu",
                "ingredients": ["jeruk nipis", "madu", "air hangat"],
                "symptoms": ["batuk ringan", "pilek ringan", "tenggorokan tidak nyaman"],
                "preparation": "Campurkan perasan jeruk nipis dan madu ke air hangat.",
                "dosage": "150-200 ml, 1-2 kali sehari untuk keluhan ringan.",
                "safety_notes": "Madu tidak untuk bayi <12 bulan. Hati-hati pada lambung sensitif. Rujuk bila sesak, demam tinggi, atau batuk darah.",
                "evidence_level": "medium",
            },
            {
                "topic": "hidung meler dan sakit tenggorokan ringan",
                "formula_name": "Kunyit-Asam-Gula Jawa",
                "ingredients": ["kunyit", "asam jawa", "gula jawa"],
                "symptoms": ["hidung meler ringan", "sakit tenggorokan ringan", "batuk pilek ringan"],
                "preparation": "Parut kunyit, ambil sari, rebus bersama sedikit gula jawa dan asam jawa, lalu saring.",
                "dosage": "150 ml, 1 kali sehari untuk penggunaan pendek.",
                "safety_notes": "Batasi gula pada diabetes. Hati-hati pada lambung sensitif atau gangguan empedu.",
                "evidence_level": "medium",
            },
        ],
    ),
    SourceSeed(
        id="ayosehat_jamu",
        url="https://ayosehat.kemkes.go.id/sehat-dengan-jamu-ayo-minum-jamu",
        expected_title="Sehat dengan Jamu, Ayo Minum Jamu",
        curated_records=[
            {
                "topic": "pemanfaatan jamu sebagai edukasi promotif",
                "formula_name": "Kunyit Asam",
                "ingredients": ["kunyit", "asam jawa", "air"],
                "symptoms": ["rasa tidak nyaman tubuh ringan", "tenggorokan tidak nyaman ringan", "gangguan pencernaan ringan"],
                "preparation": "Cuci kunyit, iris/parut, rebus bersama asam jawa, lalu saring.",
                "dosage": "150-200 ml, 1 kali sehari atau maksimal 2 kali sehari untuk penggunaan pendek.",
                "safety_notes": "Hati-hati pada lambung sensitif, gangguan empedu, atau penggunaan obat pengencer darah.",
                "evidence_level": "medium",
            }
        ],
    ),
    SourceSeed(
        id="ristoja_maluku",
        url="https://repository.badankebijakan.kemkes.go.id/id/eprint/3082/",
        expected_title="RISTOJA Pengetahuan Lokal Etnomedisin dan Tumbuhan Obat",
        curated_records=[
            {
                "topic": "digitalisasi pengetahuan ramuan tradisional",
                "formula_name": "Record RISTOJA sebagai konteks etnomedisin",
                "ingredients": ["tumbuhan obat lokal", "ramuan tradisional komunitas"],
                "symptoms": ["pengetahuan herbal masyarakat", "basis data ramuan tradisional"],
                "preparation": "Gunakan sebagai konteks pengetahuan dan sumber kurasi, bukan sebagai resep langsung tanpa validasi bahan dan dosis.",
                "dosage": "Tidak digunakan sebagai dosis langsung; perlu kurasi per ramuan.",
                "safety_notes": "Data etnomedisin harus dipetakan ulang terhadap gejala ringan, dosis aman, dan tanda kewaspadaan sebelum menjadi rekomendasi chatbot.",
                "evidence_level": "ethnomedicine_reference",
            }
        ],
    ),
]


EXTRA_CURATED_RECORDS = [
    {
        "id": "curated_serai_kayu_manis_madu",
        "topic": "sakit tenggorokan ringan",
        "formula_name": "Serai-Kayu Manis-Madu",
        "ingredients": ["serai", "kayu manis", "madu"],
        "symptoms": ["tenggorokan tidak nyaman", "batuk ringan", "rasa gatal tenggorokan"],
        "preparation": "Rebus serai dan kayu manis 10-15 menit, tunggu hangat, kemudian tambahkan madu.",
        "dosage": "150-200 ml, 1-2 kali sehari untuk keluhan ringan.",
        "safety_notes": "Madu tidak untuk bayi <12 bulan. Tidak untuk sesak, sulit menelan berat, demam tinggi, atau gejala memburuk.",
        "evidence_level": "low_to_medium",
        "source_title": "Kurasi Kemenkes madu dan literatur serai/kayu manis",
        "source_url": "https://keslan.kemkes.go.id/view_artikel/72/manfaat-madu-bagi-kesehatan",
        "curation_method": "curated_formula_inference",
    },
    {
        "id": "curated_daun_jambu_biji",
        "topic": "diare ringan tanpa darah",
        "formula_name": "Rebusan Daun Jambu Biji",
        "ingredients": ["daun jambu biji", "air"],
        "symptoms": ["diare ringan", "BAB cair tanpa darah", "perut tidak nyaman ringan"],
        "preparation": "Cuci daun jambu biji, rebus dengan air bersih, saring, dan minum saat hangat.",
        "dosage": "150-200 ml, 1-2 kali sehari sebagai pendamping; cairan dan oralit tetap utama.",
        "safety_notes": "Segera rujuk bila ada darah, dehidrasi, demam tinggi, anak sangat lemah, atau diare menetap.",
        "evidence_level": "medium",
        "source_title": "Kurasi referensi daun jambu biji",
        "source_url": "data/referensi/herbal_references.csv",
        "curation_method": "local_reference_curated",
    },
    {
        "id": "curated_sambiloto",
        "topic": "gejala saluran napas atas ringan",
        "formula_name": "Sambiloto sebagai simplisia pendamping",
        "ingredients": ["sambiloto"],
        "symptoms": ["sakit tenggorokan ringan", "demam ringan", "gejala ISPA ringan"],
        "preparation": "Gunakan hanya sesuai produk/simplisia terstandar atau arahan tenaga kesehatan; rasa sangat pahit.",
        "dosage": "Ikuti label produk/simplisia terstandar; tidak dibuat sebagai dosis bebas untuk pengguna awam.",
        "safety_notes": "Tidak untuk ibu hamil tanpa evaluasi medis. Tidak menggantikan terapi infeksi berat.",
        "evidence_level": "high",
        "source_title": "Kurasi referensi sambiloto",
        "source_url": "data/referensi/herbal_references.csv",
        "curation_method": "local_reference_curated",
    },
    {
        "id": "curated_kulit_manggis",
        "topic": "referensi penelitian tanaman herbal Indonesia",
        "formula_name": "Ekstrak Kulit Manggis",
        "ingredients": ["kulit manggis"],
        "symptoms": ["dukungan antioksidan", "edukasi tanaman herbal"],
        "preparation": "Diposisikan sebagai referensi penelitian bahan herbal, bukan resep rebusan bebas untuk keluhan akut.",
        "dosage": "Tidak direkomendasikan sebagai dosis mandiri pada prototype; perlu produk terstandar dan validasi keamanan.",
        "safety_notes": "Tidak menjadi rekomendasi utama chatbot untuk penyakit ringan karena dosis aman rumahan belum dipastikan dalam dataset ini.",
        "evidence_level": "research_reference",
        "source_title": "Kurasi literatur kulit manggis",
        "source_url": "data/referensi/penelitian_toga_dan_ramuan_tradisional.md",
        "curation_method": "research_reference_only",
    },
    {
        "id": "curated_lingzhi",
        "topic": "referensi penelitian bahan herbal",
        "formula_name": "Jamur Lingzhi / Reishi",
        "ingredients": ["Ganoderma lucidum"],
        "symptoms": ["dukungan kebugaran", "edukasi herbal"],
        "preparation": "Diposisikan sebagai referensi penelitian suplemen/herbal, bukan ramuan utama untuk keluhan akut ringan.",
        "dosage": "Tidak direkomendasikan sebagai dosis mandiri pada prototype; perlu produk terstandar dan telaah keamanan.",
        "safety_notes": "Hati-hati pada gangguan hati, penggunaan antikoagulan, dan klaim terapi penyakit berat.",
        "evidence_level": "research_reference",
        "source_title": "Kurasi literatur jamur lingzhi",
        "source_url": "data/referensi/penelitian_toga_dan_ramuan_tradisional.md",
        "curation_method": "research_reference_only",
    },
    {
        "id": "curated_lidah_buaya_topikal",
        "topic": "gatal dan ruam merah ringan",
        "formula_name": "Gel Lidah Buaya Topikal",
        "ingredients": ["lidah buaya", "air bersih"],
        "symptoms": ["gatal ringan", "ruam merah ringan", "iritasi kulit ringan"],
        "preparation": "Cuci area kulit dan tangan, gunakan gel lidah buaya bersih, oles tipis pada area kecil, hindari luka terbuka dan area mata.",
        "dosage": "Oles tipis 1-2 kali sehari maksimal 2-3 hari; hentikan bila perih, makin merah, atau gatal memburuk.",
        "safety_notes": "Tidak untuk ruam dengan demam tinggi, perdarahan, lepuh luas, sesak, bengkak wajah/bibir, atau ruam menyebar cepat. Lakukan uji tempel kecil terlebih dahulu.",
        "evidence_level": "low_to_medium",
        "source_title": "Kurasi literatur lidah buaya dan panduan ruam dengue",
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/?term=aloe+vera+dermatitis+review",
        "curation_method": "local_reference_curated",
    },
]


SFT_VARIANTS = [
    (
        "",
        "full",
        "Saya mengalami {symptoms}. Tolong lakukan anamnesis singkat dan rekomendasikan ramuan herbal bila masih ringan.",
    ),
    ("dosis", "dosis", "Bagaimana cara membuat dan dosis {formula}?"),
    (
        "triase",
        "triage",
        "Keluhan saya {symptoms}. Pertanyaan anamnesis apa yang perlu ditanyakan sebelum menyarankan {formula}?",
    ),
    (
        "keamanan",
        "safety",
        "Apa batas aman, tanda bahaya, dan siapa yang perlu hati-hati sebelum memakai {formula}?",
    ),
    (
        "bukti",
        "evidence",
        "Sumber rujukan apa yang mendukung penggunaan {formula} untuk konteks {symptoms}? Jelaskan dalam Bahasa Indonesia.",
    ),
    (
        "ringkas",
        "brief",
        "Jawab singkat untuk pasien dengan {symptoms}: apakah {formula} boleh dipertimbangkan?",
    ),
    (
        "langkah",
        "dosis",
        "Tuliskan langkah pengolahan {formula} yang higienis dan tidak berlebihan.",
    ),
    (
        "red_flag",
        "safety",
        "Kapan pengguna dengan {symptoms} harus berhenti self-care dan periksa, bukan memakai {formula}?",
    ),
    (
        "follow_up",
        "triage",
        "Buat 3 pertanyaan lanjutan sebelum memberi rekomendasi {formula} untuk {symptoms}.",
    ),
    (
        "paper_id",
        "evidence",
        "Ada referensi paper atau sumber valid untuk {formula}? Terjemahkan ringkasannya ke Bahasa Indonesia.",
    ),
    (
        "kontraindikasi",
        "safety",
        "Sebutkan kontraindikasi atau kondisi yang perlu hati-hati untuk {formula}.",
    ),
    (
        "edukasi",
        "full",
        "Edukasi pasien awam tentang penggunaan {formula} untuk {symptoms} tanpa klaim menyembuhkan.",
    ),
    (
        "higiene",
        "dosis",
        "Bagaimana menjaga kebersihan bahan dan alat saat membuat {formula}?",
    ),
    (
        "anak_hamil",
        "safety",
        "Bagaimana catatan keamanan {formula} untuk anak kecil, ibu hamil, atau pasien dengan obat rutin?",
    ),
    (
        "interaksi",
        "safety",
        "Apakah ada risiko interaksi obat atau efek samping yang perlu ditanyakan sebelum memakai {formula}?",
    ),
    (
        "keluhan_ringan",
        "triage",
        "Bagaimana memastikan {symptoms} masih termasuk keluhan ringan sebelum menyarankan herbal?",
    ),
    (
        "dosis_pendek",
        "dosis",
        "Berapa kisaran penggunaan pendek {formula} dan apa yang harus dilakukan bila tidak membaik?",
    ),
    (
        "sumber_valid",
        "evidence",
        "Jelaskan sumber valid yang dipakai untuk {formula}, termasuk bila sumbernya berbahasa Inggris.",
    ),
    (
        "batas_klaim",
        "evidence",
        "Apa batas klaim yang aman untuk {formula} berdasarkan referensi dataset?",
    ),
    (
        "rekomendasi_aman",
        "full",
        "Susun rekomendasi aman untuk {formula} pada {symptoms}: anamnesis, cara pakai, kewaspadaan, dan disclaimer.",
    ),
    (
        "validasi_gejala",
        "triage",
        "Validasi dulu gejala {symptoms} sebelum menyarankan {formula}. Apa saja yang harus diklarifikasi?",
    ),
    (
        "monitoring",
        "safety",
        "Bagaimana cara memantau respons setelah memakai {formula} untuk {symptoms}?",
    ),
    (
        "alternatif_non_obat",
        "brief",
        "Selain {formula}, edukasi non-obat apa yang aman untuk keluhan {symptoms}?",
    ),
    (
        "batas_durasi",
        "safety",
        "Berapa lama {formula} boleh dicoba untuk {symptoms} sebelum pengguna perlu periksa?",
    ),
    (
        "label_produk",
        "dosis",
        "Jika {formula} memakai produk herbal kemasan, bagaimana membaca aturan pakai dan batas amannya?",
    ),
    (
        "cara_menjawab_pasien",
        "full",
        "Buat jawaban natural untuk pasien yang bertanya tentang {formula} untuk {symptoms}.",
    ),
    (
        "penolakan_redflag",
        "safety",
        "Jika ada tanda bahaya pada {symptoms}, bagaimana menolak rekomendasi {formula} dan mengarahkan pemeriksaan?",
    ),
    (
        "terjemah_referensi",
        "evidence",
        "Terjemahkan inti referensi valid tentang {formula} ke Bahasa Indonesia yang mudah dipahami.",
    ),
    (
        "klaim_tidak_berlebihan",
        "evidence",
        "Bagaimana menjelaskan manfaat {formula} tanpa klaim menyembuhkan penyakit?",
    ),
    (
        "dosis_konservatif",
        "dosis",
        "Susun dosis konservatif {formula} untuk edukasi awal pada {symptoms}.",
    ),
    (
        "kebersihan_penyimpanan",
        "dosis",
        "Bagaimana aturan kebersihan dan penyimpanan singkat saat membuat {formula}?",
    ),
    (
        "risiko_alergi",
        "safety",
        "Apa risiko alergi atau iritasi yang perlu dijelaskan sebelum memakai {formula}?",
    ),
    (
        "pasien_kronis",
        "safety",
        "Apa pesan keamanan {formula} untuk pasien dengan penyakit kronis atau obat rutin?",
    ),
    (
        "format_chatbot",
        "full",
        "Jawab sebagai chatbot anamnesis-first untuk {symptoms} dengan opsi {formula}.",
    ),
    (
        "prioritas_medis",
        "triage",
        "Pada kondisi apa {symptoms} harus diprioritaskan ke tenaga kesehatan daripada memakai {formula}?",
    ),
    (
        "bukti_lokal_internasional",
        "evidence",
        "Bandingkan ringkas rujukan lokal dan paper/internasional untuk {formula}.",
    ),
    (
        "cara_pakai_pendamping",
        "full",
        "Jelaskan {formula} sebagai pendamping self-care untuk {symptoms}, bukan terapi utama.",
    ),
    (
        "edukasi_keluarga",
        "brief",
        "Buat edukasi singkat untuk keluarga pasien tentang penggunaan aman {formula}.",
    ),
    (
        "cek_kelayakan",
        "triage",
        "Buat checklist kelayakan sebelum {formula} direkomendasikan untuk {symptoms}.",
    ),
    (
        "review_keamanan",
        "safety",
        "Review keamanan {formula}: tanda bahaya, efek samping, dan kapan harus berhenti.",
    ),
]


def main() -> int:
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    fetched_sources = fetch_sources(SOURCE_SEEDS)
    records = build_training_records(SOURCE_SEEDS, fetched_sources)
    records.extend(EXTRA_CURATED_RECORDS)
    records.extend(records_from_reference_csv())
    deduped_records = dedupe_records(records)
    sft_rows = build_sft_examples(deduped_records)
    write_jsonl(SCRAPED_SOURCES_PATH, fetched_sources)
    write_jsonl(TRAINING_RECORDS_PATH, deduped_records)
    write_jsonl(SFT_PATH, sft_rows)
    write_combined_sft()
    write_readme(len(fetched_sources), len(deduped_records), len(sft_rows))
    print(f"Wrote {SCRAPED_SOURCES_PATH.relative_to(ROOT)}")
    print(f"Wrote {TRAINING_RECORDS_PATH.relative_to(ROOT)}")
    print(f"Wrote {SFT_PATH.relative_to(ROOT)}")
    print(f"Wrote {COMBINED_SFT_PATH.relative_to(ROOT)}")
    return 0


def fetch_sources(seeds: list[SourceSeed]) -> list[dict[str, object]]:
    return [fetch_source(seed) for seed in seeds]


def fetch_source(seed: SourceSeed) -> dict[str, object]:
    retrieved_at = datetime.now(timezone.utc).isoformat()
    try:
        raw_html = fetch_html(seed.url)
        extractor = extract_html(raw_html)
        return {
            "id": seed.id,
            "source_url": seed.url,
            "title": extractor.title or seed.expected_title,
            "status": "fetched",
            "excerpt": make_excerpt(extractor.text),
            "retrieved_at": retrieved_at,
        }
    except (ssl.SSLCertVerificationError, URLError) as error:
        if not is_ssl_verification_error(error):
            return failed_source(seed, retrieved_at, error)
        try:
            raw_html = fetch_html(seed.url, insecure_ssl=True)
            extractor = extract_html(raw_html)
            return {
                "id": seed.id,
                "source_url": seed.url,
                "title": extractor.title or seed.expected_title,
                "status": "fetched_insecure_ssl_fallback",
                "excerpt": make_excerpt(extractor.text),
                "retrieved_at": retrieved_at,
            }
        except (HTTPError, URLError, TimeoutError, OSError, ssl.SSLError) as error:
            return failed_source(seed, retrieved_at, error)
    except (HTTPError, TimeoutError, OSError) as error:
        return failed_source(seed, retrieved_at, error)


def fetch_html(url: str, insecure_ssl: bool = False) -> str:
    request = Request(url, headers={"User-Agent": "ai-herbal-thesis-research/0.1"})
    context = ssl._create_unverified_context() if insecure_ssl else None
    with urlopen(request, timeout=20, context=context) as response:
        return response.read().decode(response.headers.get_content_charset() or "utf-8", errors="replace")


def extract_html(raw_html: str) -> BasicHTMLTextExtractor:
    extractor = BasicHTMLTextExtractor()
    extractor.feed(raw_html)
    return extractor


def failed_source(seed: SourceSeed, retrieved_at: str, error: Exception) -> dict[str, object]:
    return {
        "id": seed.id,
        "source_url": seed.url,
        "title": seed.expected_title,
        "status": "fetch_failed",
        "error": str(error),
        "excerpt": "",
        "retrieved_at": retrieved_at,
    }


def is_ssl_verification_error(error: Exception) -> bool:
    if isinstance(error, ssl.SSLCertVerificationError):
        return True
    reason = getattr(error, "reason", None)
    return isinstance(reason, ssl.SSLCertVerificationError) or "CERTIFICATE_VERIFY_FAILED" in str(error)


def build_training_records(
    seeds: list[SourceSeed],
    fetched_sources: list[dict[str, object]],
) -> list[dict[str, object]]:
    source_by_id = {source["id"]: source for source in fetched_sources}
    records: list[dict[str, object]] = []
    for seed in seeds:
        source = source_by_id[seed.id]
        for index, record in enumerate(seed.curated_records, start=1):
            records.append(
                {
                    "id": f"{seed.id}_{index:02d}",
                    "source_title": source.get("title") or seed.expected_title,
                    "source_url": seed.url,
                    "curation_method": "scraped_source_with_manual_curation",
                    **record,
                }
            )
    return records


def records_from_reference_csv() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    cases_path = REFERENCE_DIR / "focused_mild_ailment_herbal_dataset.csv"
    formulas_path = REFERENCE_DIR / "herbal_formulas.csv"
    herbs_path = REFERENCE_DIR / "herbal_references.csv"

    if cases_path.exists():
        with cases_path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                records.append(
                    {
                        "id": f"case_{row['id']}",
                        "topic": row["keluhan_ringan"],
                        "formula_name": row["ramuan_rekomendasi"],
                        "ingredients": split_semicolon(row["bahan"]),
                        "symptoms": [row["keluhan_ringan"]],
                        "preparation": row["cara_pengolahan"],
                        "dosage": row["dosis_penggunaan"],
                        "safety_notes": row["catatan_kewaspadaan"],
                        "evidence_level": "curated_case",
                        "source_title": row["sumber_ringkas"],
                        "source_url": reference_urls(
                            row["sumber_ringkas"],
                            "data/referensi/focused_mild_ailment_herbal_dataset.csv",
                        ),
                        "curation_method": "local_case_dataset",
                    }
                )

    if formulas_path.exists():
        with formulas_path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                records.append(
                    {
                        "id": f"formula_{row['id']}",
                        "topic": row["gejala_target"],
                        "formula_name": row["nama_formula"],
                        "ingredients": split_semicolon(row["komposisi"]),
                        "symptoms": split_semicolon(row["gejala_target"]),
                        "preparation": "Mengikuti pengolahan tradisional sesuai bahan dan konteks formula; perlu disajikan ulang oleh chatbot berdasarkan case yang cocok.",
                        "dosage": "Mengikuti dosis case terkurasi atau produk/simplisia terstandar bila tersedia.",
                        "safety_notes": row["catatan_keamanan"],
                        "evidence_level": row["evidence_level"],
                        "source_title": row["sumber_utama"],
                        "source_url": reference_urls(row["sumber_utama"], "data/referensi/herbal_formulas.csv"),
                        "curation_method": "local_formula_dataset",
                    }
                )

    if herbs_path.exists():
        with herbs_path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                records.append(
                    {
                        "id": f"herb_{row['id']}",
                        "topic": row["gejala_target"],
                        "formula_name": row["nama_lokal"],
                        "ingredients": [row["nama_lokal"]],
                        "symptoms": split_semicolon(row["gejala_target"]),
                        "preparation": "Gunakan sebagai konteks bahan herbal; rekomendasi praktis tetap mengikuti formula/case yang memiliki cara pengolahan dan dosis.",
                        "dosage": "Tidak ditentukan pada level bahan tunggal kecuali tersedia pada case/formula terkurasi.",
                        "safety_notes": row["catatan_keamanan"],
                        "evidence_level": row["evidence_level"],
                        "source_title": row["sumber_utama"],
                        "source_url": reference_urls(row["sumber_utama"], "data/referensi/herbal_references.csv"),
                        "curation_method": "local_herb_dataset",
                    }
                )
    return records


def build_sft_examples(records: list[dict[str, object]]) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    for record in records:
        symptoms = ", ".join(as_list(record.get("symptoms"))) or str(record.get("topic", "keluhan ringan"))
        formula = str(record.get("formula_name", "ramuan herbal"))
        for suffix, focus, template in SFT_VARIANTS:
            example_id = f"sft_{record['id']}" if not suffix else f"sft_{suffix}_{record['id']}"
            examples.append(
                {
                    "id": example_id,
                    "source_record_id": record["id"],
                    "variant": suffix or "rekomendasi",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": template.format(symptoms=symptoms, formula=formula)},
                        {"role": "assistant", "content": assistant_answer(record, focus=focus)},
                    ],
                }
            )
    return examples


def assistant_answer(record: dict[str, object], focus: str = "full") -> str:
    ingredients = ", ".join(as_list(record.get("ingredients"))) or "-"
    symptoms = ", ".join(as_list(record.get("symptoms"))) or str(record.get("topic") or "-")
    formula_name = str(record.get("formula_name") or "ramuan herbal")
    source_title = str(record.get("source_title") or "-")
    evidence_level = str(record.get("evidence_level") or "-")
    evidence_summary = translated_evidence_summary(record)
    research_only = str(record.get("curation_method") or "") == "research_reference_only" or evidence_level == "research_reference"

    if research_only:
        recommendation = (
            f"{formula_name} hanya dipakai sebagai referensi penelitian/edukasi dalam dataset ini, "
            "bukan sebagai resep utama untuk self-care."
        )
    else:
        recommendation = (
            f"Untuk konteks {symptoms}, ramuan yang dapat dipertimbangkan adalah {formula_name} "
            "bila keluhan masih ringan dan tidak ada tanda bahaya."
        )

    focus_opening = {
        "dosis": "Fokus pada cara pakai yang konservatif dan higienis.",
        "safety": "Fokus utama adalah keamanan, tanda bahaya, dan kondisi yang perlu hati-hati.",
        "triage": "Mulai dari anamnesis singkat sebelum memberi saran herbal.",
        "evidence": "Gunakan rujukan sebagai batas klaim, bukan sebagai dasar diagnosis.",
        "brief": "Jawaban ringkas tetap harus memuat batasan keamanan.",
    }.get(
        focus,
        "Sebelum rekomendasi, pastikan keluhan masih ringan: tidak ada demam tinggi, sesak, nyeri berat, darah, dehidrasi, kehamilan berisiko, atau gejala memburuk.",
    )

    return (
        f"{focus_opening}\n"
        f"{recommendation}\n"
        f"Pertanyaan anamnesis kunci: sejak kapan keluhan muncul, apakah makin berat, apakah ada tanda bahaya, dan apakah pengguna sedang hamil, anak kecil, punya penyakit kronis, alergi, atau minum obat rutin?\n"
        f"Bahan: {ingredients}.\n"
        f"Cara pengolahan: {as_sentence(record.get('preparation'))}\n"
        f"Dosis/kisaran: {as_sentence(record.get('dosage'))}\n"
        f"Catatan kewaspadaan: {as_sentence(record.get('safety_notes'))}\n"
        f"Ringkasan bukti dalam Bahasa Indonesia: {evidence_summary}\n"
        f"Tingkat bukti dataset: {evidence_label(evidence_level)}.\n"
        f"Sumber/kurasi: {source_title}.\n"
        f"Sumber URL: {record.get('source_url')}.\n"
        "Informasi ini bersifat edukasi dan rekomendasi awal, bukan diagnosis medis final "
        "atau pengganti konsultasi tenaga kesehatan."
    )


def evidence_label(value: str) -> str:
    labels = {
        "high": "tinggi",
        "medium": "sedang",
        "low_to_medium": "rendah sampai sedang",
        "low": "rendah",
        "curated_case": "case terkurasi",
        "clinical_safety_context": "konteks keamanan klinis",
        "safety_guideline": "panduan keamanan",
        "evidence_framework": "kerangka klasifikasi bukti",
        "ethnomedicine_reference": "referensi etnomedisin",
        "research_reference": "referensi penelitian, bukan resep langsung",
    }
    return labels.get(value, value or "-")


def translated_evidence_summary(record: dict[str, object]) -> str:
    source_title = str(record.get("source_title") or "")
    source_url = str(record.get("source_url") or "")
    evidence_level = str(record.get("evidence_level") or "")
    formula_name = str(record.get("formula_name") or "bahan herbal")
    symptoms = ", ".join(as_list(record.get("symptoms"))) or str(record.get("topic") or "keluhan ringan")
    source_reference = f"{source_title} {source_url}"

    if re.search(r"PubMed|NCCIH|WHO|review|springer|frontiers|ncbi", source_reference, flags=re.IGNORECASE):
        return (
            f"Rujukan berbahasa Inggris/paper diterjemahkan sebagai konteks bahwa {formula_name} "
            f"memiliki dukungan untuk {symptoms}, tetapi pemakaian tetap dibatasi pada keluhan ringan, "
            "dosis konservatif, dan tidak menggantikan terapi medis."
        )
    if evidence_level in {"research_reference", "ethnomedicine_reference", "evidence_framework"}:
        return (
            f"{formula_name} disimpan sebagai bahan referensi kurasi. Dataset tidak mengubahnya menjadi klaim terapi langsung "
            "tanpa validasi dosis, keamanan, dan kecocokan keluhan."
        )
    return (
        f"Sumber Indonesia/kurasi lokal digunakan untuk memetakan {formula_name} ke {symptoms}; "
        "klaim dibuat sebagai edukasi awal dengan batasan keamanan."
    )


def reference_urls(source_title: str, fallback: str) -> str:
    urls = [fallback]
    for pmid in re.findall(r"PMID\s*(\d+)", source_title, flags=re.IGNORECASE):
        urls.append(f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/")

    lowered = source_title.lower()
    if "aloe vera dermatitis" in lowered:
        urls.append("https://pubmed.ncbi.nlm.nih.gov/?term=aloe+vera+dermatitis+review")
    if "guava diarrhea" in lowered or "jambu biji" in lowered:
        urls.append("https://www.frontiersin.org/journals/pharmacology/articles/10.3389/fphar.2024.1459066/full")
    if "sambiloto" in lowered or "andrographis" in lowered:
        urls.append("https://www.ncbi.nlm.nih.gov/books/NBK70784/")
    if "madu" in lowered or "honey" in lowered:
        urls.append("https://link.springer.com/article/10.1007/s00431-023-05066-1")

    return "; ".join(dict.fromkeys(urls))


def dedupe_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    deduped: dict[str, dict[str, object]] = {}
    for record in records:
        record_id = str(record.get("id") or f"record_{len(deduped) + 1:04d}")
        record["id"] = slugify(record_id)
        deduped.setdefault(record["id"], record)
    return list(deduped.values())


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_combined_sft() -> None:
    rows: list[dict[str, object]] = []
    for path in sorted(TRAINING_DIR.glob("*_training_sft.jsonl")):
        if path.name == COMBINED_SFT_PATH.name:
            continue
        with path.open(encoding="utf-8") as file:
            rows.extend(json.loads(line) for line in file if line.strip())
    if rows:
        write_jsonl(COMBINED_SFT_PATH, rows)


def write_readme(source_count: int, record_count: int, sft_count: int) -> None:
    herbal_preparation_records = count_jsonl_rows(TRAINING_DIR / "herbal_preparation_training_records.jsonl")
    herbal_preparation_sft = count_jsonl_rows(TRAINING_DIR / "herbal_preparation_training_sft.jsonl")
    medline_records = count_jsonl_rows(TRAINING_DIR / "medlineplus_guidance_training_records.jsonl")
    medline_sft = count_jsonl_rows(TRAINING_DIR / "medlineplus_guidance_training_sft.jsonl")
    tropical_sft = count_jsonl_rows(TRAINING_DIR / "tropical_disease_training_sft.jsonl")
    anamnesis_sft = count_jsonl_rows(TRAINING_DIR / "anamnesis_training_sft.jsonl")
    rag_sft = count_jsonl_rows(TRAINING_DIR / "rag_learning_sft.jsonl")
    combined_sft = count_jsonl_rows(COMBINED_SFT_PATH)
    combined_sft_rag = count_jsonl_rows(TRAINING_DIR / "combined_training_sft_rag.jsonl")
    runtime_records = record_count + herbal_preparation_records + medline_records

    README_PATH.write_text(
        "\n".join(
            [
                "# Data Traning Chatbot Herbal",
                "",
                "Folder ini berisi data training/kurasi untuk prototype chatbot rekomendasi ramuan herbal.",
                "Nama folder mengikuti permintaan proyek (`traning`).",
                "",
                "## File",
                "- `scraped_sources.jsonl`: metadata sumber web dan excerpt pendek hasil scraping. Excerpt dibatasi agar tidak menyalin halaman penuh.",
                "- `herbal_training_records.jsonl`: record terstruktur ramuan, gejala, cara pengolahan, dosis/kisaran, kewaspadaan, dan sumber.",
                "- `herbal_training_sft.jsonl`: contoh percakapan format messages untuk supervised fine-tuning/QLoRA.",
                "- `herbal_preparation_sources.jsonl`: metadata sumber internet untuk detail cara pengolahan ramuan herbal.",
                "- `herbal_preparation_training_records.jsonl`: record tambahan yang memperinci bahan, langkah pengolahan, dosis, dan batas keamanan.",
                "- `herbal_preparation_training_sft.jsonl`: contoh percakapan SFT khusus pertanyaan \"bagaimana cara membuat/mengolah ramuan\".",
                "- `medlineplus_guidance_sources.jsonl`: daftar topik MedlinePlus resmi yang dipilih untuk edukasi gejala, triase, follow-up, dan pencegahan.",
                "- `medlineplus_guidance_training_records.jsonl`: record `disease_guidance` skala besar hasil kurasi dari MedlinePlus XML resmi.",
                "- `medlineplus_guidance_training_sft.jsonl`: contoh percakapan SFT berbasis topik gejala/penyakit MedlinePlus.",
                "- `medlineplus_guidance_manifest.json`: manifest jumlah topik, aturan filter, dan tanggal snapshot XML MedlinePlus.",
                "- `anamnesis_training_sft.jsonl`: contoh percakapan anamnesis bila sudah digenerate dari `program/tools/build_anamnesis_dataset.py`.",
                "- `combined_training_sft.jsonl`: gabungan data herbal dan data anamnesis bila `anamnesis_training_sft.jsonl` sudah tersedia.",
                "- `combined_training_sft_rag.jsonl`: gabungan data SFT utama dengan log learning RAG yang sudah dibersihkan.",
                "",
                "## Catatan Kurasi",
                f"- Jumlah sumber web yang diproses: {source_count}.",
                f"- Jumlah record training herbal dasar: {record_count}.",
                f"- Jumlah contoh SFT herbal: {sft_count}.",
                f"- Jumlah sumber detail pengolahan herbal: {count_jsonl_rows(TRAINING_DIR / 'herbal_preparation_sources.jsonl')}.",
                f"- Jumlah record detail pengolahan herbal: {herbal_preparation_records}.",
                f"- Jumlah contoh SFT detail pengolahan herbal: {herbal_preparation_sft}.",
                f"- Jumlah record disease guidance MedlinePlus: {medline_records}.",
                f"- Jumlah contoh SFT MedlinePlus: {medline_sft}.",
                f"- Jumlah contoh SFT penyakit tropis: {tropical_sft}.",
                f"- Jumlah total runtime training records yang dibaca knowledge base: {runtime_records}.",
                f"- Jumlah contoh SFT anamnesis: {anamnesis_sft}.",
                f"- Jumlah contoh SFT learning log RAG: {rag_sft}.",
                f"- Jumlah contoh SFT gabungan utama: {combined_sft}.",
                f"- Jumlah contoh SFT gabungan + learning log RAG: {combined_sft_rag}.",
                "- Data herbal SFT diperbanyak dengan variasi instruksi terstruktur per record: rekomendasi, dosis, triase, keamanan, red flag, dan ringkasan bukti.",
                "- Referensi berbahasa Inggris/paper diringkas ke Bahasa Indonesia di jawaban SFT dan tetap dibatasi sebagai edukasi awal.",
                "- Data anamnesis diperluas dengan variasi synthetic-guided by curated references dari sumber resmi dan paper pendukung. Data ini bukan hasil diagnosis klinis dan tetap perlu review ahli sebelum produksi.",
                "- Dataset MedlinePlus dibangun dari XML resmi snapshot harian dan diekspansi menjadi lima varian record per topik: overview, triage, follow-up, prevention, dan self-care.",
                "- Data ini digunakan untuk edukasi dan rekomendasi awal keluhan ringan, bukan diagnosis medis final.",
                "- Record dari sumber web tetap dikurasi manual agar tidak mengambil klaim mentah tanpa batasan dosis dan kewaspadaan.",
                "- Untuk produksi, setiap record perlu review tenaga kesehatan/herbalis dan uji keamanan lebih lanjut.",
                "",
                "## Regenerasi",
                "```bash",
                "python3 tools/scrape_herbal_sources.py",
                "python3 tools/build_herbal_preparation_dataset.py",
                "python3 tools/build_medlineplus_guidance_dataset.py",
                "python3 tools/build_anamnesis_dataset.py",
                "python3 training/build_rag_lora_dataset.py",
                "python3 training/prepare_mlx_lora_data.py --input data/traning/combined_training_sft_rag.jsonl --output training/data_mlx/herbal_chat_rag_light_current",
                "```",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def make_excerpt(text: str, max_chars: int = 1200) -> str:
    normalized = normalize_text(text)
    if len(normalized) <= max_chars:
        return normalized
    cut = normalized[:max_chars].rsplit(" ", 1)[0]
    return f"{cut}..."


def split_semicolon(value: str) -> list[str]:
    return [item.strip() for item in (value or "").split(";") if item.strip()]


def as_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return split_semicolon(value)
    return []


def as_sentence(value: object) -> str:
    text = str(value or "-").strip()
    if text.endswith((".", "!", "?")):
        return text
    return f"{text}."


def slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_:-]+", "_", value.strip()).strip("_").lower()


if __name__ == "__main__":
    sys.exit(main())
