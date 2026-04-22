from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ANAMNESIS_DIR = ROOT / "data" / "anamnesis"
TRAINING_DIR = ROOT / "data" / "traning"
QUESTIONS_JSONL = ANAMNESIS_DIR / "anamnesis_questions.jsonl"
QUESTIONS_CSV = ANAMNESIS_DIR / "anamnesis_questions.csv"
SFT_JSONL = ANAMNESIS_DIR / "anamnesis_training_sft.jsonl"
SOURCE_REFERENCES_JSONL = ANAMNESIS_DIR / "anamnesis_reference_sources.jsonl"
TRAINING_SFT_COPY = TRAINING_DIR / "anamnesis_training_sft.jsonl"
COMBINED_SFT = TRAINING_DIR / "combined_training_sft.jsonl"
README_PATH = ANAMNESIS_DIR / "README.md"
TARGET_ANAMNESIS_SFT_ROWS = 1200

SYSTEM_PROMPT = (
    "Anda adalah chatbot edukasi kesehatan berbahasa Indonesia yang melakukan anamnesis awal. "
    "Tugas Anda adalah menggali keluhan, durasi, faktor risiko, dan tanda bahaya. "
    "Anda tidak memberikan diagnosis medis final. Untuk tanda bahaya atau dugaan penyakit serius, "
    "arahkan pengguna ke fasilitas kesehatan."
)


SOURCES = {
    "who_dengue": {
        "title": "WHO Dengue and Severe Dengue Fact Sheet",
        "url": "https://www.who.int/en/news-room/fact-sheets/detail/dengue-and-severe-dengue",
    },
    "kemenkes_dbd": {
        "title": "Tanda dan Gejala Demam Berdarah Dengue",
        "url": "https://keslan.kemkes.go.id/view_artikel/10/tanda-dan-gejala-demam-berdarah-dengue",
    },
    "cdc_dengue": {
        "title": "CDC Symptoms of Dengue and Testing",
        "url": "https://www.cdc.gov/dengue/signs-symptoms/index.html",
    },
    "pubmed_aloe": {
        "title": "Aloe vera dermatitis literature",
        "url": "https://pubmed.ncbi.nlm.nih.gov/?term=aloe+vera+dermatitis+review",
    },
    "cdc_malaria": {
        "title": "CDC Symptoms of Malaria",
        "url": "https://www.cdc.gov/malaria/symptoms/index.html",
    },
    "kemenkes_malaria": {
        "title": "Apa Saja Gejala Malaria?",
        "url": "https://malaria.kemkes.go.id/node/110",
    },
    "kemenkes_malaria_keslan": {
        "title": "Malaria Bisa Dihentikan, Ayo Bertindak Sekarang",
        "url": "https://keslan.kemkes.go.id/view_artikel/4189/malaria-bisa-dihentikan-ayo-bertindak-sekarang",
    },
    "ayosehat_tifoid": {
        "title": "Tipus/Penyakit Tipes",
        "url": "https://ayosehat.kemkes.go.id/topik-penyakit/infeksi-enterik/tipus-penyakit-tipes",
    },
    "ayosehat_chikungunya": {
        "title": "Gejala dan Pencegahan Chikungunya",
        "url": "https://ayosehat.kemkes.go.id/gejala-dan-pencegahan-chikungunya",
    },
    "kemenkes_leptospirosis": {
        "title": "Leptospirosis",
        "url": "https://keslan.kemkes.go.id/view_artikel/1952/leptospirosis",
    },
    "cdc_leptospirosis": {
        "title": "CDC About Leptospirosis",
        "url": "https://www.cdc.gov/leptospirosis/index.html",
    },
    "cdc_chikungunya": {
        "title": "CDC Symptoms, Diagnosis, and Treatment of Chikungunya",
        "url": "https://www.cdc.gov/chikungunya/symptoms-diagnosis-treatment/index.html",
    },
    "cdc_common_cold": {
        "title": "CDC About Common Cold",
        "url": "https://www.cdc.gov/common-cold/about/index.html",
    },
    "cdc_sore_throat": {
        "title": "CDC Sore Throat Basics",
        "url": "https://www.cdc.gov/sore-throat/about/index.html",
    },
    "cdc_food_poisoning": {
        "title": "CDC Food Poisoning Symptoms",
        "url": "https://www.cdc.gov/food-safety/signs-symptoms/",
    },
    "cdc_norovirus": {
        "title": "CDC About Norovirus",
        "url": "https://www.cdc.gov/norovirus/about/index.html",
    },
    "cdc_acute_bronchitis": {
        "title": "CDC Chest Cold (Acute Bronchitis) Basics",
        "url": "https://www.cdc.gov/acute-bronchitis/about/index.html",
    },
    "kemenkes_ispa": {
        "title": "Infeksi Saluran Pernapasan Atas (ISPA)",
        "url": "https://keslan.kemkes.go.id/view_artikel/1792/infeksi-saluran-pernapasan-atas-ispa",
    },
    "kemenkes_batuk": {
        "title": "Batuk dengan Pengobatan Tradisional",
        "url": "https://keslan.kemkes.go.id/view_artikel/653/batuk-dengan-pengobatan-tradisonal",
    },
    "kemenkes_diare": {
        "title": "Mari Mengenal Obat Diare",
        "url": "https://keslan.kemkes.go.id/view_artikel/4105/mari-mengenal-obat-diare",
    },
    "ayosehat_dehidrasi": {
        "title": "Dehidrasi pada Anak",
        "url": "https://ayosehat.kemkes.go.id/topik-penyakit/kesehatan-lainnya/dehidrasi-pada-anak",
    },
}


PAPER_SOURCES = {
    "hindelang_2024_history_chatbot": {
        "title": "Transforming Health Care Through Chatbots for Medical History-Taking and Future Directions: Comprehensive Systematic Review",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11393511/",
        "note": "Paper review untuk prinsip medical history-taking chatbot dan arah evaluasi chatbot kesehatan.",
    },
    "amugongo_2025_rag_healthcare": {
        "title": "Retrieval augmented generation for large language models in healthcare: A systematic review",
        "url": "https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0000877",
        "note": "Paper review untuk prinsip grounding RAG, risiko hallucination, dan kebutuhan evaluasi sistem kesehatan.",
    },
    "pengpid_peltzer_2018_indonesia_tcam": {
        "title": "Utilization of traditional and complementary medicine in Indonesia: Results of a national survey in 2014-15",
        "url": "https://pubmed.ncbi.nlm.nih.gov/30396615/",
        "note": "Paper survei nasional Indonesia untuk konteks penggunaan pengobatan tradisional dan komplementer.",
    },
    "pengpid_peltzer_2019_children_tcam": {
        "title": "Use of traditional medicines and traditional practitioners by children in Indonesia: findings from a national population survey in 2014-2015",
        "url": "https://pubmed.ncbi.nlm.nih.gov/31114218/",
        "note": "Paper survei nasional anak di Indonesia sebagai konteks batas keamanan anak dan rujukan.",
    },
}


PROFILE_VARIANTS = [
    "Saya",
    "Ibu saya",
    "Ayah saya",
    "Anak saya usia sekolah",
    "Adik saya",
    "Pasangan saya",
]

DURATION_VARIANTS = [
    "sejak pagi",
    "sejak kemarin",
    "sudah 2 hari",
    "sekitar 3 hari",
    "baru beberapa jam",
    "hilang timbul sejak minggu ini",
]

SEVERITY_VARIANTS = [
    "masih ringan",
    "cukup mengganggu",
    "kadang membaik lalu muncul lagi",
    "belum terlalu berat",
    "membuat aktivitas kurang nyaman",
]

INTENT_VARIANTS = [
    "ingin tahu pertanyaan apa yang perlu dijawab dulu sebelum memilih ramuan herbal",
    "minta rekomendasi ramuan herbal bila masih aman untuk ditangani mandiri",
    "ingin cek apakah masih termasuk keluhan ringan",
    "ingin tahu tanda bahaya apa yang harus diperhatikan",
    "ingin saran awal yang aman sebelum memutuskan langkah berikutnya",
]

ASSISTANT_OPENINGS = [
    "Baik, saya akan menggali keluhan secara bertahap terlebih dahulu.",
    "Saya perlu memastikan dulu apakah keluhan masih termasuk ringan atau ada tanda bahaya.",
    "Sebelum membahas ramuan herbal, saya akan melakukan anamnesis singkat.",
    "Agar rekomendasi tidak terburu-buru, saya akan menanyakan beberapa hal penting.",
]

ASSISTANT_CLOSINGS = [
    "Setelah pertanyaan ini dijawab, rekomendasi herbal baru boleh dipertimbangkan bila keluhan masih ringan dan tidak ada tanda bahaya.",
    "Jika ada tanda bahaya, langkah yang lebih tepat adalah pemeriksaan tenaga kesehatan; bila tidak ada, sistem dapat melanjutkan ke rekomendasi herbal terkurasi.",
    "Jawaban atas pertanyaan ini akan menentukan apakah chatbot melanjutkan ke rekomendasi ramuan atau menyarankan pemeriksaan medis.",
]


RECORDS = [
    {
        "id": "anam_001",
        "condition_group": "skrining_umum",
        "suspected_condition": "pemisahan keluhan ringan dan tanda bahaya",
        "sample_user_message": "Saya kurang enak badan dan ingin rekomendasi ramuan herbal.",
        "primary_symptoms": ["keluhan umum", "kurang fit", "butuh rekomendasi herbal"],
        "applicable_case_ids": ["case_001", "case_002", "case_003", "case_004", "case_005", "case_006", "case_007", "case_008"],
        "required_questions": [
            "Apa keluhan utama yang paling mengganggu saat ini?",
            "Sejak kapan keluhan muncul dan apakah memburuk?",
            "Apakah ada demam tinggi, sesak napas, nyeri hebat, pingsan, kejang, perdarahan, atau lemas sekali?",
            "Apakah ada muntah terus-menerus, diare berdarah, sulit minum, atau tanda dehidrasi?",
            "Apakah pengguna bayi, lansia, hamil, memiliki penyakit kronis, atau sedang minum obat rutin?",
        ],
        "red_flag_questions": [
            "Apakah ada sesak, penurunan kesadaran, kejang, nyeri hebat, atau perdarahan?",
            "Apakah keluhan terjadi pada bayi, kehamilan, atau pasien dengan penyakit kronis berat?",
        ],
        "triage_action": "Jika semua tanda bahaya negatif dan keluhan ringan, lanjutkan rekomendasi herbal terkurasi. Jika ada tanda bahaya, rujuk ke tenaga kesehatan.",
        "source_keys": ["kemenkes_dbd", "cdc_food_poisoning", "ayosehat_dehidrasi", "kemenkes_batuk", "hindelang_2024_history_chatbot"],
    },
    {
        "id": "anam_002",
        "condition_group": "demam",
        "suspected_condition": "demam akut tidak spesifik",
        "sample_user_message": "Saya demam dan sakit kepala sejak kemarin.",
        "primary_symptoms": ["demam", "sakit kepala", "meriang"],
        "applicable_case_ids": ["case_006"],
        "required_questions": [
            "Berapa suhu tubuh tertinggi dan sejak kapan demam berlangsung?",
            "Apakah demam mendadak tinggi atau naik bertahap?",
            "Apakah ada sakit kepala berat, nyeri belakang mata, nyeri otot/sendi, ruam, atau perdarahan?",
            "Apakah ada menggigil berkeringat berulang atau riwayat tinggal/perjalanan ke daerah malaria?",
            "Apakah ada batuk, pilek, sakit tenggorokan, nyeri perut, diare, mual, atau muntah?",
            "Apakah ada paparan banjir, lumpur, tikus, atau luka terbuka?",
        ],
        "red_flag_questions": [
            "Apakah demam tinggi menetap lebih dari 3 hari atau disertai lemas berat?",
            "Apakah ada perdarahan, nyeri perut hebat, muntah terus, sesak, kejang, atau penurunan kesadaran?",
        ],
        "triage_action": "Demam dengan tanda bahaya atau pola mengarah penyakit infeksi tropis harus dirujuk. Herbal hanya dapat diposisikan sebagai edukasi pendamping gejala ringan.",
        "source_keys": ["who_dengue", "kemenkes_dbd", "cdc_dengue", "cdc_malaria", "kemenkes_malaria", "kemenkes_leptospirosis", "cdc_leptospirosis"],
    },
    {
        "id": "anam_003",
        "condition_group": "penyakit_tropis",
        "suspected_condition": "dugaan dengue atau demam berdarah dengue",
        "sample_user_message": "Saya demam tinggi mendadak, sakit kepala, dan badan pegal.",
        "primary_symptoms": ["demam tinggi mendadak", "sakit kepala", "nyeri otot", "ruam", "perdarahan"],
        "applicable_case_ids": ["case_006"],
        "required_questions": [
            "Apakah demam tinggi muncul mendadak dan sudah berlangsung 2-7 hari?",
            "Apakah ada sakit kepala berat, nyeri belakang mata, nyeri otot/sendi, mual, muntah, atau ruam?",
            "Apakah ada bintik merah, mimisan, gusi berdarah, BAB hitam, atau perdarahan lain?",
            "Apakah ada nyeri perut hebat, muntah terus, sangat lemas/gelisah, tangan-kaki dingin, atau sulit minum?",
            "Apakah ada kasus demam berdarah di rumah, sekolah, kantor, atau lingkungan sekitar?",
        ],
        "red_flag_questions": [
            "Apakah ada nyeri perut hebat, muntah terus-menerus, perdarahan, gelisah/lemas berat, atau tangan-kaki dingin?",
            "Apakah demam turun tetapi kondisi tubuh justru makin lemas atau memburuk?",
        ],
        "triage_action": "Jika jawaban mengarah dengue atau ada warning sign, jangan berikan herbal sebagai terapi utama. Arahkan pemeriksaan medis dan hidrasi aman.",
        "source_keys": ["who_dengue", "kemenkes_dbd", "cdc_dengue"],
    },
    {
        "id": "anam_004",
        "condition_group": "penyakit_tropis",
        "suspected_condition": "dugaan malaria",
        "sample_user_message": "Saya demam menggigil lalu berkeringat banyak dan sakit kepala.",
        "primary_symptoms": ["demam", "menggigil", "berkeringat", "sakit kepala", "mual"],
        "applicable_case_ids": ["case_006", "case_001"],
        "required_questions": [
            "Apakah demam disertai menggigil lalu berkeringat banyak?",
            "Apakah pola demam berulang atau hilang-timbul dalam beberapa hari?",
            "Apakah baru tinggal atau bepergian ke daerah endemis malaria?",
            "Apakah ada sakit kepala, lemas, pegal, mual, muntah, nyeri perut, atau diare?",
            "Apakah ada pucat, kuning pada mata/kulit, kebingungan, kejang, atau sangat lemah?",
        ],
        "red_flag_questions": [
            "Apakah ada penurunan kesadaran, kejang, kuning, sesak, atau lemas berat?",
            "Apakah pengguna sedang hamil atau anak kecil dengan demam menggigil?",
        ],
        "triage_action": "Demam menggigil berkeringat dengan risiko daerah malaria perlu pemeriksaan medis/laboratorium. Herbal tidak menggantikan obat antimalaria.",
        "source_keys": ["cdc_malaria", "kemenkes_malaria", "kemenkes_malaria_keslan"],
    },
    {
        "id": "anam_005",
        "condition_group": "penyakit_tropis",
        "suspected_condition": "dugaan tifoid atau tipes",
        "sample_user_message": "Saya demam beberapa hari, perut tidak nyaman, dan nafsu makan turun.",
        "primary_symptoms": ["demam bertahap", "nyeri perut", "diare atau konstipasi", "nafsu makan menurun"],
        "applicable_case_ids": ["case_001", "case_003", "case_005"],
        "required_questions": [
            "Sudah berapa hari demam dan apakah demam cenderung naik bertahap?",
            "Apakah ada nyeri perut, kembung, mual, muntah, diare, atau sulit BAB?",
            "Apakah nafsu makan turun, badan sangat lemas, atau sakit kepala?",
            "Apakah ada riwayat makanan/minuman kurang higienis atau kontak dengan orang bergejala serupa?",
            "Apakah ada BAB berdarah, nyeri perut hebat, muntah terus, atau sangat lemas?",
        ],
        "red_flag_questions": [
            "Apakah demam lebih dari 3 hari disertai lemas berat atau nyeri perut hebat?",
            "Apakah ada BAB berdarah, muntah terus, atau tanda dehidrasi?",
        ],
        "triage_action": "Dugaan tifoid membutuhkan evaluasi medis. Rekomendasi herbal hanya boleh untuk pendamping gejala ringan bila tidak ada tanda bahaya.",
        "source_keys": ["ayosehat_tifoid", "kemenkes_diare"],
    },
    {
        "id": "anam_006",
        "condition_group": "penyakit_tropis",
        "suspected_condition": "dugaan chikungunya",
        "sample_user_message": "Saya demam dan sendi terasa sangat nyeri.",
        "primary_symptoms": ["demam mendadak", "nyeri sendi", "ruam", "nyeri otot"],
        "applicable_case_ids": ["case_004", "case_006"],
        "required_questions": [
            "Apakah demam muncul mendadak dan disertai nyeri sendi yang kuat?",
            "Sendi mana yang terasa nyeri dan apakah mengganggu berjalan atau aktivitas?",
            "Apakah ada ruam, sakit kepala, nyeri otot, mual, atau lemas?",
            "Apakah ada banyak nyamuk atau kasus serupa di lingkungan sekitar?",
            "Apakah pengguna bayi, lansia, hamil, atau memiliki penyakit kronis?",
        ],
        "red_flag_questions": [
            "Apakah nyeri sendi sangat berat sampai sulit bergerak?",
            "Apakah ada demam tinggi menetap, perdarahan, sesak, atau penurunan kesadaran?",
        ],
        "triage_action": "Dugaan chikungunya atau nyeri sendi berat perlu evaluasi medis. Herbal hanya sebagai edukasi pendamping kenyamanan ringan.",
        "source_keys": ["ayosehat_chikungunya", "cdc_chikungunya", "kemenkes_dbd", "cdc_dengue"],
    },
    {
        "id": "anam_007",
        "condition_group": "penyakit_tropis",
        "suspected_condition": "dugaan leptospirosis",
        "sample_user_message": "Saya demam setelah bersih-bersih banjir dan betis terasa sakit.",
        "primary_symptoms": ["demam", "nyeri betis", "mata merah", "paparan banjir", "urin gelap"],
        "applicable_case_ids": ["case_004", "case_006"],
        "required_questions": [
            "Apakah sebelum sakit ada paparan banjir, lumpur, selokan, sawah, atau air yang mungkin tercemar urine tikus?",
            "Apakah ada luka terbuka saat terpapar air/lumpur tersebut?",
            "Apakah demam disertai nyeri betis, nyeri punggung bawah, sakit kepala, atau mata merah?",
            "Apakah ada mual, muntah, diare, kulit/mata kuning, urin gelap, atau BAK berkurang?",
            "Apakah ada sesak, perdarahan, sangat lemas, atau penurunan kesadaran?",
        ],
        "red_flag_questions": [
            "Apakah ada mata/kulit kuning, BAK berkurang, sesak, perdarahan, atau lemas berat?",
            "Apakah ada riwayat banjir/lumpur dan demam dengan nyeri betis kuat?",
        ],
        "triage_action": "Dugaan leptospirosis perlu pemeriksaan medis cepat. Jangan menunda dengan herbal.",
        "source_keys": ["kemenkes_leptospirosis", "cdc_leptospirosis"],
    },
    {
        "id": "anam_008",
        "condition_group": "saluran_napas_atas",
        "suspected_condition": "ISPA ringan, batuk pilek, atau sakit tenggorokan ringan",
        "sample_user_message": "Saya batuk ringan, pilek, dan tenggorokan tidak nyaman.",
        "primary_symptoms": ["batuk", "pilek", "sakit tenggorokan", "suara serak"],
        "applicable_case_ids": ["case_002", "case_007", "case_008"],
        "required_questions": [
            "Sejak kapan batuk, pilek, atau sakit tenggorokan muncul?",
            "Apakah ada demam tinggi, sesak napas, mengi, nyeri dada, atau bibir kebiruan?",
            "Apakah batuk berdarah atau dahak banyak berwarna kuning kehijauan?",
            "Apakah sulit menelan berat atau nyeri tenggorokan sangat berat?",
            "Apakah ada paparan asap, polusi, debu, atau orang sakit di sekitar?",
        ],
        "red_flag_questions": [
            "Apakah ada sesak napas, nyeri dada, bibir kebiruan, batuk darah, atau demam tinggi?",
            "Apakah keluhan memburuk atau berlangsung lama?",
        ],
        "triage_action": "Jika ringan tanpa tanda bahaya, chatbot dapat merekomendasikan ramuan tenggorokan/batuk ringan. Jika ada tanda bahaya, rujuk.",
        "source_keys": ["kemenkes_ispa", "kemenkes_batuk", "cdc_common_cold", "cdc_sore_throat", "cdc_acute_bronchitis"],
    },
    {
        "id": "anam_009",
        "condition_group": "pencernaan",
        "suspected_condition": "diare ringan dan risiko dehidrasi",
        "sample_user_message": "Saya diare ringan sejak pagi.",
        "primary_symptoms": ["diare", "BAB cair", "mulas", "dehidrasi"],
        "applicable_case_ids": ["case_005"],
        "required_questions": [
            "Berapa kali BAB cair dalam 24 jam terakhir?",
            "Apakah ada darah atau lendir pada BAB?",
            "Apakah ada demam tinggi, muntah terus, nyeri perut hebat, atau perut kembung berat?",
            "Apakah masih bisa minum dan BAK masih normal?",
            "Apakah ada tanda dehidrasi seperti sangat haus, mulut kering, pusing, mata cekung, atau BAK berkurang?",
        ],
        "red_flag_questions": [
            "Apakah ada darah pada BAB, dehidrasi, demam tinggi, atau muntah terus?",
            "Apakah diare terjadi pada anak kecil dengan lemas atau sulit minum?",
        ],
        "triage_action": "Diare ringan tanpa darah dapat diberi edukasi cairan/oralit dan pendamping herbal terkurasi. Tanda dehidrasi atau darah harus dirujuk.",
        "source_keys": ["kemenkes_diare", "ayosehat_dehidrasi", "cdc_food_poisoning", "cdc_norovirus"],
    },
    {
        "id": "anam_010",
        "condition_group": "pencernaan",
        "suspected_condition": "mual atau muntah ringan",
        "sample_user_message": "Saya mual ringan sejak tadi pagi.",
        "primary_symptoms": ["mual", "muntah ringan", "perut tidak nyaman"],
        "applicable_case_ids": ["case_001"],
        "required_questions": [
            "Sejak kapan mual muncul dan apakah memburuk?",
            "Apakah muntah terjadi berulang sampai sulit minum?",
            "Apakah ada nyeri perut hebat, demam tinggi, diare berdarah, atau muntah darah?",
            "Apakah ada tanda dehidrasi seperti mulut kering, pusing, BAK berkurang, atau sangat lemas?",
            "Apakah sedang hamil, memakai obat tertentu, atau ada kemungkinan keracunan makanan?",
        ],
        "red_flag_questions": [
            "Apakah muntah terus-menerus atau tidak bisa minum?",
            "Apakah ada nyeri perut hebat, muntah darah, demam tinggi, atau dehidrasi?",
        ],
        "triage_action": "Mual ringan tanpa tanda bahaya dapat diarahkan ke ramuan jahe/jahe-madu. Red flag harus dirujuk.",
        "source_keys": ["kemenkes_diare", "ayosehat_dehidrasi", "cdc_food_poisoning", "cdc_norovirus"],
    },
    {
        "id": "anam_011",
        "condition_group": "pemulihan",
        "suspected_condition": "nafsu makan menurun atau lemas ringan",
        "sample_user_message": "Nafsu makan saya menurun setelah sakit ringan.",
        "primary_symptoms": ["nafsu makan menurun", "lemas ringan", "pemulihan"],
        "applicable_case_ids": ["case_003", "case_004"],
        "required_questions": [
            "Sejak kapan nafsu makan menurun dan apakah terjadi setelah sakit ringan?",
            "Apakah ada demam lama, berat badan turun drastis, muntah, diare, atau nyeri perut?",
            "Apakah masih bisa minum dan beraktivitas ringan?",
            "Apakah ada sariawan, sakit tenggorokan, mual, atau gangguan pencernaan?",
            "Apakah terjadi pada anak kecil, lansia, hamil, atau pasien penyakit kronis?",
        ],
        "red_flag_questions": [
            "Apakah ada berat badan turun drastis, demam lama, muntah terus, atau sangat lemah?",
            "Apakah anak tidak mau minum atau tampak lesu berat?",
        ],
        "triage_action": "Jika ringan, dapat direkomendasikan ramuan pemulihan seperti temulawak-kencur-asem. Bila menetap atau berat, rujuk.",
        "source_keys": ["ayosehat_tifoid", "kemenkes_diare"],
    },
    {
        "id": "anam_012",
        "condition_group": "nyeri_tubuh",
        "suspected_condition": "pegal, meriang, atau nyeri badan ringan",
        "sample_user_message": "Badan saya pegal dan meriang ringan.",
        "primary_symptoms": ["pegal", "meriang", "nyeri badan", "kurang fit"],
        "applicable_case_ids": ["case_004", "case_006"],
        "required_questions": [
            "Sejak kapan pegal atau meriang muncul?",
            "Apakah ada demam tinggi, sakit kepala berat, nyeri belakang mata, ruam, atau perdarahan?",
            "Apakah ada nyeri sendi berat, sulit bergerak, atau bengkak sendi?",
            "Apakah ada nyeri betis kuat setelah paparan banjir/lumpur?",
            "Apakah keluhan muncul setelah aktivitas berat, kurang tidur, atau pemulihan sakit ringan?",
        ],
        "red_flag_questions": [
            "Apakah ada demam tinggi, perdarahan, ruam, nyeri kepala berat, nyeri sendi berat, atau nyeri betis setelah banjir?",
            "Apakah keluhan memburuk atau mengganggu aktivitas berat?",
        ],
        "triage_action": "Jika hanya pegal ringan/kurang fit tanpa red flag, rekomendasi herbal pemulihan dapat diberikan. Jika mengarah dengue, chikungunya, atau leptospirosis, rujuk.",
        "source_keys": ["kemenkes_dbd", "cdc_dengue", "ayosehat_chikungunya", "cdc_chikungunya", "kemenkes_leptospirosis", "cdc_leptospirosis"],
    },
    {
        "id": "anam_013",
        "condition_group": "kulit",
        "suspected_condition": "gatal dan ruam merah ringan vs ruam dengan tanda bahaya",
        "sample_user_message": "Saya gatal-gatal dan timbul ruam merah.",
        "primary_symptoms": ["gatal", "ruam merah", "biduran", "bentol", "alergi kulit"],
        "applicable_case_ids": ["case_009"],
        "required_questions": [
            "Sejak kapan gatal dan ruam merah muncul?",
            "Apakah ruam disertai demam? Jika iya, apakah demamnya tinggi atau menetap?",
            "Apakah ada sakit kepala berat, nyeri belakang mata, nyeri otot/sendi, mual, muntah, atau lemas berat?",
            "Apakah ada bintik merah seperti perdarahan, mimisan, gusi berdarah, atau BAB hitam?",
            "Apakah ada riwayat makanan, obat, gigitan serangga, produk kulit baru, atau paparan tertentu sebelum ruam muncul?",
            "Apakah ruam menyebar cepat, terasa panas/nyeri, melepuh, bernanah, atau mengenai mata/mulut?",
        ],
        "red_flag_questions": [
            "Apakah ada demam tinggi, perdarahan, ruam seperti bintik darah, nyeri perut hebat, muntah terus, atau sangat lemas?",
            "Apakah ada sesak napas, bengkak pada wajah/bibir/lidah, atau tenggorokan terasa menyempit?",
            "Apakah ruam melepuh luas, kulit mengelupas, bernanah, atau menyebar cepat?",
        ],
        "triage_action": "Ruam/gatal tanpa tanda bahaya dapat ditangani sebagai keluhan kulit ringan dan dapat diberi edukasi topikal terbatas. Ruam dengan demam tinggi, perdarahan, sesak/bengkak wajah, lepuh luas, atau penyebaran cepat harus dirujuk.",
        "source_keys": ["kemenkes_dbd", "cdc_dengue", "pubmed_aloe"],
    },
]


def main() -> int:
    ANAMNESIS_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    records = [expand_sources(record) for record in RECORDS]
    write_jsonl(QUESTIONS_JSONL, records)
    write_csv(QUESTIONS_CSV, records)
    sft_records = build_sft(records)
    write_jsonl(SFT_JSONL, sft_records)
    write_jsonl(TRAINING_SFT_COPY, sft_records)
    write_reference_sources(records)
    write_combined_sft(sft_records)
    write_readme(records, sft_records)
    print(f"Wrote {QUESTIONS_JSONL.relative_to(ROOT)}")
    print(f"Wrote {QUESTIONS_CSV.relative_to(ROOT)}")
    print(f"Wrote {SFT_JSONL.relative_to(ROOT)}")
    print(f"Wrote {SOURCE_REFERENCES_JSONL.relative_to(ROOT)}")
    print(f"Wrote {COMBINED_SFT.relative_to(ROOT)}")
    return 0


def expand_sources(record: dict[str, object]) -> dict[str, object]:
    sources = [source_by_key(key) for key in record["source_keys"]]
    expanded = {key: value for key, value in record.items() if key != "source_keys"}
    expanded["source_titles"] = [source["title"] for source in sources]
    expanded["source_urls"] = [source["url"] for source in sources]
    expanded["curation_method"] = "official_source_guided_manual_curation"
    expanded["disclaimer"] = "Anamnesis ini untuk deteksi awal dan triase edukatif, bukan diagnosis medis final."
    return expanded


def source_by_key(key: str) -> dict[str, str]:
    source = SOURCES.get(key) or PAPER_SOURCES.get(key)
    if source is None:
        raise KeyError(f"Sumber anamnesis tidak ditemukan: {key}")
    return source


def build_sft(records: list[dict[str, object]]) -> list[dict[str, object]]:
    sft_records: list[dict[str, object]] = []
    for record in records:
        sft_records.append(
            {
                "id": f"sft_{record['id']}",
                "source_record_id": record["id"],
                "condition_group": record["condition_group"],
                "variant_type": "base_record",
                "source_titles": record["source_titles"],
                "source_urls": record["source_urls"],
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": record["sample_user_message"]},
                    {"role": "assistant", "content": build_assistant_content(record, variant_index=0)},
                ],
            }
        )

    variants_needed = max(0, TARGET_ANAMNESIS_SFT_ROWS - len(sft_records))
    variants_per_record = max(1, (variants_needed + len(records) - 1) // len(records))
    variant_index = 1
    for record in records:
        for record_variant_index, (profile, duration, severity, intent) in enumerate(variant_combinations(), start=1):
            if len(sft_records) >= TARGET_ANAMNESIS_SFT_ROWS or record_variant_index > variants_per_record:
                break
            variant_id = f"sft_{record['id']}_var_{record_variant_index:04d}"
            sft_records.append(
                {
                    "id": variant_id,
                    "source_record_id": record["id"],
                    "condition_group": record["condition_group"],
                    "variant_type": "synthetic_anamnesis_variant",
                    "source_titles": record["source_titles"],
                    "source_urls": record["source_urls"],
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": build_user_message(record, profile, duration, severity, intent, variant_index),
                        },
                        {
                            "role": "assistant",
                            "content": build_assistant_content(record, variant_index=variant_index),
                        },
                    ],
                }
            )
            variant_index += 1
    return sft_records


def variant_combinations():
    for profile in PROFILE_VARIANTS:
        for duration in DURATION_VARIANTS:
            for severity in SEVERITY_VARIANTS:
                for intent in INTENT_VARIANTS:
                    yield profile, duration, severity, intent


def build_user_message(
    record: dict[str, object],
    profile: str,
    duration: str,
    severity: str,
    intent: str,
    variant_index: int,
) -> str:
    symptoms = list(record["primary_symptoms"])
    first_symptom = symptoms[variant_index % len(symptoms)] if symptoms else str(record["suspected_condition"])
    second_symptom = symptoms[(variant_index + 1) % len(symptoms)] if len(symptoms) > 1 else ""
    connector = f" dan {second_symptom}" if second_symptom and second_symptom != first_symptom else ""
    herbal_tail = " Saya ingin ramuan herbal, tapi tolong tanya dulu gejala yang perlu dipastikan."
    return f"{profile} mengalami {first_symptom}{connector} {duration}, rasanya {severity}. Saya {intent}.{herbal_tail}"


def build_assistant_content(record: dict[str, object], variant_index: int) -> str:
    required_questions = list(record["required_questions"])
    red_flag_questions = list(record["red_flag_questions"])
    questions = rotate(required_questions, variant_index)
    red_flags = rotate(red_flag_questions, variant_index)
    opening = ASSISTANT_OPENINGS[variant_index % len(ASSISTANT_OPENINGS)]
    closing = ASSISTANT_CLOSINGS[variant_index % len(ASSISTANT_CLOSINGS)]
    question_text = "\n".join(f"- {question}" for question in questions[:6])
    red_flag_text = "\n".join(f"- {question}" for question in red_flags[:3])
    source_hint = "; ".join(record["source_titles"][:3])
    return (
        f"{opening} Fokus anamnesis awal adalah kemungkinan {record['suspected_condition']}, "
        "tetapi hasil ini bukan diagnosis medis final.\n\n"
        f"Pertanyaan anamnesis utama:\n{question_text}\n\n"
        f"Pertanyaan tanda bahaya:\n{red_flag_text}\n\n"
        f"Keputusan awal: {record['triage_action']}\n\n"
        f"{closing}\n\n"
        f"Sumber kurasi ringkas: {source_hint}."
    )


def rotate(values: list[str], offset: int) -> list[str]:
    if not values:
        return []
    shift = offset % len(values)
    return values[shift:] + values[:shift]


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "id",
            "condition_group",
            "suspected_condition",
            "primary_symptoms",
            "applicable_case_ids",
            "required_questions",
            "red_flag_questions",
            "triage_action",
            "source_titles",
            "source_urls",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: flatten(row.get(field)) for field in fieldnames})


def write_combined_sft(anamnesis_sft: list[dict[str, object]]) -> None:
    combined: list[dict[str, object]] = []
    herbal_sft = TRAINING_DIR / "herbal_training_sft.jsonl"
    if herbal_sft.exists():
        with herbal_sft.open(encoding="utf-8") as file:
            combined.extend(json.loads(line) for line in file if line.strip())
    combined.extend(anamnesis_sft)
    write_jsonl(COMBINED_SFT, combined)


def write_reference_sources(records: list[dict[str, object]]) -> None:
    rows: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for source in [*SOURCES.values(), *PAPER_SOURCES.values()]:
        if source["url"] in seen_urls:
            continue
        seen_urls.add(source["url"])
        rows.append(
            {
                "title": source["title"],
                "url": source["url"],
                "note": source.get("note", "Sumber gejala, tanda bahaya, atau konteks metode anamnesis."),
            }
        )
    for record in records:
        for title, url in zip(record["source_titles"], record["source_urls"], strict=False):
            if url in seen_urls:
                continue
            seen_urls.add(url)
            rows.append({"title": title, "url": url, "note": "Sumber tambahan pada record anamnesis."})
    write_jsonl(SOURCE_REFERENCES_JSONL, rows)


def write_readme(records: list[dict[str, object]], sft_records: list[dict[str, object]]) -> None:
    source_lines = []
    seen_urls: set[str] = set()
    for record in records:
        for title, url in zip(record["source_titles"], record["source_urls"], strict=False):
            if url in seen_urls:
                continue
            seen_urls.add(url)
            source_lines.append(f"- [{title}]({url})")

    README_PATH.write_text(
        "\n".join(
            [
                "# Dataset Anamnesis Chatbot Herbal",
                "",
                "Folder ini berisi pertanyaan anamnesis terstruktur untuk membantu chatbot menggali keluhan, durasi, faktor risiko, dan tanda bahaya sebelum memberi rekomendasi ramuan herbal.",
                "",
                "## File",
                "- `anamnesis_questions.jsonl`: dataset utama berisi kelompok kondisi, gejala, pertanyaan wajib, pertanyaan red flag, tindakan triase, dan sumber.",
                "- `anamnesis_questions.csv`: versi spreadsheet dari dataset utama.",
                "- `anamnesis_training_sft.jsonl`: contoh percakapan format messages untuk fine-tuning/SFT.",
                "- `anamnesis_reference_sources.jsonl`: daftar sumber resmi dan paper yang digunakan untuk kurasi dataset anamnesis.",
                "",
                "## Integrasi Training",
                "- Salinan `anamnesis_training_sft.jsonl` juga dibuat di `../traning/anamnesis_training_sft.jsonl`.",
                "- File gabungan QLoRA dibuat di `../traning/combined_training_sft.jsonl`.",
                "",
                "## Batasan",
                "- Dataset ini untuk deteksi awal dan triase edukatif, bukan diagnosis medis final.",
                "- Jika muncul tanda bahaya, chatbot harus mengarahkan pengguna ke tenaga kesehatan/fasilitas kesehatan.",
                "- Rekomendasi herbal hanya boleh diberikan untuk keluhan ringan yang sesuai batas sistem.",
                "",
                "## Ringkasan",
                f"- Jumlah record anamnesis: {len(records)}.",
                f"- Jumlah contoh SFT anamnesis: {len(sft_records)}.",
                f"- Target minimum contoh SFT anamnesis: {TARGET_ANAMNESIS_SFT_ROWS}.",
                "- Contoh SFT anamnesis diperluas menggunakan variasi terstruktur atas profil pengguna, durasi, tingkat keluhan, intent, dan urutan pertanyaan. Data ini bersifat synthetic-guided by curated references, sehingga tetap perlu review ahli sebelum produksi.",
                "",
                "## Sumber Utama",
                *source_lines,
                "",
                "## Paper Pendukung",
                *[f"- [{source['title']}]({source['url']})" for source in PAPER_SOURCES.values()],
                "",
                "## Regenerasi",
                "```bash",
                "python3 program/tools/build_anamnesis_dataset.py",
                "```",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def flatten(value: object) -> str:
    if isinstance(value, list):
        return " | ".join(flatten(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return "" if value is None else str(value)


if __name__ == "__main__":
    raise SystemExit(main())
