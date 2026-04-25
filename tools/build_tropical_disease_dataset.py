from __future__ import annotations

import json
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
DOCUMENT_DIR = ROOT / "document" / "disease_tropic"
TRAINING_DIR = ROOT / "data" / "traning"
SOURCES_PATH = TRAINING_DIR / "tropical_disease_sources.jsonl"
RECORDS_PATH = TRAINING_DIR / "tropical_disease_training_records.jsonl"
SFT_PATH = TRAINING_DIR / "tropical_disease_training_sft.jsonl"
MANIFEST_PATH = TRAINING_DIR / "tropical_disease_manifest.json"
COMBINED_SFT_PATH = TRAINING_DIR / "combined_training_sft.jsonl"

TRIAGE_SYSTEM_PROMPT = (
    "Anda adalah chatbot edukasi kesehatan berbahasa Indonesia. "
    "Untuk gejala yang mengarah ke penyakit tropis, penyakit menular kronis, atau infeksi kulit/jamur yang tidak ringan, "
    "tugas Anda adalah menggali gejala penting, menyebut tanda bahaya, memberi edukasi pencegahan singkat, "
    "dan menegaskan bahwa herbal bukan terapi utama. Jangan memberi diagnosis medis final."
)
PREVENTION_SYSTEM_PROMPT = (
    "Anda adalah chatbot edukasi kesehatan berbahasa Indonesia yang menjelaskan pencegahan penyakit dengan singkat, jelas, dan aman. "
    "Jangan mengarang terapi herbal untuk penyakit serius. Bila perlu, arahkan ke pemeriksaan medis."
)


@dataclass(frozen=True)
class DiseaseDocSpec:
    id: str
    filename: str
    disease_name: str
    topic: str
    primary_symptoms: list[str]
    warning_signs: list[str]
    screening_questions: list[str]
    care_recommendation: str
    sample_user_message: str
    prevention_question: str
    symptom_headings: tuple[str, ...]
    diagnosis_headings: tuple[str, ...]
    prevention_headings: tuple[str, ...]


SPECS = [
    DiseaseDocSpec(
        id="tropic_dbd",
        filename="01_dbd.docx",
        disease_name="Demam Berdarah Dengue (DBD)",
        topic="dugaan demam berdarah dengue (dbd)",
        primary_symptoms=["demam tinggi mendadak", "sakit kepala berat", "nyeri belakang mata", "nyeri otot atau sendi", "mual atau muntah", "ruam"],
        warning_signs=["nyeri perut berat", "muntah terus-menerus", "gusi berdarah atau mimisan", "muntah darah atau BAB berdarah", "lemas setelah demam turun", "kulit dingin pucat dan napas cepat"],
        screening_questions=[
            "Apakah demam tinggi muncul mendadak dan sudah berlangsung 2-7 hari?",
            "Apakah ada sakit kepala berat, nyeri belakang mata, nyeri otot/sendi, ruam, mual, atau muntah?",
            "Apakah ada mimisan, gusi berdarah, muntah darah, atau BAB berdarah/hitam?",
            "Apakah ada nyeri perut hebat, muntah terus, tangan-kaki dingin, gelisah, atau sangat lemas?",
        ],
        care_recommendation="Jika keluhan mengarah ke DBD atau ada warning sign, prioritaskan pemeriksaan medis dan hidrasi aman. Jangan memposisikan herbal sebagai terapi utama.",
        sample_user_message="Saya demam tinggi mendadak, sakit kepala, badan pegal, dan khawatir apakah ini DBD.",
        prevention_question="Bagaimana cara mencegah DBD dan tanda bahaya apa yang harus diwaspadai?",
        symptom_headings=("gejala demam berdarah", "gejala dbd"),
        diagnosis_headings=("diagnosis demam berdarah",),
        prevention_headings=("pencegahan demam berdarah",),
    ),
    DiseaseDocSpec(
        id="tropic_malaria",
        filename="02_Malaria.docx",
        disease_name="Malaria",
        topic="dugaan malaria",
        primary_symptoms=["demam", "menggigil", "berkeringat banyak", "sakit kepala", "lemas", "mual atau muntah"],
        warning_signs=["kebingungan atau kejang", "kuning pada mata atau kulit", "sesak napas", "anemia berat", "lemas berat", "demam berulang setelah bepergian ke daerah endemis"],
        screening_questions=[
            "Apakah demam disertai menggigil lalu berkeringat banyak?",
            "Apakah pola demam berulang atau hilang-timbul dalam beberapa hari?",
            "Apakah baru tinggal atau bepergian ke daerah endemis malaria?",
            "Apakah ada pucat, kuning pada mata/kulit, kebingungan, kejang, atau sangat lemah?",
        ],
        care_recommendation="Demam yang mengarah malaria perlu pemeriksaan medis/laboratorium dan obat antimalaria yang tepat. Herbal tidak menggantikan terapi utama.",
        sample_user_message="Saya demam menggigil lalu berkeringat banyak, sakit kepala, dan baru dari daerah endemis malaria.",
        prevention_question="Bagaimana cara mencegah malaria saat tinggal atau bepergian ke daerah endemis?",
        symptom_headings=("gejala malaria", "penyebab dan gejala malaria"),
        diagnosis_headings=("diagnosis malaria",),
        prevention_headings=("pencegahan malaria",),
    ),
    DiseaseDocSpec(
        id="tropic_tbc",
        filename="Tuberkulosis:TBC.docx",
        disease_name="Tuberkulosis (TBC)",
        topic="dugaan tuberkulosis tbc",
        primary_symptoms=["batuk lebih dari 3 minggu", "dahak atau batuk darah", "demam", "nyeri dada", "keringat malam", "penurunan berat badan"],
        warning_signs=["batuk darah", "sesak napas", "penurunan berat badan", "demam lama", "keringat malam", "kontak erat dengan penderita TBC"],
        screening_questions=[
            "Apakah batuk sudah berlangsung lebih dari 3 minggu?",
            "Apakah batuk disertai dahak, batuk darah, nyeri dada, atau sesak?",
            "Apakah ada demam, keringat malam, berat badan turun, atau nafsu makan menurun?",
            "Apakah tinggal serumah atau kontak erat lama dengan penderita TBC?",
        ],
        care_recommendation="Gejala yang mengarah ke TBC memerlukan pemeriksaan medis, tes dahak, dan terapi antibiotik jangka panjang. Herbal bukan terapi utama.",
        sample_user_message="Saya batuk lebih dari 3 minggu, berat badan turun, dan sering berkeringat malam. Apakah ini perlu dicurigai TBC?",
        prevention_question="Bagaimana pencegahan TBC di rumah dan kapan harus periksa?",
        symptom_headings=("gejala tbc (tuberkulosis)", "penularan dan gejala tuberkulosis (tbc)"),
        diagnosis_headings=("diagnosis tbc (tuberkulosis)",),
        prevention_headings=("pencegahan tbc (tuberkulosis)", "pengobatan dan pencegahan tuberkulosis (tbc)"),
    ),
    DiseaseDocSpec(
        id="tropic_filariasis",
        filename="kakigajah.docx",
        disease_name="Kaki Gajah (Filariasis)",
        topic="dugaan kaki gajah atau filariasis",
        primary_symptoms=["pembengkakan tungkai", "pembengkakan lengan atau kelamin", "pembengkakan kelenjar getah bening", "sering digigit nyamuk", "tinggal di daerah endemis"],
        warning_signs=["pembengkakan menetap atau berulang", "kulit menebal dan pecah-pecah", "luka pada tungkai bengkak", "nyeri dan kecacatan fungsional"],
        screening_questions=[
            "Apakah ada pembengkakan tungkai, lengan, kelamin, atau dada yang menetap?",
            "Apakah pembengkakan disertai pembesaran saluran atau kelenjar getah bening?",
            "Apakah tinggal atau baru bepergian ke daerah endemis kaki gajah?",
            "Apakah sering digigit nyamuk atau tinggal di lingkungan dengan banyak genangan?",
        ],
        care_recommendation="Pembengkakan yang mengarah ke filariasis perlu evaluasi medis dan terapi antiparasit. Pembengkakan kronis tidak boleh ditunda dengan herbal saja.",
        sample_user_message="Kaki saya bengkak makin besar dan saya tinggal di daerah yang banyak nyamuk. Apa ini bisa mengarah ke kaki gajah?",
        prevention_question="Bagaimana cara mencegah kaki gajah di daerah endemis?",
        symptom_headings=("gejala kaki gajah",),
        diagnosis_headings=("diagnosis kaki gajah",),
        prevention_headings=("pencegahan kaki gajah",),
    ),
    DiseaseDocSpec(
        id="tropic_kusta",
        filename="Kusta.docx",
        disease_name="Kusta",
        topic="dugaan kusta",
        primary_symptoms=["bercak pucat mati rasa", "kulit mati rasa", "luka tanpa nyeri", "kelemahan otot tangan atau kaki", "pembesaran saraf"],
        warning_signs=["mati rasa progresif", "kelemahan anggota gerak", "mata kering atau jarang berkedip", "kelainan wajah atau hidung", "luka telapak kaki tanpa rasa nyeri"],
        screening_questions=[
            "Apakah ada bercak kulit yang lebih pucat atau kemerahan dan terasa mati rasa?",
            "Apakah ada kelemahan pada tangan atau kaki, atau pembesaran saraf di siku/lutut?",
            "Apakah ada luka pada telapak kaki yang tidak terasa nyeri?",
            "Apakah gejala berkembang perlahan dan menetap dalam waktu lama?",
        ],
        care_recommendation="Gejala yang mengarah ke kusta perlu pemeriksaan medis dan terapi antibiotik kombinasi. Deteksi dini penting untuk mencegah kerusakan saraf permanen.",
        sample_user_message="Saya punya bercak kulit pucat yang mati rasa dan tangan terasa lebih lemah. Apakah ini gejala kusta?",
        prevention_question="Apa yang perlu diketahui tentang pencegahan dan deteksi dini kusta?",
        symptom_headings=("gejala kusta",),
        diagnosis_headings=("diagnosis kusta",),
        prevention_headings=("pencegahan kusta",),
    ),
    DiseaseDocSpec(
        id="tropic_skistosomiasis",
        filename="Skistosomiasis.docx",
        disease_name="Skistosomiasis",
        topic="dugaan skistosomiasis",
        primary_symptoms=["ruam atau gatal setelah kontak air tawar", "demam", "sakit perut", "diare", "urine berdarah", "BAB berdarah"],
        warning_signs=["urine berdarah", "BAB berdarah", "sesak napas", "nyeri dada", "kejang", "kelumpuhan tungkai"],
        screening_questions=[
            "Apakah ada riwayat berenang, mandi, atau bekerja di air tawar seperti sungai, danau, atau waduk?",
            "Apakah setelah itu muncul ruam/gatal, demam, batuk, sakit perut, atau diare?",
            "Apakah ada urine berdarah, BAB berdarah, atau sulit berkemih?",
            "Apakah ada sakit kepala, sesak, kejang, atau kelemahan tungkai?",
        ],
        care_recommendation="Paparan air tawar yang diikuti gejala sistemik atau perdarahan urine/tinja perlu pemeriksaan medis dan obat antiparasit. Herbal bukan terapi utama.",
        sample_user_message="Saya sempat mandi di air sungai lalu sekarang demam, gatal, dan urine berdarah. Apa ini bisa skistosomiasis?",
        prevention_question="Bagaimana mencegah skistosomiasis saat berada di area air tawar?",
        symptom_headings=("gejala skistosomiasis",),
        diagnosis_headings=("diagnosis skistosomiasis",),
        prevention_headings=("pencegahan skistosomiasis",),
    ),
    DiseaseDocSpec(
        id="tropic_candidiasis",
        filename="Candidiasis.docx",
        disease_name="Candidiasis",
        topic="kandidiasis pada mulut, kulit, atau kelamin",
        primary_symptoms=["bercak putih atau kuning di mulut", "gatal ekstrem di vagina", "keputihan menggumpal", "ruam gatal di lipatan kulit", "kulit pecah-pecah"],
        warning_signs=["nyeri saat menelan", "ruam bernanah atau melepuh", "infeksi berulang", "daya tahan tubuh lemah", "dugaan infeksi menyebar"],
        screening_questions=[
            "Apakah keluhan berada di mulut, vagina, atau lipatan kulit?",
            "Apakah ada bercak putih di mulut, keputihan menggumpal, atau ruam gatal di lipatan kulit?",
            "Apakah ada diabetes, HIV, kanker, penggunaan antibiotik lama, atau kortikosteroid jangka panjang?",
            "Apakah ruam melepuh, bernanah, atau keluhan berulang dan makin luas?",
        ],
        care_recommendation="Candidiasis perlu dibedakan menurut lokasi dan faktor risiko. Bila luas, berulang, atau pada pasien imun lemah, pemeriksaan medis lebih tepat daripada swamedikasi herbal.",
        sample_user_message="Saya gatal di area kelamin dan keputihan menggumpal. Apakah ini bisa kandidiasis?",
        prevention_question="Bagaimana cara mencegah candidiasis dan kapan perlu periksa?",
        symptom_headings=("gejala candidiasis",),
        diagnosis_headings=("diagnosis candidiasis",),
        prevention_headings=("pencegahan candidiasis",),
    ),
    DiseaseDocSpec(
        id="tropic_jamur_kuku",
        filename="Jamur kuku.docx",
        disease_name="Infeksi Jamur Kuku",
        topic="infeksi jamur kuku",
        primary_symptoms=["kuku menebal", "bintik putih atau kuning pada kuku", "kuku rapuh", "kuku berubah warna", "bau tidak sedap pada kuku"],
        warning_signs=["kuku terlepas dari kulit", "infeksi menyebar ke kulit sekitar", "komplikasi pada diabetes", "kerusakan kuku permanen"],
        screening_questions=[
            "Apakah kuku menebal, berubah warna, rapuh, atau mudah patah?",
            "Apakah infeksi lebih banyak terjadi di kuku kaki dan sudah berlangsung lama?",
            "Apakah ada diabetes, gangguan sirkulasi, atau daya tahan tubuh lemah?",
            "Apakah sering berjalan tanpa alas kaki di area lembap atau memakai sepatu tertutup lama?",
        ],
        care_recommendation="Jamur kuku sering memerlukan terapi antijamur jangka panjang dan kontrol dokter, terutama bila pasien diabetes atau kuku sudah rusak berat.",
        sample_user_message="Kuku kaki saya menebal, kuning, dan rapuh. Apakah ini gejala jamur kuku?",
        prevention_question="Bagaimana cara mencegah jamur kuku kambuh?",
        symptom_headings=("gejala jamur kuku",),
        diagnosis_headings=("diagnosis jamur kuku",),
        prevention_headings=("pencegahan jamur kuku",),
    ),
    DiseaseDocSpec(
        id="tropic_panu",
        filename="panu.docx",
        disease_name="Panu",
        topic="panu atau tinea versicolor",
        primary_symptoms=["bercak lebih terang atau gelap", "bercak bersisik", "gatal saat berkeringat", "bercak di punggung dada leher atau wajah"],
        warning_signs=["bercak makin luas", "tidak membaik setelah 2-3 minggu", "sering kambuh", "perlu konfirmasi diagnosis"],
        screening_questions=[
            "Apakah bercaknya lebih terang atau lebih gelap dari kulit sekitar dan terasa bersisik?",
            "Apakah bercak muncul di punggung, dada, leher, lengan atas, atau wajah?",
            "Apakah gatal bertambah saat berkeringat atau di cuaca panas dan lembap?",
            "Apakah keluhan sering kambuh atau tidak membaik setelah beberapa minggu pengobatan?",
        ],
        care_recommendation="Panu umumnya tidak gawat, tetapi tetap perlu dibedakan dari kelainan kulit lain. Bila luas, sering kambuh, atau tidak membaik, konsultasi ke dokter lebih tepat.",
        sample_user_message="Kulit saya muncul bercak putih kecokelatan yang gatal saat berkeringat. Apakah ini panu?",
        prevention_question="Bagaimana cara mencegah panu kambuh?",
        symptom_headings=("gejala panu", "tanda dan karakteristik panu"),
        diagnosis_headings=("diagnosis panu",),
        prevention_headings=("pencegahan panu",),
    ),
]

DOCX_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
GENERIC_HEADING_PREFIXES = (
    "penyebab",
    "penyebab dan gejala",
    "penularan dan gejala",
    "gejala",
    "gejala tbc pada anak",
    "tanda dan karakteristik",
    "kapan harus ke dokter",
    "kapan harus ke",
    "diagnosis",
    "pengobatan",
    "pengobatan dan pencegahan",
    "komplikasi",
    "pencegahan",
    "faktor risiko",
)


def main() -> int:
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    sources = build_sources()
    records = build_records()
    sft_rows = build_sft_examples(records)
    write_jsonl(SOURCES_PATH, sources)
    write_jsonl(RECORDS_PATH, records)
    write_jsonl(SFT_PATH, sft_rows)
    write_combined_sft()
    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "document_dir": str(DOCUMENT_DIR),
                "source_rows": len(sources),
                "training_record_rows": len(records),
                "training_sft_rows": len(sft_rows),
                "included_documents": [spec.filename for spec in SPECS],
                "skipped_files": sorted(
                    file.name
                    for file in DOCUMENT_DIR.iterdir()
                    if file.is_file() and file.name not in {spec.filename for spec in SPECS}
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {SOURCES_PATH.relative_to(ROOT)}")
    print(f"Wrote {RECORDS_PATH.relative_to(ROOT)}")
    print(f"Wrote {SFT_PATH.relative_to(ROOT)}")
    print(f"Wrote {COMBINED_SFT_PATH.relative_to(ROOT)}")
    return 0


def build_sources() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    supported_names = {spec.filename for spec in SPECS}
    for spec in SPECS:
        path = DOCUMENT_DIR / spec.filename
        paragraphs = extract_docx_paragraphs(path)
        rows.append(
            {
                "id": spec.id,
                "source_path": str(path.relative_to(ROOT)),
                "title": spec.disease_name,
                "status": "parsed_docx",
                "excerpt": make_excerpt(" ".join(paragraphs)),
            }
        )

    for path in sorted(DOCUMENT_DIR.iterdir()):
        if not path.is_file() or path.name in supported_names or path.name == ".DS_Store":
            continue
        rows.append(
            {
                "id": slugify(path.stem),
                "source_path": str(path.relative_to(ROOT)),
                "title": path.name,
                "status": "skipped_non_docx_reference",
                "excerpt": "",
            }
        )
    return rows


def build_records() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in SPECS:
        path = DOCUMENT_DIR / spec.filename
        paragraphs = extract_docx_paragraphs(path)
        overview = summarize_text(collect_intro_paragraphs(paragraphs), max_sentences=3, max_chars=520)
        diagnosis_summary = summarize_text(
            collect_section_text(paragraphs, spec.diagnosis_headings),
            max_sentences=3,
            max_chars=520,
        )
        prevention_steps = to_points(collect_section_paragraphs(paragraphs, spec.prevention_headings), limit=5)
        rows.append(
            {
                "id": spec.id,
                "content_type": "disease_guidance",
                "topic": spec.topic,
                "formula_name": spec.disease_name,
                "ingredients": [],
                "symptoms": spec.primary_symptoms,
                "overview": overview,
                "preparation": "Tidak ada ramuan herbal utama. Fokus pada triase, deteksi dini, dan pemeriksaan medis yang tepat sesuai dokumen sumber.",
                "dosage": "Tidak berlaku sebagai dosis herbal; gunakan sebagai edukasi gejala, pencegahan, dan arahan pemeriksaan.",
                "diagnosis_summary": diagnosis_summary,
                "prevention_steps": prevention_steps,
                "warning_signs": spec.warning_signs,
                "screening_questions": spec.screening_questions,
                "care_recommendation": spec.care_recommendation,
                "safety_notes": spec.care_recommendation,
                "evidence_level": "clinical_guideline_reference",
                "source_title": spec.disease_name,
                "source_url": str(path.relative_to(ROOT)),
                "curation_method": "local_docx_guided_manual_curation",
            }
        )
    return rows


def build_sft_examples(records: list[dict[str, object]]) -> list[dict[str, object]]:
    by_id = {record["id"]: record for record in records}
    rows: list[dict[str, object]] = []
    for spec in SPECS:
        record = by_id[spec.id]
        rows.append(
            {
                "id": f"sft_{spec.id}_triage",
                "source_record_id": spec.id,
                "messages": [
                    {"role": "system", "content": TRIAGE_SYSTEM_PROMPT},
                    {"role": "user", "content": spec.sample_user_message},
                    {"role": "assistant", "content": triage_answer(record)},
                ],
            }
        )
        rows.append(
            {
                "id": f"sft_{spec.id}_prevention",
                "source_record_id": spec.id,
                "messages": [
                    {"role": "system", "content": PREVENTION_SYSTEM_PROMPT},
                    {"role": "user", "content": spec.prevention_question},
                    {"role": "assistant", "content": prevention_answer(record)},
                ],
            }
        )
    return rows


def triage_answer(record: dict[str, object]) -> str:
    disease_name = str(record.get("formula_name") or record.get("topic") or "penyakit")
    symptoms = as_list(record.get("symptoms"))[:5]
    screening_questions = as_list(record.get("screening_questions"))[:4]
    warning_signs = as_list(record.get("warning_signs"))[:4]
    care_recommendation = str(record.get("care_recommendation") or record.get("safety_notes") or "")
    overview = str(record.get("overview") or "")
    diagnosis_summary = str(record.get("diagnosis_summary") or "")

    blocks = [
        f"Fokus anamnesis awal adalah kemungkinan {disease_name.lower()}, tetapi ini bukan diagnosis medis final.",
    ]
    if overview:
        blocks.append(f"Ringkasan singkat: {overview}")
    if symptoms:
        blocks.append(f"Gejala kunci yang sering terkait: {', '.join(symptoms)}.")
    if screening_questions:
        blocks.append(
            "Pertanyaan anamnesis utama:\n"
            + "\n".join(f"- {question}" for question in screening_questions)
        )
    if warning_signs:
        blocks.append(
            "Tanda bahaya yang harus diwaspadai:\n"
            + "\n".join(f"- {item}" for item in warning_signs)
        )
    if diagnosis_summary:
        blocks.append(f"Pemeriksaan medis yang lazim: {diagnosis_summary}")
    blocks.append(
        f"Keputusan awal: {care_recommendation}\n\n"
        "Jika gejala mengarah ke penyakit infeksi tropis, penyakit menular kronis, atau infeksi jamur yang luas/berulang, "
        "jangan menjadikan herbal sebagai terapi utama."
    )
    blocks.append(
        "Informasi ini bersifat edukasi awal, bukan diagnosis medis final dan bukan pengganti konsultasi tenaga kesehatan."
    )
    return "\n\n".join(blocks)


def prevention_answer(record: dict[str, object]) -> str:
    disease_name = str(record.get("formula_name") or record.get("topic") or "penyakit")
    prevention_steps = as_list(record.get("prevention_steps"))[:5]
    warning_signs = as_list(record.get("warning_signs"))[:3]
    care_recommendation = str(record.get("care_recommendation") or record.get("safety_notes") or "")

    blocks = [f"Pencegahan {disease_name} yang bisa diprioritaskan:"]
    if prevention_steps:
        blocks.append("\n".join(f"- {step}" for step in prevention_steps))
    else:
        blocks.append("- Hindari faktor risiko utama yang disebutkan dalam dokumen sumber.")
    if warning_signs:
        blocks.append(
            "Segera periksa bila muncul gejala berikut:\n"
            + "\n".join(f"- {item}" for item in warning_signs)
        )
    blocks.append(
        f"Arahan tambahan: {care_recommendation}\n\n"
        "Untuk penyakit serius atau menular, pencegahan lingkungan, perlindungan diri, dan pemeriksaan dini lebih penting "
        "daripada mengandalkan herbal sebagai terapi utama."
    )
    blocks.append(
        "Informasi ini bersifat edukasi awal, bukan diagnosis medis final dan bukan pengganti konsultasi tenaga kesehatan."
    )
    return "\n\n".join(blocks)


def write_combined_sft() -> None:
    rows: list[dict[str, object]] = []
    for path in sorted(TRAINING_DIR.glob("*_training_sft.jsonl")):
        if path.name == COMBINED_SFT_PATH.name:
            continue
        with path.open(encoding="utf-8") as file:
            rows.extend(json.loads(line) for line in file if line.strip())
    if rows:
        write_jsonl(COMBINED_SFT_PATH, rows)


def extract_docx_paragraphs(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        xml_bytes = archive.read("word/document.xml")
    root = ET.fromstring(xml_bytes)
    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:p", DOCX_NS):
        texts = [node.text for node in paragraph.findall(".//w:t", DOCX_NS) if node.text]
        normalized = normalize_text("".join(texts))
        if normalized:
            paragraphs.append(normalized)
    return paragraphs


def collect_intro_paragraphs(paragraphs: list[str]) -> list[str]:
    collected: list[str] = []
    title_skipped = False
    for paragraph in paragraphs:
        lower = paragraph.lower()
        if not title_skipped and lower.startswith("penyakit :"):
            title_skipped = True
            continue
        if is_heading(paragraph):
            break
        collected.append(paragraph)
    return collected


def collect_section_text(paragraphs: list[str], headings: tuple[str, ...]) -> str:
    return " ".join(collect_section_paragraphs(paragraphs, headings))


def collect_section_paragraphs(paragraphs: list[str], headings: tuple[str, ...]) -> list[str]:
    heading_prefixes = tuple(normalize_text(item).lower() for item in headings)
    exact_match = collect_section_paragraphs_with_mode(paragraphs, heading_prefixes, allow_loose_match=False)
    if exact_match:
        return exact_match
    return collect_section_paragraphs_with_mode(paragraphs, heading_prefixes, allow_loose_match=True)


def collect_section_paragraphs_with_mode(
    paragraphs: list[str],
    heading_prefixes: tuple[str, ...],
    *,
    allow_loose_match: bool,
) -> list[str]:
    collected: list[str] = []
    active = False
    for paragraph in paragraphs:
        lower = paragraph.lower()
        if matches_section_heading(lower, paragraph, heading_prefixes, allow_loose_match=allow_loose_match):
            active = True
            continue
        if active and is_heading(paragraph):
            break
        if active:
            if lower.startswith("terakhir diperbarui:") or lower.startswith("ditinjau oleh:"):
                continue
            collected.append(paragraph)
    return collected


def matches_section_heading(
    lower: str,
    paragraph: str,
    heading_prefixes: tuple[str, ...],
    *,
    allow_loose_match: bool,
) -> bool:
    if not is_heading(paragraph):
        return False
    if any(lower == prefix for prefix in heading_prefixes):
        return True
    if not allow_loose_match:
        return False
    return any(lower.startswith(prefix) or lower.endswith(prefix) for prefix in heading_prefixes)


def is_heading(paragraph: str) -> bool:
    lower = normalize_text(paragraph).lower()
    return any(lower.startswith(prefix) for prefix in GENERIC_HEADING_PREFIXES)


def summarize_text(text: str | list[str], *, max_sentences: int, max_chars: int) -> str:
    if isinstance(text, list):
        text = " ".join(text)
    normalized = normalize_text(text)
    if not normalized:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    trimmed = " ".join(sentences[:max_sentences]).strip()
    if len(trimmed) <= max_chars:
        return trimmed
    return trimmed[: max_chars - 3].rstrip() + "..."


def to_points(paragraphs: list[str], limit: int) -> list[str]:
    points: list[str] = []
    for paragraph in paragraphs:
        clean = paragraph.lstrip("•- ").strip()
        if not clean:
            continue
        lower = clean.lower()
        if lower.startswith("terakhir diperbarui:") or lower.startswith("ditinjau oleh:"):
            continue
        if clean not in points:
            points.append(clean)
        if len(points) >= limit:
            break
    if points:
        return points[:limit]

    fallback_sentences = re.split(r"(?<=[.!?])\s+", normalize_text(" ".join(paragraphs)))
    return [sentence.strip() for sentence in fallback_sentences[:limit] if sentence.strip()]


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def make_excerpt(text: str, max_chars: int = 1200) -> str:
    normalized = normalize_text(text)
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def as_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(";") if item.strip()]
    return []


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_:-]+", "_", value.strip()).strip("_").lower()


if __name__ == "__main__":
    sys.exit(main())
