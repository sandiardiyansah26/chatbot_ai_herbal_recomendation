from __future__ import annotations

import csv
import html
import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = ROOT / "data" / "referensi"
TRAINING_DIR = ROOT / "data" / "traning"

CASES_CSV_PATH = REFERENCE_DIR / "focused_mild_ailment_herbal_dataset.csv"
CASES_JSONL_PATH = REFERENCE_DIR / "focused_mild_ailment_herbal_dataset.jsonl"
FORMULAS_CSV_PATH = REFERENCE_DIR / "herbal_formulas.csv"
FORMULAS_JSONL_PATH = REFERENCE_DIR / "herbal_formulas.jsonl"
HERBS_CSV_PATH = REFERENCE_DIR / "herbal_references.csv"
HERBS_JSONL_PATH = REFERENCE_DIR / "herbal_references.jsonl"
PREPARATION_OVERRIDES_PATH = REFERENCE_DIR / "pengolahan_dan_dosis_ramuan.csv"
GENERATED_SOURCES_PATH = REFERENCE_DIR / "generated_reference_sources.jsonl"
GENERATED_MANIFEST_PATH = REFERENCE_DIR / "generated_reference_manifest.json"

HERBAL_TRAINING_RECORDS_PATH = TRAINING_DIR / "herbal_training_records.jsonl"
HERBAL_TRAINING_SFT_PATH = TRAINING_DIR / "herbal_training_sft.jsonl"
COMBINED_TRAINING_SFT_PATH = TRAINING_DIR / "combined_training_sft.jsonl"
RAG_SFT_PATH = TRAINING_DIR / "rag_learning_sft.jsonl"
COMBINED_TRAINING_SFT_RAG_PATH = TRAINING_DIR / "combined_training_sft_rag.jsonl"

USER_AGENT = "Mozilla/5.0 (compatible; HerbalDoctorDatasetBuilder/1.0)"
REQUEST_TIMEOUT_SECONDS = 25
MAX_TARGETS_PER_HERB = 4
CASE_VARIANTS_PER_TARGET = 5

SYSTEM_PROMPT = (
    "Anda adalah chatbot edukasi ramuan herbal berbahasa Indonesia. "
    "Lakukan anamnesis ringan, hanya rekomendasikan ramuan untuk keluhan ringan, "
    "jelaskan cara pengolahan dan dosis, dan selalu tekankan bahwa ini bukan diagnosis "
    "medis final serta bukan pengganti konsultasi tenaga kesehatan."
)

WHO_MONOGRAPH_SOURCES = [
    {
        "source_title": "WHO Monographs on Selected Medicinal Plants Volume 1",
        "source_url": "https://iris.who.int/handle/10665/42052",
        "text_url": "https://iris.who.int/server/api/core/bitstreams/8016c469-435f-47a1-8f18-2fd2e8c97967/content",
    },
    {
        "source_title": "WHO Monographs on Selected Medicinal Plants Volume 2",
        "source_url": "https://iris.who.int/handle/10665/42053",
        "text_url": "https://iris.who.int/server/api/core/bitstreams/0bf1d2f0-3a21-4f9e-9f4e-3302d5ed2fd7/content",
    },
    {
        "source_title": "WHO Monographs on Selected Medicinal Plants Volume 3",
        "source_url": "https://iris.who.int/handle/10665/42054",
        "text_url": "https://iris.who.int/server/api/core/bitstreams/232879ef-08bc-4b1c-aed3-995e61153275/content",
    },
    {
        "source_title": "WHO Monographs on Selected Medicinal Plants Volume 4",
        "source_url": "https://iris.who.int/handle/10665/42055",
        "text_url": "https://iris.who.int/server/api/core/bitstreams/15fc0105-fa6f-4c64-ac07-03764ee8ac2e/content",
    },
]

NCCIH_INDEX_URL = "https://www.nccih.nih.gov/health/herbsataglance"
NCCIH_BLOCKED_PATHS = {
    "/health/atoz",
    "/health/complementary-alternative-or-integrative-health-whats-in-a-name",
    "/health/espanol",
    "/health/herblist-app",
    "/health/herbsataglance",
    "/health/know-science",
    "/health/pain",
    "/health/providers",
    "/health/safety",
    "/health/tips",
}

EXCLUDED_RECOMMENDATION_KEYS = {
    "ephedra",
    "herba ephedrae",
    "yohimbe",
    "thunder god vine",
    "brucea",
    "fructus bruceae",
    "rauwolfia",
    "radix rauwolfiae",
    "european mistletoe",
    "hoodia",
}

TITLE_PART_MAP = {
    "Bulbus": "umbi/bulb",
    "Radix": "akar/root",
    "Rhizoma": "rimpang/rhizome",
    "Folium": "daun/leaf",
    "Flos": "bunga/flower",
    "Herba": "herba/whole herb",
    "Fructus": "buah/fruit",
    "Semen": "biji/seed",
    "Cortex": "kulit batang/bark",
    "Aetheroleum": "minyak atsiri/essential oil",
    "Oleum": "minyak/oil",
    "Gummi": "resin/gum",
    "Ramulus": "ranting/stem",
    "Strobilus": "strobilus",
}

LOCAL_NAME_OVERRIDES = {
    "allium cepa": "Bawang bombai",
    "allium sativum": "Bawang putih",
    "andrographis paniculata": "Sambiloto",
    "centella asiatica": "Pegagan",
    "curcuma longa": "Kunyit",
    "zingiber officinale": "Jahe",
    "cinnamomum": "Kayu manis",
    "foeniculum vulgare": "Adas",
    "glycyrrhiza": "Akar manis",
    "matricaria recutita": "Kamomil",
    "melissa officinalis": "Melissa",
    "mentha piperita": "Peppermint",
    "syzygium aromaticum": "Cengkih",
    "thymus vulgaris": "Thyme",
    "valeriana officinalis": "Valerian",
    "elettaria cardamomum": "Kapulaga",
    "withania somnifera": "Ashwagandha",
}


@dataclass(frozen=True)
class SymptomProfile:
    id: str
    label: str
    keywords: tuple[str, ...]
    case_variants: tuple[str, ...]
    anamnesis_question: str
    context: str
    red_flags: str
    preferred_mode: str = "oral"


@dataclass
class HerbSource:
    key: str
    display_name: str
    scientific_name: str
    part_used: str
    aliases: set[str] = field(default_factory=set)
    supported_uses: str = ""
    traditional_uses: str = ""
    folk_uses: str = ""
    dosage_forms: str = ""
    posology: str = ""
    safety: str = ""
    source_titles: list[str] = field(default_factory=list)
    source_urls: list[str] = field(default_factory=list)

    def recommendation_allowed(self) -> bool:
        haystack = " ".join([self.display_name, self.scientific_name, *self.aliases]).lower()
        return not any(term in haystack for term in EXCLUDED_RECOMMENDATION_KEYS)


PROFILES: tuple[SymptomProfile, ...] = (
    SymptomProfile(
        id="mual_ringan",
        label="mual ringan",
        keywords=("nausea", "vomiting", "emesis", "morning sickness", "travel sickness"),
        case_variants=(
            "mual ringan",
            "mual ringan setelah makan",
            "perut terasa tidak nyaman disertai mual ringan",
            "mual ringan tanpa muntah terus-menerus",
            "mual ringan yang masih bisa makan dan minum",
        ),
        anamnesis_question="Sejak kapan mual muncul? Apakah ada muntah terus-menerus, nyeri perut hebat, muntah darah, atau tanda dehidrasi?",
        context="pendamping keluhan mual ringan",
        red_flags="muntah terus-menerus, muntah darah, nyeri perut hebat, pingsan, atau dehidrasi",
    ),
    SymptomProfile(
        id="gangguan_pencernaan_ringan",
        label="gangguan pencernaan ringan",
        keywords=("dyspepsia", "indigestion", "gastrointestinal discomfort", "flatulence", "carminative", "abdominal discomfort", "bloating", "colic"),
        case_variants=(
            "perut terasa tidak nyaman ringan",
            "kembung dan begah ringan",
            "gangguan pencernaan ringan setelah makan",
            "nyaman di lambung berkurang tetapi belum berat",
            "keluhan pencernaan ringan tanpa muntah hebat",
        ),
        anamnesis_question="Keluhan pencernaan ini sejak kapan? Apakah ada muntah hebat, BAB hitam, nyeri perut sangat kuat, atau demam tinggi?",
        context="pendamping rasa tidak nyaman lambung dan pencernaan ringan",
        red_flags="muntah hebat, nyeri perut tajam, BAB hitam, muntah darah, atau demam tinggi",
    ),
    SymptomProfile(
        id="nafsu_makan_menurun",
        label="nafsu makan menurun",
        keywords=("loss of appetite", "appetite"),
        case_variants=(
            "nafsu makan menurun ringan",
            "selera makan berkurang setelah badan kurang fit",
            "nafsu makan belum kembali seperti biasa",
            "makan terasa tidak terlalu enak sejak sakit ringan",
            "nafsu makan turun tetapi masih bisa minum",
        ),
        anamnesis_question="Nafsu makan menurun sudah berapa lama? Apakah ada penurunan berat badan drastis, muntah, demam lama, atau lemas berat?",
        context="pendamping pemulihan ringan saat selera makan menurun",
        red_flags="berat badan turun cepat, muntah terus, demam lama, atau sangat lemas",
    ),
    SymptomProfile(
        id="batuk_pilek_ringan",
        label="batuk atau pilek ringan",
        keywords=("cough", "common cold", "cold", "catarrh", "upper respiratory"),
        case_variants=(
            "batuk ringan",
            "pilek ringan dan tenggorokan terasa tidak nyaman",
            "batuk pilek ringan tanpa sesak",
            "batuk ringan pada keluhan saluran napas atas",
            "pilek ringan disertai rasa kurang nyaman di tenggorokan",
        ),
        anamnesis_question="Apakah ada sesak napas, demam tinggi, dahak berdarah, nyeri dada, atau napas terasa cepat?",
        context="pendamping gejala saluran napas atas yang masih ringan",
        red_flags="sesak, demam tinggi, dahak berdarah, nyeri dada, atau napas cepat",
    ),
    SymptomProfile(
        id="tenggorokan_tidak_nyaman",
        label="tenggorokan tidak nyaman",
        keywords=("sore throat", "pharyngitis", "laryngitis", "throat irritation", "mouth and throat irritation"),
        case_variants=(
            "tenggorokan tidak nyaman",
            "tenggorokan terasa kering dan perih ringan",
            "nyeri tenggorokan ringan saat menelan",
            "suara terasa serak ringan",
            "iritasi tenggorokan ringan tanpa sesak",
        ),
        anamnesis_question="Apakah ada sesak, sulit menelan berat, demam tinggi, air liur sulit ditelan, atau bengkak pada leher/wajah?",
        context="pendamping iritasi tenggorokan yang masih ringan",
        red_flags="sesak, sulit menelan berat, demam tinggi, air liur sulit ditelan, atau bengkak leher/wajah",
        preferred_mode="gargle",
    ),
    SymptomProfile(
        id="diare_ringan",
        label="diare ringan",
        keywords=("diarrhoea", "diarrhea", "loose stool"),
        case_variants=(
            "diare ringan",
            "BAB cair ringan tanpa darah",
            "pencernaan sedang sensitif dengan BAB lebih sering",
            "diare ringan tetapi masih bisa minum",
            "BAB cair ringan tanpa tanda dehidrasi",
        ),
        anamnesis_question="Sudah berapa kali BAB cair? Apakah ada darah, muntah terus, lemas berat, haus sekali, atau BAK sangat berkurang?",
        context="pendamping diare ringan tanpa tanda bahaya",
        red_flags="BAB berdarah, muntah terus, lemas berat, BAK berkurang, atau dehidrasi",
    ),
    SymptomProfile(
        id="konstipasi_ringan",
        label="konstipasi ringan",
        keywords=("constipation", "laxative"),
        case_variants=(
            "sulit BAB ringan",
            "BAB terasa keras dan tidak lancar",
            "konstipasi ringan beberapa hari",
            "perut terasa penuh karena BAB tidak lancar",
            "BAB sulit tetapi belum disertai nyeri hebat",
        ),
        anamnesis_question="Sejak kapan sulit BAB? Apakah ada muntah, perut membesar, BAB hitam, perdarahan, atau nyeri perut berat?",
        context="pendamping konstipasi ringan jangka pendek",
        red_flags="muntah, perut membesar, BAB hitam, perdarahan, atau nyeri perut berat",
    ),
    SymptomProfile(
        id="nyeri_pegal_ringan",
        label="nyeri atau pegal ringan",
        keywords=("muscle pain", "joint pain", "arthralgia", "sprain", "menstrual cramps", "rheumatic", "back pain", "muscular"),
        case_variants=(
            "pegal ringan pada tubuh",
            "nyeri otot ringan setelah aktivitas",
            "sendi terasa tidak nyaman ringan",
            "pegal ringan tanpa bengkak berat",
            "nyeri ringan yang masih bisa beraktivitas",
        ),
        anamnesis_question="Bagian tubuh mana yang nyeri? Apakah ada bengkak besar, kemerahan luas, trauma berat, demam tinggi, atau sulit digerakkan?",
        context="pendamping nyeri otot atau pegal yang masih ringan",
        red_flags="bengkak besar, trauma berat, demam tinggi, kelemahan anggota gerak, atau sulit digerakkan",
    ),
    SymptomProfile(
        id="sulit_tidur_gelisah",
        label="sulit tidur atau gelisah ringan",
        keywords=("insomnia", "sleep", "sedative", "anxiety", "nervousness", "stress", "restlessness"),
        case_variants=(
            "sulit tidur ringan",
            "badan terasa gelisah ringan",
            "tidur kurang nyenyak karena stres ringan",
            "sulit rileks di malam hari",
            "gelisah ringan tanpa sesak atau nyeri dada",
        ),
        anamnesis_question="Apakah keluhan gelisah disertai nyeri dada, sesak, pikiran untuk menyakiti diri, atau berdebar sangat kuat?",
        context="pendamping relaksasi dan tidur ringan",
        red_flags="nyeri dada, sesak, pingsan, pikiran menyakiti diri, atau berdebar sangat kuat",
    ),
    SymptomProfile(
        id="gatal_iritasi_kulit_ringan",
        label="gatal atau iritasi kulit ringan",
        keywords=("dermatitis", "eczema", "itching", "itch", "skin irritation", "rash"),
        case_variants=(
            "gatal ringan pada kulit",
            "iritasi kulit ringan di area terbatas",
            "ruam ringan tanpa sesak",
            "kulit terasa gatal dan kering ringan",
            "keluhan kulit ringan tanpa luka luas",
        ),
        anamnesis_question="Ruam atau gatal ada di area mana? Apakah ada sesak, bengkak wajah, nanah luas, demam, atau luka menyebar cepat?",
        context="pendamping iritasi kulit ringan lokal",
        red_flags="sesak, bengkak wajah, nanah luas, demam, atau ruam menyebar cepat",
        preferred_mode="topical",
    ),
    SymptomProfile(
        id="luka_memar_ringan",
        label="luka kecil atau memar ringan",
        keywords=("wound", "bruise", "scar", "keloid", "burn"),
        case_variants=(
            "memar ringan",
            "lecet ringan pada kulit",
            "bekas luka ringan yang sedang dipulihkan",
            "iritasi kulit setelah luka kecil",
            "kulit memar ringan tanpa perdarahan aktif",
        ),
        anamnesis_question="Apakah lukanya dalam, bernanah, berdarah terus, sangat nyeri, atau disertai demam?",
        context="pendamping perawatan luka permukaan atau memar ringan",
        red_flags="luka dalam, perdarahan terus, nanah, nyeri berat, atau demam",
        preferred_mode="topical",
    ),
    SymptomProfile(
        id="haid_tidak_nyaman",
        label="haid tidak nyaman",
        keywords=("dysmenorr", "menstrual", "premenstrual"),
        case_variants=(
            "nyeri haid ringan",
            "perut terasa tidak nyaman saat haid",
            "haid disertai pegal ringan",
            "keluhan prahaid ringan",
            "haid terasa kurang nyaman tetapi masih bisa aktivitas",
        ),
        anamnesis_question="Apakah nyeri haid sangat berat, pendarahan sangat banyak, pingsan, demam, atau kemungkinan hamil?",
        context="pendamping ketidaknyamanan haid yang masih ringan",
        red_flags="perdarahan sangat banyak, pingsan, nyeri sangat berat, demam, atau kemungkinan hamil",
    ),
    SymptomProfile(
        id="sariawan_mulut_ringan",
        label="sariawan atau mulut tidak nyaman",
        keywords=("stomatitis", "mouth ulcer", "oral inflammation", "oral mucosa"),
        case_variants=(
            "sariawan ringan",
            "mulut terasa perih ringan",
            "iritasi mulut ringan saat makan",
            "mulut tidak nyaman karena luka kecil",
            "keluhan mulut ringan tanpa sulit minum",
        ),
        anamnesis_question="Apakah luka mulut sangat banyak, sulit minum, demam tinggi, atau bengkak hebat di mulut/tenggorokan?",
        context="pendamping iritasi mulut ringan",
        red_flags="sulit minum, demam tinggi, bengkak hebat, atau luka mulut luas",
        preferred_mode="gargle",
    ),
    SymptomProfile(
        id="sakit_kepala_ringan",
        label="sakit kepala ringan",
        keywords=("headache", "migraine"),
        case_variants=(
            "sakit kepala ringan",
            "kepala terasa tidak nyaman ringan",
            "nyeri kepala ringan setelah kurang tidur",
            "kepala terasa berat ringan",
            "sakit kepala ringan tanpa kaku leher",
        ),
        anamnesis_question="Apakah sakit kepala disertai kaku leher, muntah hebat, gangguan bicara, lemah satu sisi, atau penurunan kesadaran?",
        context="pendamping sakit kepala ringan tanpa red flag neurologis",
        red_flags="kaku leher, muntah hebat, lemah satu sisi, gangguan bicara, atau penurunan kesadaran",
    ),
    SymptomProfile(
        id="saluran_kemih_ringan",
        label="keluhan saluran kemih ringan",
        keywords=("cystitis", "urinary", "bladder irritation"),
        case_variants=(
            "keluhan saluran kemih ringan",
            "anyang-anyangan ringan",
            "BAK terasa kurang nyaman ringan",
            "BAK lebih sering tetapi belum berat",
            "rasa tidak nyaman ringan pada saluran kemih",
        ),
        anamnesis_question="Apakah ada demam, nyeri pinggang, urin berdarah, muntah, atau hamil?",
        context="pendamping rasa tidak nyaman ringan pada BAK",
        red_flags="demam, nyeri pinggang, urin berdarah, muntah, atau kehamilan",
    ),
    SymptomProfile(
        id="kelelahan_ringan",
        label="kelelahan ringan atau masa pemulihan",
        keywords=("convalescence", "fatigue", "weakness", "tonic"),
        case_variants=(
            "badan terasa kurang fit ringan",
            "kelelahan ringan setelah sakit",
            "masa pemulihan dengan tenaga belum pulih penuh",
            "badan terasa lemas ringan",
            "kurang fit tetapi masih bisa makan dan minum",
        ),
        anamnesis_question="Apakah ada sesak, demam tinggi, penurunan kesadaran, nyeri dada, atau lemas sampai sulit bangun?",
        context="pendamping masa pemulihan ringan",
        red_flags="sesak, demam tinggi, nyeri dada, sulit bangun, atau penurunan kesadaran",
    ),
)

PROFILE_LOOKUP = {profile.id: profile for profile in PROFILES}


def main() -> int:
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    legacy_cases = load_csv_rows(CASES_CSV_PATH)
    legacy_formulas = load_csv_rows(FORMULAS_CSV_PATH)
    legacy_herbs = load_csv_rows(HERBS_CSV_PATH)
    preparation_overrides = load_preparation_overrides(PREPARATION_OVERRIDES_PATH)

    who_sources = fetch_who_sources()
    nccih_sources = fetch_nccih_sources()
    merged_sources = merge_herb_sources([*who_sources, *nccih_sources], legacy_herbs)
    matched_profiles = build_profile_matches(merged_sources)

    herb_rows = build_herb_rows(merged_sources, matched_profiles, legacy_herbs)
    formula_rows, formula_contexts = build_formula_rows(merged_sources, matched_profiles, legacy_formulas)
    case_rows = build_case_rows(formula_contexts, preparation_overrides, legacy_cases)

    write_rows(CASES_CSV_PATH, case_rows)
    write_jsonl(CASES_JSONL_PATH, case_rows)
    write_rows(FORMULAS_CSV_PATH, formula_rows)
    write_jsonl(FORMULAS_JSONL_PATH, formula_rows)
    write_rows(HERBS_CSV_PATH, herb_rows)
    write_jsonl(HERBS_JSONL_PATH, herb_rows)

    generated_source_rows = build_generated_source_rows(merged_sources, matched_profiles)
    write_jsonl(GENERATED_SOURCES_PATH, generated_source_rows)

    training_records = build_training_records(case_rows, formula_rows, herb_rows)
    herbal_sft = build_sft_examples(training_records)
    write_jsonl(HERBAL_TRAINING_RECORDS_PATH, training_records)
    write_jsonl(HERBAL_TRAINING_SFT_PATH, herbal_sft)
    refresh_combined_sft()
    refresh_combined_sft_rag()

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "assumption": (
            "Target minimal 1500 ditafsirkan sebagai total gabungan katalog runtime "
            "cases + formulas + herbs, dengan fokus menjaga akurasi sumber primer."
        ),
        "source_documents": [
            {"title": source["source_title"], "url": source["source_url"]} for source in WHO_MONOGRAPH_SOURCES
        ]
        + [
            {"title": "NCCIH Herbs at a Glance", "url": NCCIH_INDEX_URL},
        ],
        "counts": {
            "who_monograph_herbs": len(who_sources),
            "nccih_herbs": len(nccih_sources),
            "merged_herbs": len(merged_sources),
            "herb_rows": len(herb_rows),
            "formula_rows": len(formula_rows),
            "case_rows": len(case_rows),
            "runtime_total": len(herb_rows) + len(formula_rows) + len(case_rows),
            "herbal_training_records": len(training_records),
            "herbal_training_sft": len(herbal_sft),
        },
        "profile_breakdown": profile_breakdown(formula_contexts),
        "excluded_recommendation_terms": sorted(EXCLUDED_RECOMMENDATION_KEYS),
    }
    GENERATED_MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))
    return 0


def fetch_who_sources() -> list[HerbSource]:
    herbs: list[HerbSource] = []
    for source in WHO_MONOGRAPH_SOURCES:
        try:
            text = fetch_text(source["text_url"])
        except (HTTPError, URLError, TimeoutError):
            continue
        herbs.extend(parse_who_monographs(text, source["source_title"], source["source_url"]))
    return herbs


def parse_who_monographs(text: str, source_title: str, source_url: str) -> list[HerbSource]:
    normalized = normalize_who_text(text)
    matches = list(re.finditer(r"(?:\n|\f)([A-Z][A-Za-z .\-\'()]+?)\n\nDefinition", normalized))
    herbs: list[HerbSource] = []
    for index, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.start(1)
        end = matches[index + 1].start(1) if index + 1 < len(matches) else len(normalized)
        chunk = normalized[start:end]
        definition = extract_section(chunk, "Definition", ("Synonyms", "Selected vernacular names", "Description"))
        vernacular = extract_section(chunk, "Selected vernacular names", ("Description", "Plant material of interest"))
        supported = extract_section(
            chunk,
            "Uses supported by clinical data",
            (
                "Uses described in pharmacopoeias and in traditional systems of medicine",
                "Uses described in folk medicine, not supported by experimental or clinical data",
                "Pharmacology",
            ),
        )
        traditional = extract_section(
            chunk,
            "Uses described in pharmacopoeias and in traditional systems of medicine",
            ("Uses described in folk medicine, not supported by experimental or clinical data", "Pharmacology"),
        )
        folk = extract_section(
            chunk,
            "Uses described in folk medicine, not supported by experimental or clinical data",
            ("Pharmacology", "Contraindications"),
        )
        safety = unique_join(
            [
                extract_section(chunk, "Contraindications", ("Warnings", "Precautions", "Adverse reactions", "Posology")),
                extract_section(chunk, "Warnings", ("Precautions", "Adverse reactions", "Posology")),
                extract_section(chunk, "Precautions", ("Adverse reactions", "Posology")),
                extract_section(chunk, "Adverse reactions", ("Posology", "References")),
            ]
        )
        dosage_forms = extract_section(chunk, "Dosage forms", ("Medicinal uses", "Uses supported by clinical data"))
        posology = extract_section(chunk, "Posology", ("References",))
        scientific_name = extract_scientific_name(definition)
        common_name = extract_common_name(vernacular) or title
        part_used = derive_part_used(title, definition)
        herbs.append(
            HerbSource(
                key=herb_key(scientific_name, common_name),
                display_name=localize_display_name(common_name, scientific_name),
                scientific_name=scientific_name,
                part_used=part_used,
                aliases={common_name, title},
                supported_uses=supported,
                traditional_uses=traditional,
                folk_uses=folk,
                dosage_forms=dosage_forms,
                posology=posology,
                safety=safety,
                source_titles=[source_title],
                source_urls=[source_url],
            )
        )
    return herbs


def fetch_nccih_sources() -> list[HerbSource]:
    try:
        index_html = fetch_text(NCCIH_INDEX_URL)
    except (HTTPError, URLError, TimeoutError):
        return []
    paths = sorted(set(re.findall(r'"(/health/[a-z0-9\-]+)"', index_html)))
    herbs: list[HerbSource] = []
    for path in paths:
        if path in NCCIH_BLOCKED_PATHS:
            continue
        try:
            payload = fetch_json(f"https://www.nccih.nih.gov/page-data{path}/page-data.json")
        except (HTTPError, URLError, TimeoutError, KeyError, json.JSONDecodeError):
            continue
        fact = payload.get("result", {}).get("data", {}).get("factsheetJson", {}).get("factsheet", {})
        name = clean_text(str(fact.get("name") or "")).strip()
        if not name:
            continue
        common_names = split_names(str(fact.get("commonNames") or ""))
        latin_names = split_names(str(fact.get("latinNames") or ""))
        blocks = fact.get("blocks") or []
        learned = ""
        safety = ""
        background = ""
        for block in blocks:
            heading = clean_text(str(block.get("heading") or "")).lower()
            description = html_to_text(str(block.get("description") or ""))
            if heading == "what have we learned?":
                learned = description
            elif heading == "what do we know about safety?":
                safety = description
            elif heading == "background":
                background = description

        part_used = derive_part_from_background(background)
        scientific_name = latin_names[0] if latin_names else ""
        herbs.append(
            HerbSource(
                key=herb_key(scientific_name, name),
                display_name=localize_display_name(name, scientific_name),
                scientific_name=scientific_name,
                part_used=part_used,
                aliases=set([name, *common_names, *latin_names]),
                supported_uses=learned,
                traditional_uses=background,
                folk_uses="",
                dosage_forms="Factsheet NCCIH; gunakan hanya sediaan terstandar atau bentuk tradisional yang aman sesuai konteks.",
                posology="Ikuti etiket produk terstandar atau saran profesional; NCCIH tidak menetapkan satu dosis rumah tangga baku untuk semua bentuk.",
                safety=safety,
                source_titles=["NCCIH Herbs at a Glance"],
                source_urls=[f"https://www.nccih.nih.gov{path}"],
            )
        )
    return herbs


def merge_herb_sources(sources: list[HerbSource], legacy_herbs: list[dict[str, str]]) -> list[HerbSource]:
    legacy_name_map: dict[str, str] = {}
    for row in legacy_herbs:
        scientific_name = normalize_key(row.get("nama_latin", ""))
        local_name = row.get("nama_lokal", "").strip()
        if scientific_name and local_name:
            legacy_name_map[scientific_name] = local_name

    merged: dict[str, HerbSource] = {}
    for source in sources:
        key = herb_key(source.scientific_name, source.display_name)
        if key not in merged:
            merged[key] = HerbSource(
                key=key,
                display_name=source.display_name,
                scientific_name=source.scientific_name,
                part_used=source.part_used,
                aliases=set(source.aliases),
                supported_uses=source.supported_uses,
                traditional_uses=source.traditional_uses,
                folk_uses=source.folk_uses,
                dosage_forms=source.dosage_forms,
                posology=source.posology,
                safety=source.safety,
                source_titles=list(source.source_titles),
                source_urls=list(source.source_urls),
            )
            continue

        existing = merged[key]
        existing.display_name = prefer_local_name(existing.display_name, source.display_name, legacy_name_map.get(normalize_key(existing.scientific_name), ""))
        existing.part_used = existing.part_used or source.part_used
        existing.aliases.update(source.aliases)
        existing.supported_uses = unique_join([existing.supported_uses, source.supported_uses])
        existing.traditional_uses = unique_join([existing.traditional_uses, source.traditional_uses])
        existing.folk_uses = unique_join([existing.folk_uses, source.folk_uses])
        existing.dosage_forms = unique_join([existing.dosage_forms, source.dosage_forms])
        existing.posology = unique_join([existing.posology, source.posology])
        existing.safety = unique_join([existing.safety, source.safety])
        existing.source_titles = unique_list([*existing.source_titles, *source.source_titles])
        existing.source_urls = unique_list([*existing.source_urls, *source.source_urls])

    for source in merged.values():
        if source.scientific_name:
            local_name = legacy_name_map.get(normalize_key(source.scientific_name), "")
            if local_name:
                source.display_name = local_name
    return sorted(merged.values(), key=lambda item: (item.display_name.lower(), item.scientific_name.lower()))


def build_profile_matches(sources: list[HerbSource]) -> dict[str, list[tuple[SymptomProfile, str]]]:
    matches: dict[str, list[tuple[SymptomProfile, str]]] = {}
    for source in sources:
        if not source.recommendation_allowed():
            matches[source.key] = []
            continue
        scored: list[tuple[int, SymptomProfile, str]] = []
        for profile in PROFILES:
            supported_match = contains_any(source.supported_uses, profile.keywords)
            traditional_match = contains_any(source.traditional_uses, profile.keywords)
            folk_match = contains_any(source.folk_uses, profile.keywords)
            if not any((supported_match, traditional_match, folk_match)):
                continue
            if supported_match:
                scored.append((3, profile, "high"))
            elif traditional_match:
                scored.append((2, profile, "medium"))
            elif folk_match:
                scored.append((1, profile, "low_to_medium"))

        scored.sort(key=lambda item: (-item[0], item[1].label))
        deduped: list[tuple[SymptomProfile, str]] = []
        seen_profiles: set[str] = set()
        for score, profile, evidence in scored:
            if profile.id in seen_profiles:
                continue
            seen_profiles.add(profile.id)
            deduped.append((profile, evidence))
            if len(deduped) >= MAX_TARGETS_PER_HERB:
                break
        matches[source.key] = deduped
    return matches


def build_herb_rows(
    sources: list[HerbSource],
    profile_matches: dict[str, list[tuple[SymptomProfile, str]]],
    legacy_herbs: list[dict[str, str]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen_keys: set[str] = set()
    counter = 1

    legacy_by_key = {normalize_key(f"{row.get('nama_lokal', '')}|{row.get('nama_latin', '')}"): row for row in legacy_herbs}

    for source in sources:
        matches = profile_matches.get(source.key, [])
        if not matches:
            continue
        symptom_labels = [profile.label for profile, _ in matches]
        evidence = strongest_evidence([item[1] for item in matches])
        row = {
            "id": f"herb_{counter:04d}",
            "nama_lokal": source.display_name,
            "nama_latin": source.scientific_name or first_alias(source.aliases),
            "bagian_digunakan": source.part_used or "bagian tanaman tidak dispesifikkan",
            "gejala_target": ";".join(symptom_labels),
            "konteks_penyakit_tropis": ";".join(symptom_labels),
            "ringkasan_manfaat": (
                f"{source.display_name} memiliki rujukan WHO/NCCIH untuk penggunaan terkait "
                f"{', '.join(symptom_labels)} sebagai pendamping keluhan ringan, bukan terapi penyakit berat."
            ),
            "evidence_level": evidence,
            "catatan_keamanan": build_safety_note(source, None),
            "sumber_utama": "; ".join(source.source_titles[:3]),
        }
        legacy_key = normalize_key(f"{row['nama_lokal']}|{row['nama_latin']}")
        legacy = legacy_by_key.get(legacy_key)
        if legacy:
            row["nama_lokal"] = legacy.get("nama_lokal", row["nama_lokal"]) or row["nama_lokal"]
            row["gejala_target"] = merge_semicolon_values(row["gejala_target"], legacy.get("gejala_target", ""))
            row["ringkasan_manfaat"] = unique_join([legacy.get("ringkasan_manfaat", ""), row["ringkasan_manfaat"]])
            row["catatan_keamanan"] = unique_join([row["catatan_keamanan"], legacy.get("catatan_keamanan", "")])
            row["sumber_utama"] = merge_semicolon_values(row["sumber_utama"], legacy.get("sumber_utama", ""))
        row_key = normalize_key(f"{row['nama_lokal']}|{row['nama_latin']}")
        if row_key in seen_keys:
            continue
        seen_keys.add(row_key)
        rows.append(row)
        counter += 1

    for legacy in legacy_herbs:
        row_key = normalize_key(f"{legacy.get('nama_lokal', '')}|{legacy.get('nama_latin', '')}")
        if row_key in seen_keys:
            continue
        copied = dict(legacy)
        copied["id"] = f"herb_{counter:04d}"
        rows.append(copied)
        seen_keys.add(row_key)
        counter += 1

    return rows


def build_formula_rows(
    sources: list[HerbSource],
    profile_matches: dict[str, list[tuple[SymptomProfile, str]]],
    legacy_formulas: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rows: list[dict[str, str]] = []
    contexts: list[dict[str, str]] = []
    seen_formula_keys: set[str] = set()
    counter = 1

    for legacy in legacy_formulas:
        copied = dict(legacy)
        copied["id"] = f"formula_{counter:04d}"
        rows.append(copied)
        seen_formula_keys.add(normalize_key(copied.get("nama_formula", "")))
        counter += 1

    for source in sources:
        for profile, evidence in profile_matches.get(source.key, []):
            formula_name = f"{source.display_name} untuk {profile.label}"
            formula_key = normalize_key(formula_name)
            if formula_key in seen_formula_keys:
                continue
            row = {
                "id": f"formula_{counter:04d}",
                "nama_formula": formula_name,
                "komposisi": source.display_name,
                "gejala_target": profile.label,
                "konteks_penggunaan": profile.context,
                "ringkasan_manfaat": (
                    f"Monograf WHO/NCCIH menempatkan {source.display_name} sebagai pendamping "
                    f"keluhan {profile.label} pada konteks yang masih ringan dan aman disaring dengan anamnesis."
                ),
                "evidence_level": evidence,
                "traditional_formula_inference": "false",
                "catatan_keamanan": build_safety_note(source, profile),
                "sumber_utama": "; ".join(source.source_titles[:3]),
            }
            rows.append(row)
            contexts.append(
                {
                    "formula_id": row["id"],
                    "formula_name": formula_name,
                    "display_name": source.display_name,
                    "scientific_name": source.scientific_name,
                    "profile_id": profile.id,
                    "evidence_level": evidence,
                    "source_titles": "; ".join(source.source_titles[:3]),
                    "source_key": source.key,
                }
            )
            seen_formula_keys.add(formula_key)
            counter += 1

    return rows, contexts


def build_case_rows(
    formula_contexts: list[dict[str, str]],
    preparation_overrides: dict[str, dict[str, str]],
    legacy_cases: list[dict[str, str]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen_case_keys: set[str] = set()
    counter = 1

    for legacy in legacy_cases:
        copied = dict(legacy)
        copied["id"] = f"case_{counter:04d}"
        rows.append(copied)
        case_key = normalize_key(f"{copied.get('keluhan_ringan', '')}|{copied.get('ramuan_rekomendasi', '')}")
        seen_case_keys.add(case_key)
        counter += 1

    for context in formula_contexts:
        profile = PROFILE_LOOKUP[context["profile_id"]]
        prep, dose = build_preparation_and_dose(
            display_name=context["display_name"],
            scientific_name=context["scientific_name"],
            preferred_mode=profile.preferred_mode,
            preparation_overrides=preparation_overrides,
        )
        for keluhan in profile.case_variants[:CASE_VARIANTS_PER_TARGET]:
            case_key = normalize_key(f"{keluhan}|{context['formula_name']}")
            if case_key in seen_case_keys:
                continue
            rows.append(
                {
                    "id": f"case_{counter:04d}",
                    "keluhan_ringan": keluhan,
                    "pertanyaan_anamnesis": profile.anamnesis_question,
                    "ramuan_rekomendasi": context["formula_name"],
                    "bahan": context["display_name"],
                    "cara_pengolahan": prep,
                    "dosis_penggunaan": dose,
                    "catatan_kewaspadaan": (
                        f"Jika ada {profile.red_flags}, hentikan self-care dan arahkan ke tenaga kesehatan. "
                        "Gunakan hanya untuk keluhan ringan dan hentikan bila muncul efek samping."
                    ),
                    "sumber_ringkas": context["source_titles"],
                }
            )
            seen_case_keys.add(case_key)
            counter += 1
    return rows


def build_generated_source_rows(
    sources: list[HerbSource],
    profile_matches: dict[str, list[tuple[SymptomProfile, str]]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source in sources:
        rows.append(
            {
                "id": source.key,
                "display_name": source.display_name,
                "scientific_name": source.scientific_name,
                "part_used": source.part_used,
                "aliases": sorted(source.aliases),
                "profiles": [
                    {"label": profile.label, "evidence_level": evidence}
                    for profile, evidence in profile_matches.get(source.key, [])
                ],
                "supported_uses": source.supported_uses,
                "traditional_uses": source.traditional_uses,
                "folk_uses": source.folk_uses,
                "dosage_forms": source.dosage_forms,
                "posology": source.posology,
                "safety": source.safety,
                "source_titles": source.source_titles,
                "source_urls": source.source_urls,
            }
        )
    return rows


def build_training_records(
    case_rows: list[dict[str, str]],
    formula_rows: list[dict[str, str]],
    herb_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for row in case_rows:
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
                "source_url": str(CASES_CSV_PATH),
                "curation_method": "generated_runtime_case_catalog",
            }
        )
    for row in formula_rows:
        records.append(
            {
                "id": f"formula_{row['id']}",
                "topic": row["gejala_target"],
                "formula_name": row["nama_formula"],
                "ingredients": split_semicolon(row["komposisi"]),
                "symptoms": split_semicolon(row["gejala_target"]),
                "preparation": (
                    "Ikuti pengolahan konservatif sesuai bahan dan konteks keluhan ringan; "
                    "untuk produk terstandar ikuti etiket."
                ),
                "dosage": "Mulai dari porsi kecil atau ikuti etiket produk terstandar; hentikan bila muncul efek samping.",
                "safety_notes": row["catatan_keamanan"],
                "evidence_level": row["evidence_level"],
                "source_title": row["sumber_utama"],
                "source_url": str(FORMULAS_CSV_PATH),
                "curation_method": "generated_runtime_formula_catalog",
            }
        )
    for row in herb_rows:
        records.append(
            {
                "id": f"herb_{row['id']}",
                "topic": row["gejala_target"],
                "formula_name": row["nama_lokal"],
                "ingredients": [row["nama_lokal"]],
                "symptoms": split_semicolon(row["gejala_target"]),
                "preparation": (
                    "Gunakan sebagai konteks bahan herbal; rekomendasi praktis tetap mengikuti "
                    "case/formula yang memiliki pengolahan lebih jelas."
                ),
                "dosage": "Tidak ada dosis tunggal baku untuk semua sediaan; prioritaskan bentuk terstandar atau porsi konservatif.",
                "safety_notes": row["catatan_keamanan"],
                "evidence_level": row["evidence_level"],
                "source_title": row["sumber_utama"],
                "source_url": str(HERBS_CSV_PATH),
                "curation_method": "generated_runtime_herb_catalog",
            }
        )
    return records


def build_sft_examples(records: list[dict[str, object]]) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    for record in records:
        symptoms = ", ".join(as_list(record.get("symptoms"))) or str(record.get("topic") or "keluhan ringan")
        formula_name = str(record.get("formula_name") or "ramuan herbal")
        examples.append(
            {
                "id": f"sft_{record['id']}",
                "source_record_id": record["id"],
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Saya mengalami {symptoms}. Tolong lakukan anamnesis singkat "
                            "dan rekomendasikan ramuan herbal bila masih ringan."
                        ),
                    },
                    {"role": "assistant", "content": assistant_answer(record)},
                ],
            }
        )
        examples.append(
            {
                "id": f"sft_dosis_{record['id']}",
                "source_record_id": record["id"],
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Bagaimana cara membuat dan dosis {formula_name}?"},
                    {"role": "assistant", "content": assistant_answer(record, focus="dosis")},
                ],
            }
        )
    return examples


def assistant_answer(record: dict[str, object], focus: str = "full") -> str:
    ingredients = ", ".join(as_list(record.get("ingredients"))) or "-"
    symptoms = ", ".join(as_list(record.get("symptoms"))) or str(record.get("topic") or "-")
    opening = (
        "Sebelum rekomendasi, pastikan keluhan masih ringan: tidak ada sesak, perdarahan, "
        "dehidrasi, demam tinggi berkepanjangan, nyeri berat, atau gejala yang cepat memburuk. "
    )
    if focus == "dosis":
        opening = ""
    return (
        f"{opening}Untuk konteks {symptoms}, ramuan yang dapat dipertimbangkan adalah "
        f"{record.get('formula_name')}.\n"
        f"Bahan: {ingredients}.\n"
        f"Cara pengolahan: {record.get('preparation')}.\n"
        f"Dosis/kisaran: {record.get('dosage')}.\n"
        f"Catatan kewaspadaan: {record.get('safety_notes')}.\n"
        f"Sumber/kurasi: {record.get('source_title')}.\n"
        "Informasi ini bersifat edukasi dan rekomendasi awal, bukan diagnosis medis final "
        "dan bukan pengganti konsultasi tenaga kesehatan."
    )


def refresh_combined_sft() -> None:
    rows: list[dict[str, object]] = []
    for path in sorted(TRAINING_DIR.glob("*_training_sft.jsonl")):
        if path.name == COMBINED_TRAINING_SFT_PATH.name:
            continue
        with path.open(encoding="utf-8") as file:
            rows.extend(json.loads(line) for line in file if line.strip())
    write_jsonl(COMBINED_TRAINING_SFT_PATH, rows)


def refresh_combined_sft_rag() -> None:
    base_rows = load_jsonl(COMBINED_TRAINING_SFT_PATH)
    rag_rows = load_jsonl(RAG_SFT_PATH)
    write_jsonl(COMBINED_TRAINING_SFT_RAG_PATH, [*base_rows, *rag_rows])


def build_preparation_and_dose(
    *,
    display_name: str,
    scientific_name: str,
    preferred_mode: str,
    preparation_overrides: dict[str, dict[str, str]],
) -> tuple[str, str]:
    override = find_preparation_override(display_name, scientific_name, preparation_overrides)
    if override:
        return override["langkah_pengolahan"], override["dosis_saran"]

    if preferred_mode == "topical":
        return (
            "Gunakan sediaan luar yang sudah higienis (gel, salep, kompres, atau infus dingin) pada area terbatas dan jangan pada luka dalam atau luas.",
            "Oles/kompres tipis 1-2 kali sehari pada area terbatas; hentikan bila iritasi bertambah.",
        )
    if preferred_mode == "gargle":
        return (
            "Seduh simplisia kering atau rebus ringan bahan yang sesuai, tunggu hangat, lalu gunakan untuk minum perlahan atau berkumur sesuai toleransi.",
            "Gunakan 100-150 ml sekali pakai, 1-2 kali sehari, dan hentikan bila muncul iritasi.",
        )
    return (
        "Cuci bahan, gunakan simplisia atau bentuk herbal yang bersih, lalu seduh/rebus ringan 5-15 menit sesuai kekerasan bahan sebelum disaring.",
        "Mulai dari 1 cangkir kecil 1 kali sehari; utamakan porsi konservatif atau ikuti etiket produk terstandar.",
    )


def find_preparation_override(
    display_name: str,
    scientific_name: str,
    preparation_overrides: dict[str, dict[str, str]],
) -> dict[str, str] | None:
    keys = {
        normalize_key(display_name),
        normalize_key(scientific_name),
    }
    for key in keys:
        if key and key in preparation_overrides:
            return preparation_overrides[key]
    return None


def load_preparation_overrides(path: Path) -> dict[str, dict[str, str]]:
    overrides: dict[str, dict[str, str]] = {}
    for row in load_csv_rows(path):
        key = normalize_key(row.get("nama_item", ""))
        if not key:
            continue
        overrides[key] = {
            "langkah_pengolahan": row.get("langkah_pengolahan", "").strip(),
            "dosis_saran": row.get("dosis_saran", "").strip(),
        }
    return overrides


def extract_common_name(vernacular_section: str) -> str:
    match = re.search(r'most commonly known as [“"]([^”"]+)[”"]', vernacular_section, flags=re.IGNORECASE)
    if match:
        return clean_text(match.group(1))
    if "known as" in vernacular_section.lower():
        fragment = vernacular_section.split(".", 1)[0]
        fragment = re.sub(r".*known as", "", fragment, flags=re.IGNORECASE).strip(" :")
        if fragment:
            return clean_text(fragment)
    return ""


def extract_scientific_name(definition: str) -> str:
    match = re.search(r"of ([A-Z][a-z]+ [a-z][a-z\-]+(?: [A-Za-z().\-]+)?)", definition)
    if not match:
        return ""
    return clean_text(match.group(1))


def extract_section(text: str, heading: str, next_headings: Iterable[str]) -> str:
    start = text.find(f"\n{heading}\n")
    if start == -1:
        return ""
    start += len(heading) + 2
    end = len(text)
    for next_heading in next_headings:
        marker = f"\n{next_heading}\n"
        position = text.find(marker, start)
        if position != -1 and position < end:
            end = position
    return clean_text(text[start:end])


def derive_part_used(title: str, definition: str) -> str:
    if title == "Aloe Vera Gel":
        return "gel/gel"
    for prefix, label in TITLE_PART_MAP.items():
        if title.startswith(prefix):
            return label
    lowered = definition.lower()
    if "leaf" in lowered:
        return "daun/leaf"
    if "root" in lowered:
        return "akar/root"
    if "rhizome" in lowered:
        return "rimpang/rhizome"
    return "bagian tanaman/herbal part"


def derive_part_from_background(background: str) -> str:
    lowered = background.lower()
    match = re.search(r"its ([a-z \-]+?) is used", lowered)
    if not match:
        match = re.search(r"their ([a-z \-]+?) are used", lowered)
    if not match:
        return "bagian tanaman/herbal part"
    value = clean_text(match.group(1))
    if "rhizome" in value:
        return "rimpang/rhizome"
    if "root" in value:
        return "akar/root"
    if "leaf" in value:
        return "daun/leaf"
    if "flower" in value:
        return "bunga/flower"
    if "fruit" in value:
        return "buah/fruit"
    if "seed" in value:
        return "biji/seed"
    if "oil" in value:
        return "minyak/oil"
    return value or "bagian tanaman/herbal part"


def contains_any(text: str, keywords: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def build_safety_note(source: HerbSource, profile: SymptomProfile | None) -> str:
    snippets = []
    if profile:
        snippets.append(f"Jika ada {profile.red_flags}, jangan lanjutkan self-care herbal.")
    if source.safety:
        snippets.append(limit_text(source.safety, 260))
    snippets.append("Hentikan penggunaan bila muncul reaksi alergi atau keluhan memburuk.")
    return unique_join(snippets)


def strongest_evidence(values: list[str]) -> str:
    if "high" in values:
        return "high"
    if "medium" in values:
        return "medium"
    return values[0] if values else "low_to_medium"


def profile_breakdown(formula_contexts: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in formula_contexts:
        label = PROFILE_LOOKUP[item["profile_id"]].label
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items(), key=lambda pair: (-pair[1], pair[0])))


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return [dict(row) for row in csv.DictReader(file)]


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def fetch_text(url: str) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        return response.read().decode("utf-8", "ignore")


def fetch_json(url: str) -> dict[str, object]:
    return json.loads(fetch_text(url))


def normalize_who_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\r", "")
    normalized = normalized.replace("\u00ad", "")
    return normalized


def html_to_text(value: str) -> str:
    text = value.replace("</li>", "; ").replace("<li>", " ")
    text = re.sub(r"<[^>]+>", " ", text)
    return clean_text(html.unescape(text))


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def split_names(value: str) -> list[str]:
    separators = re.split(r";|,|\band\b", clean_text(value), flags=re.IGNORECASE)
    return [item.strip() for item in separators if item.strip()]


def split_semicolon(value: str) -> list[str]:
    return [item.strip() for item in (value or "").split(";") if item.strip()]


def as_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return split_semicolon(value)
    return []


def merge_semicolon_values(left: str, right: str) -> str:
    return ";".join(unique_list([*split_semicolon(left), *split_semicolon(right)]))


def unique_join(values: Iterable[str]) -> str:
    cleaned: list[str] = []
    for value in values:
        item = clean_text(value)
        if not item:
            continue
        if item not in cleaned:
            cleaned.append(item)
    return " ".join(cleaned).strip()


def unique_list(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = clean_text(str(value))
        key = item.lower()
        if not item or key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def limit_text(value: str, max_chars: int) -> str:
    text = clean_text(value)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def herb_key(scientific_name: str, display_name: str) -> str:
    return normalize_key(scientific_name or display_name)


def normalize_key(value: str) -> str:
    text = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def localize_display_name(common_name: str, scientific_name: str) -> str:
    candidates = [
        clean_text(scientific_name).lower(),
        normalize_key(scientific_name),
        clean_text(common_name).lower(),
        normalize_key(common_name),
    ]
    for candidate in candidates:
        if candidate in LOCAL_NAME_OVERRIDES:
            return LOCAL_NAME_OVERRIDES[candidate]
    return common_name.strip() or scientific_name.strip()


def prefer_local_name(current: str, incoming: str, legacy: str) -> str:
    if legacy:
        return legacy
    if current and current.lower() != current:
        return current
    return current or incoming


def first_alias(aliases: set[str]) -> str:
    return sorted(clean_text(alias) for alias in aliases if clean_text(alias))[0]


if __name__ == "__main__":
    raise SystemExit(main())
