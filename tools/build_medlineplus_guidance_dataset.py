from __future__ import annotations

import html
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = ROOT / "data" / "traning"
CACHE_DIR = TRAINING_DIR / "source_cache"
SOURCES_PATH = TRAINING_DIR / "medlineplus_guidance_sources.jsonl"
RECORDS_PATH = TRAINING_DIR / "medlineplus_guidance_training_records.jsonl"
SFT_PATH = TRAINING_DIR / "medlineplus_guidance_training_sft.jsonl"
MANIFEST_PATH = TRAINING_DIR / "medlineplus_guidance_manifest.json"
COMBINED_SFT_PATH = TRAINING_DIR / "combined_training_sft.jsonl"

XML_INDEX_URL = "https://medlineplus.gov/xml.html"
XML_FILE_PATTERN = re.compile(r"https://medlineplus\.gov/xml/(mplus_topics_(\d{4}-\d{2}-\d{2})\.xml)")

ALLOWED_GROUPS = {
    "Bones, Joints and Muscles",
    "Digestive System",
    "Ear, Nose and Throat",
    "Eyes and Vision",
    "Female Reproductive System",
    "Immune System",
    "Infections",
    "Kidneys and Urinary System",
    "Lungs and Breathing",
    "Mental Health and Behavior",
    "Mouth and Teeth",
    "Poisoning, Toxicology, Environmental Health",
    "Pregnancy and Reproduction",
    "Skin, Hair and Nails",
    "Symptoms",
}

EXCLUDED_TITLE_PATTERNS = (
    "a1c",
    "anatomy",
    "antibiotics",
    "birth control",
    "blood transfusion",
    "bone marrow transplantation",
    "brain aneurysm",
    "brain tumors",
    "breast reconstruction",
    "cancer",
    "clinical trials",
    "colostomy",
    "cosmetic",
    "dialysis",
    "doctor or health care provider",
    "drug reactions",
    "eating disorders",
    "electrolytes",
    "fluoride",
    "genetics",
    "health checkup",
    "hiv",
    "hospital",
    "immunization",
    "insurance",
    "intensive care",
    "kidney failure",
    "leukemia",
    "liver transplantation",
    "lymphoma",
    "medical encyclopedia",
    "medicine",
    "medicines",
    "organ donation",
    "organ transplant",
    "palliative care",
    "pregnancy",
    "rehabilitation",
    "screening",
    "sepsis",
    "sexual assault",
    "surgery",
    "test",
    "transplant",
    "vaccines",
)

VARIANT_LABELS = (
    "overview",
    "triage",
    "follow_up",
    "prevention",
    "self_care",
)

GROUP_TRANSLATIONS = {
    "Bones, Joints and Muscles": "tulang, sendi, dan otot",
    "Digestive System": "sistem pencernaan",
    "Ear, Nose and Throat": "telinga, hidung, dan tenggorokan",
    "Eyes and Vision": "mata dan penglihatan",
    "Female Reproductive System": "reproduksi perempuan",
    "Immune System": "imunitas dan alergi",
    "Infections": "infeksi",
    "Kidneys and Urinary System": "ginjal dan saluran kemih",
    "Lungs and Breathing": "paru dan pernapasan",
    "Mental Health and Behavior": "kesehatan mental dan perilaku",
    "Mouth and Teeth": "mulut dan gigi",
    "Poisoning, Toxicology, Environmental Health": "paparan lingkungan dan toksikologi",
    "Pregnancy and Reproduction": "kehamilan dan reproduksi",
    "Skin, Hair and Nails": "kulit, rambut, dan kuku",
    "Symptoms": "gejala umum",
}

GROUP_DEFAULT_SYMPTOMS = {
    "Bones, Joints and Muscles": ["nyeri otot atau sendi", "bengkak", "kaku", "gangguan gerak"],
    "Digestive System": ["nyeri perut", "mual", "muntah", "diare atau sembelit"],
    "Ear, Nose and Throat": ["nyeri tenggorokan", "pilek", "nyeri telinga", "hidung tersumbat"],
    "Eyes and Vision": ["mata merah", "nyeri mata", "penglihatan kabur", "gatal atau berair"],
    "Female Reproductive System": ["keputihan", "nyeri panggul", "haid tidak nyaman", "gatal area intim"],
    "Immune System": ["ruam", "gatal", "bersin", "bengkak atau reaksi alergi"],
    "Infections": ["demam", "lemas", "nyeri", "gejala infeksi sesuai organ"],
    "Kidneys and Urinary System": ["nyeri saat BAK", "sering BAK", "nyeri pinggang", "urine keruh atau berdarah"],
    "Lungs and Breathing": ["batuk", "pilek", "sesak napas", "mengi"],
    "Mental Health and Behavior": ["cemas", "sulit tidur", "murung", "stres"],
    "Mouth and Teeth": ["sariawan", "gusi bengkak", "nyeri gigi", "bau mulut"],
    "Poisoning, Toxicology, Environmental Health": ["pusing", "mual", "iritasi kulit", "sesak setelah paparan"],
    "Pregnancy and Reproduction": ["mual", "nyeri panggul", "perdarahan", "keluhan reproduksi"],
    "Skin, Hair and Nails": ["ruam", "gatal", "kemerahan", "kulit bersisik"],
    "Symptoms": ["demam", "nyeri", "lemas", "gejala yang perlu dipertajam"],
}

GROUP_DEFAULT_WARNINGS = {
    "Bones, Joints and Muscles": ["nyeri hebat", "bengkak berat", "tidak bisa digerakkan", "demam tinggi"],
    "Digestive System": ["nyeri perut hebat", "muntah terus", "BAB berdarah", "dehidrasi"],
    "Ear, Nose and Throat": ["sesak napas", "sulit menelan", "demam tinggi", "nyeri berat makin memburuk"],
    "Eyes and Vision": ["penglihatan turun mendadak", "nyeri hebat", "mata sangat merah", "trauma mata"],
    "Female Reproductive System": ["perdarahan banyak", "nyeri hebat", "demam tinggi", "pingsan"],
    "Immune System": ["sesak napas", "bengkak wajah atau bibir", "ruam cepat menyebar", "pusing berat"],
    "Infections": ["demam tinggi", "sesak napas", "sangat lemas", "tanda dehidrasi"],
    "Kidneys and Urinary System": ["urine berdarah", "demam tinggi", "nyeri pinggang hebat", "sulit BAK"],
    "Lungs and Breathing": ["sesak napas", "bibir kebiruan", "nyeri dada", "demam tinggi menetap"],
    "Mental Health and Behavior": ["ingin menyakiti diri", "bingung berat", "sulit berfungsi", "tidak tidur sama sekali"],
    "Mouth and Teeth": ["bengkak menjalar ke wajah", "demam tinggi", "sulit menelan", "nyeri berat"],
    "Poisoning, Toxicology, Environmental Health": ["sesak napas", "kejang", "pingsan", "paparan bahan berbahaya"],
    "Pregnancy and Reproduction": ["perdarahan", "nyeri hebat", "pusing berat", "demam tinggi"],
    "Skin, Hair and Nails": ["lepuh luas", "nanah", "ruam cepat menyebar", "bengkak wajah atau sesak"],
    "Symptoms": ["sesak napas", "nyeri hebat", "perdarahan", "demam tinggi atau memburuk"],
}

GROUP_DEFAULT_PREVENTION = {
    "Bones, Joints and Muscles": ["kurangi aktivitas pemicu", "jaga postur", "kompres sesuai keluhan", "periksa bila memburuk"],
    "Digestive System": ["cukup minum", "makan ringan sesuai toleransi", "jaga kebersihan makanan", "periksa bila gejala berat"],
    "Ear, Nose and Throat": ["cukup istirahat", "cukup cairan", "hindari asap dan iritan", "periksa bila makin berat"],
    "Eyes and Vision": ["jaga kebersihan tangan", "hindari menggosok mata", "batasi lensa kontak saat iritasi", "periksa bila nyeri atau penglihatan turun"],
    "Female Reproductive System": ["jaga kebersihan area intim", "hindari pencetus iritasi", "catat pola gejala", "periksa bila berulang atau berat"],
    "Immune System": ["hindari pencetus", "siapkan obat dokter bila ada riwayat alergi", "catat reaksi berulang", "cari bantuan bila ada sesak"],
    "Infections": ["istirahat cukup", "cukup minum", "jaga kebersihan tangan", "hindari menularkan ke orang lain"],
    "Kidneys and Urinary System": ["cukup minum", "jangan menahan BAK", "jaga kebersihan area genital", "periksa bila ada demam atau darah"],
    "Lungs and Breathing": ["cukup istirahat", "cukup cairan", "hindari asap rokok", "gunakan masker bila perlu"],
    "Mental Health and Behavior": ["atur tidur", "kurangi stresor", "cari dukungan sosial", "hubungi tenaga profesional bila berat"],
    "Mouth and Teeth": ["jaga kebersihan mulut", "batasi pemicu iritasi", "gunakan alat pribadi", "periksa bila nyeri atau bengkak"],
    "Poisoning, Toxicology, Environmental Health": ["hindari sumber paparan", "pakai pelindung sesuai risiko", "ventilasi ruangan", "cari bantuan bila gejala akut"],
    "Pregnancy and Reproduction": ["catat gejala", "cukup istirahat", "hindari swamedikasi berisiko", "periksa bila ada perdarahan atau nyeri hebat"],
    "Skin, Hair and Nails": ["jaga kulit tetap bersih dan kering", "hindari garukan", "hindari pencetus iritasi", "periksa bila bernanah atau meluas"],
    "Symptoms": ["catat durasi gejala", "cukup istirahat", "cukup cairan", "periksa bila ada tanda bahaya"],
}

PHRASE_TRANSLATIONS = {
    "abdominal pain": "nyeri perut",
    "acute bronchitis": "bronkitis akut",
    "acute sinusitis": "sinusitis akut",
    "air pollution": "paparan polusi udara",
    "allergic reaction": "reaksi alergi",
    "animal bites": "gigitan hewan",
    "athlete's foot": "jamur kaki",
    "back pain": "nyeri punggung",
    "bad breath": "bau mulut",
    "bladder infection": "infeksi kandung kemih",
    "bladder control problems": "gangguan kontrol BAK",
    "blood in stool": "BAB berdarah",
    "body lice": "kutu badan",
    "breathing problems": "gangguan pernapasan",
    "burning with urination": "nyeri saat BAK",
    "chest pain": "nyeri dada",
    "common cold": "flu atau pilek",
    "conjunctivitis": "konjungtivitis",
    "contact dermatitis": "dermatitis kontak",
    "constipation": "sembelit",
    "coughing up blood": "batuk darah",
    "ear infections": "infeksi telinga",
    "ear pain": "nyeri telinga",
    "food poisoning": "keracunan makanan",
    "frequent urination": "sering BAK",
    "gastroesophageal reflux": "refluks asam lambung",
    "headache": "sakit kepala",
    "heat exhaustion": "kelelahan akibat panas",
    "heat stroke": "serangan panas",
    "hives": "biduran",
    "indigestion": "dispepsia atau maag",
    "insect bites and stings": "gigitan atau sengatan serangga",
    "itching": "gatal",
    "lower back pain": "nyeri punggung bawah",
    "motion sickness": "mabuk perjalanan",
    "nausea and vomiting": "mual dan muntah",
    "pink eye": "mata merah",
    "red eyes": "mata merah",
    "runny nose": "pilek",
    "sore throat": "nyeri tenggorokan",
    "shortness of breath": "sesak napas",
    "skin infections": "infeksi kulit",
    "stomach ache": "sakit perut",
    "stomach flu": "muntah diare",
    "sunburn": "kulit terbakar matahari",
    "urinary tract infections": "infeksi saluran kemih",
    "urination": "BAK",
    "vaginal yeast infections": "infeksi jamur vagina",
    "vomiting blood": "muntah darah",
    "watery eyes": "mata berair",
    "whooping cough": "batuk rejan",
    "you have abdominal pain that is sudden and sharp": "nyeri perut muncul mendadak dan tajam",
    "you also have pain in your chest, neck or shoulder": "nyeri menjalar ke dada, leher, atau bahu",
    "you're vomiting blood or have blood in your stool": "muntah darah atau BAB berdarah",
    "your abdomen is stiff, hard and tender to touch": "perut kaku, keras, dan sangat nyeri saat disentuh",
    "you can't move your bowels, especially if you're also vomiting": "tidak bisa BAB, terutama bila juga muntah",
    "difficulty breathing": "sulit bernapas",
    "trouble breathing": "gangguan bernapas",
    "severe pain": "nyeri hebat",
    "high fever": "demam tinggi",
    "blood in your stool": "BAB berdarah",
    "blood in your urine": "urine berdarah",
    "loss of vision": "penurunan penglihatan",
}

TOKEN_TRANSLATIONS = {
    "also": "juga",
    "acute": "akut",
    "allergy": "alergi",
    "anxiety": "kecemasan",
    "arthritis": "radang sendi",
    "asthma": "asma",
    "bacterial": "bakteri",
    "bladder": "kandung kemih",
    "bleeding": "perdarahan",
    "blisters": "lepuh",
    "bronchitis": "bronkitis",
    "burns": "luka bakar",
    "cannot": "tidak bisa",
    "cant": "tidak bisa",
    "chest": "dada",
    "chills": "menggigil",
    "chronic": "kronis",
    "cough": "batuk",
    "dandruff": "ketombe",
    "dehydration": "dehidrasi",
    "depression": "depresi",
    "diarrhea": "diare",
    "dizziness": "pusing",
    "ear": "telinga",
    "eczema": "eksim",
    "eye": "mata",
    "fainting": "pingsan",
    "fever": "demam",
    "flu": "flu",
    "fungal": "jamur",
    "gums": "gusi",
    "hard": "keras",
    "hair": "rambut",
    "head": "kepala",
    "hearing": "pendengaran",
    "heartburn": "nyeri ulu hati",
    "infection": "infeksi",
    "infections": "infeksi",
    "itchy": "gatal",
    "joint": "sendi",
    "kidney": "ginjal",
    "lasting": "berlangsung",
    "lips": "bibir",
    "mouth": "mulut",
    "move": "bergerak",
    "muscle": "otot",
    "nail": "kuku",
    "neck": "leher",
    "nose": "hidung",
    "pain": "nyeri",
    "pelvic": "panggul",
    "persistent": "menetap",
    "rash": "ruam",
    "reflux": "refluks",
    "respiratory": "pernapasan",
    "scabies": "skabies",
    "sinusitis": "sinusitis",
    "skin": "kulit",
    "sleep": "tidur",
    "sharp": "tajam",
    "sprains": "keseleo",
    "stiff": "kaku",
    "stress": "stres",
    "sudden": "mendadak",
    "swelling": "bengkak",
    "swallowing": "menelan",
    "throat": "tenggorokan",
    "tender": "nyeri",
    "tooth": "gigi",
    "teeth": "gigi",
    "touch": "disentuh",
    "trouble": "gangguan",
    "urinary": "kemih",
    "urine": "urine",
    "vision": "penglihatan",
    "vomiting": "muntah",
    "warts": "kutil",
    "week": "minggu",
    "weeks": "minggu",
    "wheezing": "mengi",
    "wounds": "luka",
    "yeast": "jamur",
    "your": "",
    "you": "",
    "youre": "",
}

SYMPTOM_PATTERNS = (
    ("abdominal pain", "nyeri perut"),
    ("back pain", "nyeri punggung"),
    ("bad breath", "bau mulut"),
    ("bleeding", "perdarahan"),
    ("blister", "lepuh"),
    ("blood in stool", "BAB berdarah"),
    ("blood in urine", "urine berdarah"),
    ("burning with urination", "nyeri saat BAK"),
    ("chest pain", "nyeri dada"),
    ("confusion", "bingung"),
    ("constipation", "sembelit"),
    ("cough", "batuk"),
    ("dehydration", "dehidrasi"),
    ("diarrhea", "diare"),
    ("discharge", "cairan tidak biasa"),
    ("dizziness", "pusing"),
    ("ear pain", "nyeri telinga"),
    ("fatigue", "lelah"),
    ("fever", "demam"),
    ("frequent urination", "sering BAK"),
    ("headache", "sakit kepala"),
    ("itch", "gatal"),
    ("itchy", "gatal"),
    ("nausea", "mual"),
    ("pain", "nyeri"),
    ("rash", "ruam"),
    ("red eyes", "mata merah"),
    ("runny nose", "pilek"),
    ("shortness of breath", "sesak napas"),
    ("sore throat", "nyeri tenggorokan"),
    ("stiff neck", "kaku leher"),
    ("swelling", "bengkak"),
    ("urinating", "keluhan BAK"),
    ("vomiting", "muntah"),
    ("weakness", "lemas"),
    ("wheezing", "mengi"),
)

PREVENTION_RULES = (
    ("wash your hands", "Cuci tangan dengan sabun."),
    ("avoid smoking", "Hindari rokok dan asap rokok."),
    ("avoid smoke", "Hindari asap dan iritan."),
    ("drink plenty of fluids", "Perbanyak minum air sesuai toleransi."),
    ("drink enough fluids", "Perbanyak minum air sesuai toleransi."),
    ("get plenty of rest", "Istirahat yang cukup."),
    ("rest", "Istirahat yang cukup."),
    ("vaccin", "Lengkapi vaksinasi bila tersedia dan sesuai indikasi."),
    ("mask", "Gunakan masker bila ada risiko penularan atau paparan."),
    ("avoid sharing", "Hindari berbagi alat pribadi."),
    ("stay home", "Batasi aktivitas dan istirahat di rumah bila sedang sakit."),
    ("avoid contact", "Hindari kontak erat bila keluhan diduga menular."),
    ("keep the area clean", "Jaga area yang terkena tetap bersih."),
    ("keep your skin dry", "Jaga kulit tetap kering."),
    ("avoid scratching", "Hindari menggaruk area yang gatal."),
    ("drink water", "Cukupi kebutuhan cairan."),
    ("eat bland foods", "Pilih makanan ringan yang mudah ditoleransi."),
)


class SummaryHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.paragraphs: list[str] = []
        self.list_items: list[str] = []
        self._current: list[str] = []
        self._current_tag: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"p", "li"}:
            self._current = []
            self._current_tag = tag

    def handle_data(self, data: str) -> None:
        if self._current_tag in {"p", "li"}:
            self._current.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != self._current_tag:
            return
        text = normalize_text("".join(self._current))
        if text:
            if tag == "p":
                self.paragraphs.append(text)
            elif tag == "li":
                self.list_items.append(text)
        self._current = []
        self._current_tag = None


@dataclass(frozen=True)
class TopicSpec:
    topic_id: str
    title: str
    url: str
    meta_desc: str
    summary_html: str
    groups: list[str]
    related_topics: list[str]
    primary_institute: str


def main() -> int:
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    xml_path, xml_url, xml_date = ensure_latest_xml()
    topics = parse_topics(xml_path)
    sources = build_sources(topics, xml_url, xml_date)
    records = build_records(topics, xml_url)
    sft_rows = build_sft_rows(records)

    write_jsonl(SOURCES_PATH, sources)
    write_jsonl(RECORDS_PATH, records)
    write_jsonl(SFT_PATH, sft_rows)
    write_combined_sft()

    manifest = {
        "xml_url": xml_url,
        "xml_cache_path": str(xml_path),
        "xml_date": xml_date,
        "selected_topic_count": len(topics),
        "generated_record_count": len(records),
        "generated_sft_count": len(sft_rows),
        "variant_count_per_topic": len(VARIANT_LABELS),
        "allowed_groups": sorted(ALLOWED_GROUPS),
        "excluded_title_patterns": list(EXCLUDED_TITLE_PATTERNS),
        "group_breakdown": dict(Counter(group for topic in topics for group in topic.groups if group in ALLOWED_GROUPS)),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def ensure_latest_xml() -> tuple[Path, str, str]:
    index_html = run_curl(XML_INDEX_URL)
    match = XML_FILE_PATTERN.search(index_html)
    if not match:
        raise SystemExit("Tidak dapat menemukan file XML MedlinePlus terbaru dari halaman index.")

    xml_url = match.group(0)
    xml_date = match.group(2)
    xml_path = CACHE_DIR / f"medlineplus_topics_{xml_date}.xml"

    if not xml_path.exists():
        xml_content = run_curl(xml_url)
        xml_path.write_text(xml_content, encoding="utf-8")
    return xml_path, xml_url, xml_date


def run_curl(url: str) -> str:
    return subprocess.check_output(
        ["curl", "-L", "--max-time", "120", "-A", "Mozilla/5.0", "-s", url],
        text=True,
    )


def parse_topics(xml_path: Path) -> list[TopicSpec]:
    root = ET.parse(xml_path).getroot()
    topics: list[TopicSpec] = []

    for node in root.findall("health-topic"):
        if (node.get("language") or "").strip() != "English":
            continue

        title = normalize_text(node.get("title") or "")
        if not title or should_exclude_title(title):
            continue

        groups = unique_preserve_order(
            group.text.strip()
            for group in node.findall("group")
            if group.text and group.text.strip() and all(ord(char) < 128 for char in group.text)
        )
        relevant_groups = [group for group in groups if group in ALLOWED_GROUPS]
        if not relevant_groups:
            continue

        topic_id = (node.get("id") or title.lower()).strip()
        url = normalize_text(node.get("url") or "")
        meta_desc = normalize_text(node.get("meta-desc") or "")
        summary_html = html.unescape(node.findtext("full-summary") or "")
        related_topics = unique_preserve_order(
            related.text.strip()
            for related in node.findall("related-topic")
            if related.text and related.text.strip()
        )[:8]
        primary_institute = normalize_text(node.findtext("primary-institute") or "")

        topics.append(
            TopicSpec(
                topic_id=topic_id,
                title=title,
                url=url,
                meta_desc=meta_desc,
                summary_html=summary_html,
                groups=relevant_groups,
                related_topics=related_topics,
                primary_institute=primary_institute,
            )
        )

    return topics


def should_exclude_title(title: str) -> bool:
    normalized = normalize_text(title).lower()
    return any(pattern in normalized for pattern in EXCLUDED_TITLE_PATTERNS)


def build_sources(topics: list[TopicSpec], xml_url: str, xml_date: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        {
            "id": f"medlineplus_xml_{xml_date}",
            "title": "MedlinePlus Health Topic XML",
            "url": xml_url,
            "note": "Sumber resmi topik kesehatan MedlinePlus dalam format XML harian.",
        }
    ]
    for topic in topics:
        rows.append(
            {
                "id": f"medlineplus_{topic.topic_id}",
                "title": topic.title,
                "url": topic.url,
                "note": f"Topik MedlinePlus untuk {', '.join(topic.groups[:3])}.",
            }
        )
    return rows


def build_records(topics: list[TopicSpec], xml_url: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for topic in topics:
        summary = parse_summary(topic.summary_html)
        summary_text = " ".join([*summary.paragraphs[:4], *summary.list_items[:4]])
        overview_excerpt = limit_text(summary_text, 560)
        symptoms = derive_symptoms(topic, summary_text)
        warning_signs = derive_warning_signs(topic, summary, summary_text)
        prevention_steps = derive_prevention_steps(topic, summary_text)
        screening_questions = build_screening_questions(topic, symptoms, warning_signs)
        translated_title = translate_label(topic.title)
        display_title = build_display_title(topic.title, translated_title)
        group_text = ", ".join(GROUP_TRANSLATIONS.get(group, group.lower()) for group in topic.groups[:3])
        related_preview = ", ".join(translate_label(item) for item in topic.related_topics[:4])
        institute_text = topic.primary_institute or "MedlinePlus"
        safety_notes = (
            "Segera cari bantuan medis bila muncul tanda bahaya, gejala memburuk cepat, "
            "atau keluhan tidak sesuai kategori ringan."
        )

        for variant in VARIANT_LABELS:
            rows.append(
                {
                    "id": f"medlineplus_{topic.topic_id}_{variant}",
                    "topic": build_variant_topic(topic.title, translated_title, variant),
                    "formula_name": display_title,
                    "ingredients": [],
                    "symptoms": symptoms,
                    "preparation": "",
                    "dosage": "",
                    "safety_notes": safety_notes,
                    "evidence_level": "clinical_guideline_reference",
                    "source_title": f"MedlinePlus - {topic.title}",
                    "source_url": topic.url or xml_url,
                    "curation_method": "official_medlineplus_xml_template_expansion",
                    "content_type": "disease_guidance",
                    "overview": build_variant_overview(
                        variant=variant,
                        display_title=display_title,
                        group_text=group_text,
                        overview_excerpt=overview_excerpt,
                        related_preview=related_preview,
                    ),
                    "diagnosis_summary": (
                        f"Topik ini dikurasi dari MedlinePlus dan dikaitkan dengan kategori {group_text}. "
                        f"Institut rujukan utama: {institute_text}."
                    ),
                    "prevention_steps": prevention_steps,
                    "warning_signs": warning_signs,
                    "screening_questions": screening_questions,
                    "care_recommendation": build_care_recommendation(variant, display_title, warning_signs),
                }
            )
    return rows


def build_sft_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        formula_name = str(record.get("formula_name") or record.get("topic") or "topik kesehatan")
        variant = str(record.get("id") or "").rsplit("_", 1)[-1]
        system_prompt = build_system_prompt(variant)
        user_prompt = build_user_prompt(variant, formula_name)
        assistant_prompt = build_assistant_prompt(record)
        rows.append(
            {
                "id": record["id"],
                "source": "medlineplus_guidance_dataset",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_prompt},
                ],
            }
        )
    return rows


def build_system_prompt(variant: str) -> str:
    if variant == "follow_up":
        return (
            "Anda adalah chatbot edukasi kesehatan berbahasa Indonesia. "
            "Tugas Anda menyusun pertanyaan anamnesis yang tidak berulang, humanizing, dan aman."
        )
    if variant == "prevention":
        return (
            "Anda adalah chatbot edukasi kesehatan berbahasa Indonesia. "
            "Jelaskan pencegahan dan kapan perlu periksa tanpa mengklaim diagnosis final."
        )
    if variant == "self_care":
        return (
            "Anda adalah chatbot edukasi kesehatan berbahasa Indonesia. "
            "Jelaskan self-care ringan yang aman sambil menekankan red flag."
        )
    return (
        "Anda adalah chatbot edukasi kesehatan berbahasa Indonesia. "
        "Berikan ringkasan triase awal yang aman, singkat, dan tetap humanizing."
    )


def build_user_prompt(variant: str, formula_name: str) -> str:
    if variant == "follow_up":
        return f"Kalau keluhan pengguna mengarah ke {formula_name}, pertanyaan anamnesis lanjutan apa yang sebaiknya diajukan?"
    if variant == "prevention":
        return f"Bagaimana pencegahan dan kapan perlu periksa bila ada gejala yang mengarah ke {formula_name}?"
    if variant == "self_care":
        return f"Edukasi awal apa yang aman untuk kasus ringan yang mengarah ke {formula_name} sambil tetap waspada?"
    if variant == "triage":
        return f"Tolong bantu triase awal untuk keluhan yang mungkin terkait {formula_name}."
    return f"Saya ingin ringkasan gejala, tanda bahaya, dan langkah awal untuk {formula_name}."


def build_assistant_prompt(record: dict[str, object]) -> str:
    overview = str(record.get("overview") or "").strip()
    screening_questions = as_list(record.get("screening_questions"))
    warning_signs = as_list(record.get("warning_signs"))
    prevention_steps = as_list(record.get("prevention_steps"))
    care_recommendation = str(record.get("care_recommendation") or "").strip()

    blocks = [overview]
    if screening_questions:
        blocks.append(
            "Pertanyaan anamnesis yang perlu ditanyakan:\n"
            + "\n".join(f"{index}. {question}" for index, question in enumerate(screening_questions[:5], start=1))
        )
    if warning_signs:
        blocks.append(
            "Tanda bahaya yang perlu diwaspadai:\n"
            + "\n".join(f"- {item}" for item in warning_signs[:5])
        )
    if prevention_steps:
        blocks.append(
            "Langkah pencegahan atau edukasi awal:\n"
            + "\n".join(f"- {item}" for item in prevention_steps[:5])
        )
    if care_recommendation:
        blocks.append(care_recommendation)
    blocks.append(
        "Informasi ini bersifat edukasi awal, bukan diagnosis medis final dan bukan pengganti konsultasi tenaga kesehatan."
    )
    return "\n\n".join(block for block in blocks if block)


def build_variant_topic(title: str, translated_title: str, variant: str) -> str:
    label = {
        "overview": "ringkasan gejala dan konteks",
        "triage": "triase awal dan tanda bahaya",
        "follow_up": "pertanyaan anamnesis lanjutan",
        "prevention": "pencegahan dan edukasi",
        "self_care": "self-care ringan dan batas aman",
    }[variant]
    return f"{translated_title or title} - {label}"


def build_variant_overview(
    *,
    variant: str,
    display_title: str,
    group_text: str,
    overview_excerpt: str,
    related_preview: str,
) -> str:
    if variant == "follow_up":
        return (
            f"Topik {display_title} berada pada kategori {group_text}. "
            "Sebelum memberi kesimpulan, perlu klarifikasi gejala utama, durasi, tingkat gangguan aktivitas, dan red flag."
        )
    if variant == "prevention":
        return (
            f"Edukasi pencegahan untuk {display_title} perlu menyesuaikan kategori {group_text}. "
            f"Ringkasan sumber: {overview_excerpt}"
        )
    if variant == "self_care":
        return (
            f"Bila keluhan yang mengarah ke {display_title} masih ringan, edukasi awal dapat diberikan sambil tetap menilai red flag. "
            f"Topik terkait: {related_preview or 'belum ada topik terkait yang ditampilkan'}."
        )
    if variant == "triage":
        return (
            f"Triase awal {display_title} difokuskan pada durasi, progresivitas gejala, dan tanda bahaya. "
            f"Ringkasan sumber: {overview_excerpt}"
        )
    return (
        f"{display_title} dikaitkan dengan kategori {group_text}. "
        f"Ringkasan sumber resmi: {overview_excerpt}"
    )


def build_care_recommendation(variant: str, display_title: str, warning_signs: list[str]) -> str:
    warning_preview = ", ".join(warning_signs[:3]) or "tanda bahaya klinis"
    if variant == "follow_up":
        return (
            f"Gunakan maksimal tiga pertanyaan lanjutan yang paling relevan untuk {display_title}. "
            f"Bila jawaban mengarah ke {warning_preview}, arahkan pemeriksaan medis."
        )
    if variant == "prevention":
        return (
            f"Pencegahan dan monitoring mandiri hanya untuk konteks edukasi. "
            f"Perlu pemeriksaan langsung bila ada {warning_preview}."
        )
    if variant == "self_care":
        return (
            f"Self-care hanya untuk gejala ringan dan stabil. "
            f"Jika muncul {warning_preview}, hentikan swaperawatan dan cari bantuan medis."
        )
    return (
        f"Untuk {display_title}, prioritaskan anamnesis, evaluasi red flag, dan jangan memberikan klaim diagnosis final. "
        f"Periksa ke tenaga kesehatan bila ada {warning_preview}."
    )


@dataclass(frozen=True)
class ParsedSummary:
    paragraphs: list[str]
    list_items: list[str]


def parse_summary(summary_html: str) -> ParsedSummary:
    parser = SummaryHTMLParser()
    parser.feed(summary_html)
    return ParsedSummary(paragraphs=parser.paragraphs, list_items=parser.list_items)


def derive_symptoms(topic: TopicSpec, summary_text: str) -> list[str]:
    text = normalize_text(" ".join([topic.title, topic.meta_desc, summary_text, " ".join(topic.related_topics)])).lower()
    symptoms: list[str] = []
    for pattern, label in SYMPTOM_PATTERNS:
        if pattern in text:
            symptoms.append(label)
    for group in topic.groups:
        symptoms.extend(GROUP_DEFAULT_SYMPTOMS.get(group, []))

    translated_title = translate_label(topic.title)
    if translated_title and translated_title.lower() != topic.title.lower():
        symptoms.append(translated_title.lower())
    symptoms.append(topic.title.lower())
    return unique_preserve_order(item for item in symptoms if item)[:10]


def derive_warning_signs(topic: TopicSpec, summary: ParsedSummary, summary_text: str) -> list[str]:
    warning_signs: list[str] = []
    lower_html = topic.summary_html.lower()
    if any(
        phrase in lower_html
        for phrase in (
            "get medical help immediately if",
            "get medical help right away if",
            "seek medical care",
            "call your health care provider if",
            "call your provider if",
            "go to the emergency room",
        )
    ):
        for item in summary.list_items[:6]:
            translated = translate_sentence(item)
            if translated:
                warning_signs.append(translated)

    for item in summary.list_items[:6]:
        normalized = item.lower()
        if any(
            token in normalized
            for token in ("blood", "bleed", "breath", "severe", "high fever", "dehydration", "confusion", "stiff", "seizure")
        ):
            warning_signs.append(translate_sentence(item))

    if not warning_signs:
        for group in topic.groups:
            warning_signs.extend(GROUP_DEFAULT_WARNINGS.get(group, []))

    warning_signs = [normalize_text(item) for item in warning_signs if normalize_text(item)]
    return unique_preserve_order(warning_signs)[:6]


def derive_prevention_steps(topic: TopicSpec, summary_text: str) -> list[str]:
    lower_text = summary_text.lower()
    steps: list[str] = []
    for pattern, step in PREVENTION_RULES:
        if pattern in lower_text:
            steps.append(step)

    if not steps:
        for group in topic.groups:
            steps.extend(GROUP_DEFAULT_PREVENTION.get(group, []))

    return unique_preserve_order(steps)[:6]


def build_screening_questions(topic: TopicSpec, symptoms: list[str], warning_signs: list[str]) -> list[str]:
    preview_symptoms = ", ".join(symptoms[:3]) or "gejala utama"
    preview_warning = warning_signs[0] if warning_signs else "tanda bahaya"
    questions = [
        f"Sejak kapan keluhan yang mengarah ke {translate_label(topic.title) or topic.title} muncul dan apakah makin berat?",
        f"Apakah ada gejala seperti {preview_symptoms} yang paling mengganggu saat ini?",
        "Apakah keluhan mengganggu makan, minum, tidur, bernapas, berjalan, atau aktivitas harian?",
        f"Apakah ada tanda bahaya seperti {preview_warning}?",
    ]

    if "Infections" in topic.groups:
        questions.append("Apakah ada demam, kontak dengan orang sakit, atau keluhan serupa pada orang sekitar?")
    if "Skin, Hair and Nails" in topic.groups:
        questions.append("Apakah ada ruam meluas, bernanah, gatal berat, atau pencetus baru seperti sabun, obat, dan makanan?")
    if "Digestive System" in topic.groups:
        questions.append("Apakah ada mual, muntah, diare, sembelit, BAB berdarah, atau nyeri perut hebat?")
    if "Lungs and Breathing" in topic.groups:
        questions.append("Apakah ada batuk, pilek, mengi, nyeri dada, atau sesak saat aktivitas maupun saat istirahat?")
    if "Kidneys and Urinary System" in topic.groups:
        questions.append("Apakah ada nyeri saat BAK, anyang-anyangan, urine keruh atau berdarah, dan nyeri pinggang?")

    return unique_preserve_order(questions)[:5]


def build_display_title(original_title: str, translated_title: str) -> str:
    if translated_title and translated_title.lower() != original_title.lower():
        return f"{translated_title} / {original_title}"
    return original_title


def translate_label(text: str) -> str:
    if not text:
        return ""
    translated = f" {normalize_text(text).lower()} "
    for source, target in sorted(PHRASE_TRANSLATIONS.items(), key=lambda item: len(item[0]), reverse=True):
        translated = translated.replace(f" {source} ", f" {target} ")
    tokens = [TOKEN_TRANSLATIONS.get(token, token) for token in translated.split()]
    result = " ".join(tokens).strip()
    result = re.sub(r"\s+", " ", result)
    return result


def translate_sentence(text: str) -> str:
    translated = translate_label(text)
    if not translated:
        return ""
    translated = re.sub(r"\b(have|has|that|the|and|or|is|are|to|of|if|with|when|while|in|on|for|from)\b", " ", translated)
    translated = re.sub(r"\s+", " ", translated).strip(" ,.-")
    if not translated:
        return ""
    translated = translated[0].upper() + translated[1:]
    return translated


def as_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(value or "")).strip()


def limit_text(text: str, max_chars: int) -> str:
    compact = normalize_text(text)
    if len(compact) <= max_chars:
        return compact
    trimmed = compact[:max_chars].rsplit(" ", 1)[0]
    return f"{trimmed}..."


def unique_preserve_order(values) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = normalize_text(str(value))
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        result.append(normalized)
    return result


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


if __name__ == "__main__":
    raise SystemExit(main())
