from __future__ import annotations

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
TRAINING_DIR = ROOT / "data" / "traning"
SOURCES_PATH = TRAINING_DIR / "herbal_preparation_sources.jsonl"
RECORDS_PATH = TRAINING_DIR / "herbal_preparation_training_records.jsonl"
SFT_PATH = TRAINING_DIR / "herbal_preparation_training_sft.jsonl"
MANIFEST_PATH = TRAINING_DIR / "herbal_preparation_manifest.json"
COMBINED_SFT_PATH = TRAINING_DIR / "combined_training_sft.jsonl"

SYSTEM_PROMPT = (
    "Anda adalah chatbot edukasi ramuan herbal berbahasa Indonesia. "
    "Jelaskan cara pengolahan ramuan secara praktis, higienis, aman, dan tetap memberi batasan bahwa ini bukan diagnosis medis final."
)


@dataclass(frozen=True)
class PreparationSource:
    id: str
    title: str
    url: str
    note: str


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
        elif len(text) > 24:
            self.text_parts.append(text)

    @property
    def title(self) -> str:
        return normalize_text(" ".join(self.title_parts))

    @property
    def text(self) -> str:
        return normalize_text(" ".join(self.text_parts))


SOURCES = [
    PreparationSource(
        id="ayosehat_wedang_sinden",
        title="Kencur untuk Atasi Dampak Polusi",
        url="https://ayosehat.kemkes.go.id/kencur-untuk-atasi-dampak-polusi",
        note="Sumber resmi Ayo Sehat Kemenkes dengan bahan dan langkah Wedang Sinden.",
    ),
    PreparationSource(
        id="keslan_7_jamu",
        title="7 Jamu Herbal yang Wajib Kamu Tahu",
        url="https://keslan.kemkes.go.id/view_artikel/2062/7-jamu-herbal-yang-wajib-kamu-tahu",
        note="Sumber Kemenkes untuk konteks kunyit asam, temulawak, beras kencur, dan jamu tradisional.",
    ),
    PreparationSource(
        id="keslan_batuk_tradisional",
        title="Batuk dengan Pengobatan Tradisional",
        url="https://keslan.kemkes.go.id/view_artikel/653/batuk-dengan-pengobatan-tradisonal",
        note="Sumber Kemenkes untuk ramuan batuk pilek ringan dan kewaspadaan.",
    ),
    PreparationSource(
        id="keslan_madu",
        title="Manfaat Madu bagi Kesehatan",
        url="https://keslan.kemkes.go.id/view_artikel/72/manfaat-madu-bagi-kesehatan",
        note="Sumber Kemenkes untuk konteks madu sebagai pendamping tenggorokan/batuk ringan.",
    ),
    PreparationSource(
        id="keslan_jahe",
        title="Takhlukan Mual Pasca Kemoterapi dengan Jahe",
        url="https://keslan.kemkes.go.id/view_artikel/978/takhlukan-mual-pasca-kemoterapi-dengan-jahe",
        note="Sumber Kemenkes untuk konteks jahe pada mual.",
    ),
    PreparationSource(
        id="jdih_formularium",
        title="Formularium Obat Herbal Asli Indonesia",
        url="https://jdih.kemkes.go.id/monografi/formularium-obat-herbal-asli-indonesia",
        note="Sumber Kemenkes/JDIH sebagai rujukan formularium obat herbal asli Indonesia.",
    ),
    PreparationSource(
        id="nccih_safety",
        title="Safe Use of Complementary Health Products and Practices",
        url="https://www.nccih.nih.gov/health/safety",
        note="Sumber NCCIH untuk prinsip keamanan, interaksi obat, dan kontaminasi produk herbal.",
    ),
]


PREPARATION_RECORDS = [
    {
        "id": "prep_wedang_sinden_detail",
        "source_id": "ayosehat_wedang_sinden",
        "topic": "batuk ringan suara serak dan paparan polusi",
        "formula_name": "Wedang Sinden",
        "ingredients": ["kencur 10 gram", "jahe 20 gram", "sereh 1 batang", "air 250 ml", "madu secukupnya", "air jeruk secukupnya"],
        "symptoms": ["batuk ringan", "suara serak", "tenggorokan tidak nyaman", "paparan polusi ringan"],
        "preparation": (
            "Cuci kencur, jahe, dan sereh sampai bersih lalu tiriskan. Rajang tipis kencur, jahe, dan sereh agar sari lebih mudah keluar. "
            "Masukkan bahan rajangan ke gelas atau teko bersih. Seduh dengan 250 ml air mendidih, tutup, lalu diamkan sekitar 10 menit. "
            "Setelah hangat, tambahkan madu dan air perasan jeruk secukupnya. Sajikan hangat dan jangan gunakan ulang ampas yang sudah lama."
        ),
        "dosage": "Sekitar 250 ml, 1 kali sehari untuk pendamping keluhan ringan dan penggunaan pendek.",
        "safety_notes": "Tidak untuk sesak, mengi berat, batuk darah, demam tinggi, atau keluhan memburuk. Madu tidak untuk bayi di bawah 12 bulan.",
        "evidence_level": "official_preparation_detail",
        "curation_method": "official_source_guided_manual_curation",
    },
    {
        "id": "prep_jahe_madu_detail",
        "source_id": "keslan_jahe",
        "topic": "mual ringan dan perut tidak nyaman",
        "formula_name": "Jahe / Jahe-Madu",
        "ingredients": ["jahe segar", "air 150-200 ml", "madu opsional"],
        "symptoms": ["mual ringan", "perut tidak nyaman", "tenggorokan tidak nyaman ringan"],
        "preparation": (
            "Cuci jahe segar dan kupas bagian yang kotor bila perlu. Iris tipis atau memarkan jahe agar ekstraksinya lebih baik. "
            "Rebus atau seduh jahe dengan 150-200 ml air panas selama 10-15 menit, lalu saring. Tunggu sampai hangat sebelum menambahkan madu. "
            "Aduk dan minum saat hangat; jangan menambahkan madu ke air yang masih mendidih."
        ),
        "dosage": "150-200 ml, 1-2 kali sehari untuk keluhan ringan. Madu 1-2 sendok teh bila digunakan.",
        "safety_notes": "Tidak untuk muntah terus, dehidrasi, nyeri perut hebat, demam tinggi, atau pengguna dengan risiko interaksi obat tanpa konsultasi.",
        "evidence_level": "official_source_guided",
        "curation_method": "official_source_guided_manual_curation",
    },
    {
        "id": "prep_jeruk_nipis_madu_detail",
        "source_id": "keslan_batuk_tradisional",
        "topic": "batuk pilek ringan",
        "formula_name": "Jeruk Nipis-Madu",
        "ingredients": ["jeruk nipis", "madu", "air hangat 150-200 ml"],
        "symptoms": ["batuk ringan", "pilek ringan", "tenggorokan tidak nyaman"],
        "preparation": (
            "Cuci jeruk nipis, belah, lalu peras secukupnya ke gelas bersih. Tambahkan air hangat 150-200 ml. "
            "Masukkan madu setelah air tidak terlalu panas, aduk sampai larut, lalu minum saat hangat. "
            "Gunakan bahan segar dan buat untuk sekali minum agar higienis."
        ),
        "dosage": "150-200 ml, 1-2 kali sehari untuk keluhan ringan dan penggunaan pendek.",
        "safety_notes": "Madu tidak untuk bayi di bawah 12 bulan. Hati-hati pada lambung sensitif, diabetes, sesak, demam tinggi, atau batuk darah.",
        "evidence_level": "official_source_guided",
        "curation_method": "official_source_guided_manual_curation",
    },
    {
        "id": "prep_kunyit_asam_detail",
        "source_id": "keslan_7_jamu",
        "topic": "rasa tidak nyaman tubuh ringan dan jamu tradisional",
        "formula_name": "Kunyit Asam",
        "ingredients": ["kunyit", "asam jawa", "air", "gula merah opsional", "sedikit garam opsional"],
        "symptoms": ["rasa tidak nyaman tubuh ringan", "nyeri tenggorokan ringan", "gangguan pencernaan ringan"],
        "preparation": (
            "Cuci kunyit, iris tipis atau parut. Rebus kunyit dengan air bersih selama 10-15 menit bersama asam jawa. "
            "Bila perlu, tambahkan sedikit gula merah sebagai pemanis; batasi gula pada pengguna diabetes. "
            "Saring rebusan, tunggu hangat, lalu sajikan. Buat dalam porsi kecil dan hindari penyimpanan lama."
        ),
        "dosage": "150-200 ml, 1 kali sehari atau maksimal 2 kali sehari untuk penggunaan pendek.",
        "safety_notes": "Hati-hati pada lambung sensitif, gangguan empedu, penggunaan obat pengencer darah, atau keluhan yang menetap/berat.",
        "evidence_level": "official_source_guided",
        "curation_method": "official_source_guided_manual_curation",
    },
    {
        "id": "prep_temulawak_kencur_asem_detail",
        "source_id": "keslan_7_jamu",
        "topic": "nafsu makan menurun dan pemulihan ringan",
        "formula_name": "Temulawak-Kencur-Asem",
        "ingredients": ["temulawak", "kencur", "asem kawak/asam jawa", "air", "gula aren opsional"],
        "symptoms": ["nafsu makan menurun", "mual ringan", "lemas ringan"],
        "preparation": (
            "Cuci temulawak dan kencur sampai bersih, lalu iris tipis atau geprek. Rebus rimpang dengan air bersih dan asam jawa selama 10-15 menit. "
            "Saring rebusan ke gelas bersih. Bila perlu, tambahkan sedikit gula aren saat hangat. "
            "Sajikan hangat dan gunakan sebagai pendamping pemulihan ringan, bukan pengganti evaluasi medis bila gejala menetap."
        ),
        "dosage": "150-200 ml, 1 kali sehari untuk penggunaan pendek.",
        "safety_notes": "Periksa ke tenaga kesehatan bila nafsu makan turun lama, berat badan turun, demam lama, muntah, atau anak sangat lemah.",
        "evidence_level": "official_source_guided",
        "curation_method": "official_source_guided_manual_curation",
    },
    {
        "id": "prep_beras_kencur_detail",
        "source_id": "keslan_7_jamu",
        "topic": "pegal ringan kurang fit dan nafsu makan menurun",
        "formula_name": "Beras Kencur",
        "ingredients": ["beras", "kencur", "jahe opsional", "kunyit opsional", "air matang", "gula secukupnya opsional"],
        "symptoms": ["pegal ringan", "kurang fit", "nafsu makan menurun"],
        "preparation": (
            "Rendam beras bersih sampai lebih lunak, lalu tiriskan. Cuci kencur dan rimpang tambahan, iris atau geprek. "
            "Haluskan beras dan kencur dengan sedikit air matang atau air rebusan hangat. Tambahkan air matang secukupnya, aduk, lalu saring. "
            "Bila menggunakan gula, tambahkan secukupnya saja. Sajikan segar dan jangan disimpan lama karena mudah berubah rasa."
        ),
        "dosage": "150-200 ml, 1 kali sehari untuk pendamping keluhan ringan.",
        "safety_notes": "Tidak untuk demam tinggi, nyeri hebat, kelemahan anggota tubuh, sesak, atau keluhan yang cepat memburuk.",
        "evidence_level": "official_source_guided",
        "curation_method": "official_source_guided_manual_curation",
    },
    {
        "id": "prep_serai_kayu_manis_madu_detail",
        "source_id": "keslan_madu",
        "topic": "tenggorokan tidak nyaman dan batuk ringan",
        "formula_name": "Serai-Kayu Manis-Madu",
        "ingredients": ["serai", "kayu manis", "madu", "air 150-200 ml"],
        "symptoms": ["tenggorokan tidak nyaman", "batuk ringan", "rasa gatal tenggorokan"],
        "preparation": (
            "Cuci serai, memarkan bagian batang putihnya, lalu rebus bersama kayu manis dengan 150-200 ml air selama 10-15 menit. "
            "Matikan api, saring, dan tunggu sampai hangat. Tambahkan madu setelah tidak mendidih, aduk, lalu minum hangat. "
            "Gunakan alat bersih dan buat porsi kecil untuk sekali minum."
        ),
        "dosage": "150-200 ml, 1-2 kali sehari untuk keluhan ringan.",
        "safety_notes": "Madu tidak untuk bayi di bawah 12 bulan. Tidak untuk sesak, sulit menelan berat, demam tinggi, alergi madu, atau keluhan memburuk.",
        "evidence_level": "curated_preparation_inference",
        "curation_method": "official_honey_source_plus_curated_formula_inference",
    },
    {
        "id": "prep_lidah_buaya_topikal_detail",
        "source_id": "nccih_safety",
        "topic": "gatal dan ruam merah ringan",
        "formula_name": "Gel Lidah Buaya Topikal",
        "ingredients": ["lidah buaya", "air bersih"],
        "symptoms": ["gatal ringan", "ruam merah ringan", "iritasi kulit ringan"],
        "preparation": (
            "Cuci tangan dan area kulit yang akan dioles. Gunakan gel lidah buaya yang bersih; bila mengambil dari daun segar, buang bagian kulit luar dan gunakan gel beningnya saja. "
            "Oleskan sangat tipis pada area kecil terlebih dahulu sebagai uji tempel. Tunggu dan perhatikan apakah ada perih, makin merah, gatal bertambah, atau bengkak. "
            "Jika tidak ada reaksi tidak nyaman, oles tipis pada area gatal/kemerahan. Hindari luka terbuka, area mata, dan area mukosa."
        ),
        "dosage": "Oles tipis 1-2 kali sehari maksimal 2-3 hari; hentikan bila iritasi atau gatal memburuk.",
        "safety_notes": "Tidak untuk ruam dengan demam tinggi, lepuh luas, ruam berdarah, sesak, bengkak wajah/bibir, atau ruam menyebar cepat.",
        "evidence_level": "safety_guided_curated",
        "curation_method": "safety_source_guided_manual_curation",
    },
    {
        "id": "prep_general_hygiene_safety",
        "source_id": "jdih_formularium",
        "topic": "prinsip umum pengolahan ramuan herbal rumahan",
        "formula_name": "Prinsip Higienis Pengolahan Herbal",
        "ingredients": ["bahan herbal bersih", "air bersih", "alat bersih"],
        "symptoms": ["edukasi keamanan", "pengolahan herbal"],
        "preparation": (
            "Pilih bahan yang masih baik, tidak berjamur, tidak busuk, dan tidak tercemar. Cuci bahan di air mengalir. "
            "Gunakan talenan, pisau, panci, gelas, dan saringan yang bersih. Untuk rebusan, gunakan air bersih dan sajikan setelah hangat. "
            "Untuk seduhan, tutup wadah beberapa menit agar sari bahan keluar. Jangan mencampur banyak bahan tanpa data keamanan dan jangan menyimpan ramuan terlalu lama."
        ),
        "dosage": "Ikuti dosis record formula/case terkurasi; jangan meningkatkan dosis karena merasa bahan alami pasti aman.",
        "safety_notes": "Pertimbangkan interaksi obat, alergi, kontaminasi produk, kehamilan, anak kecil, penyakit kronis, dan tanda bahaya.",
        "evidence_level": "safety_guideline",
        "curation_method": "formularium_and_safety_guideline_curation",
    },
]


def main() -> int:
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    source_rows = fetch_sources()
    records = build_records(source_rows)
    sft_rows = build_sft_examples(records)
    write_jsonl(SOURCES_PATH, source_rows)
    write_jsonl(RECORDS_PATH, records)
    write_jsonl(SFT_PATH, sft_rows)
    write_combined_sft()
    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "source_rows": len(source_rows),
                "training_record_rows": len(records),
                "training_sft_rows": len(sft_rows),
                "records_path": str(RECORDS_PATH.relative_to(ROOT)),
                "sft_path": str(SFT_PATH.relative_to(ROOT)),
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
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


def fetch_sources() -> list[dict[str, object]]:
    return [fetch_source(source) for source in SOURCES]


def fetch_source(source: PreparationSource) -> dict[str, object]:
    retrieved_at = datetime.now(timezone.utc).isoformat()
    try:
        raw_html = fetch_html(source.url)
        extractor = extract_html(raw_html)
        status = "fetched"
    except (ssl.SSLCertVerificationError, URLError) as error:
        if not is_ssl_verification_error(error):
            return failed_source(source, retrieved_at, error)
        try:
            raw_html = fetch_html(source.url, insecure_ssl=True)
            extractor = extract_html(raw_html)
            status = "fetched_insecure_ssl_fallback"
        except (HTTPError, URLError, TimeoutError, OSError, ssl.SSLError) as retry_error:
            return failed_source(source, retrieved_at, retry_error)
    except (HTTPError, TimeoutError, OSError) as error:
        return failed_source(source, retrieved_at, error)

    return {
        "id": source.id,
        "source_url": source.url,
        "title": extractor.title or source.title,
        "status": status,
        "note": source.note,
        "excerpt": make_excerpt(extractor.text),
        "retrieved_at": retrieved_at,
    }


def fetch_html(url: str, insecure_ssl: bool = False) -> str:
    request = Request(url, headers={"User-Agent": "ai-herbal-thesis-research/0.1"})
    context = ssl._create_unverified_context() if insecure_ssl else None
    with urlopen(request, timeout=20, context=context) as response:
        return response.read().decode(response.headers.get_content_charset() or "utf-8", errors="replace")


def extract_html(raw_html: str) -> BasicHTMLTextExtractor:
    extractor = BasicHTMLTextExtractor()
    extractor.feed(raw_html)
    return extractor


def failed_source(source: PreparationSource, retrieved_at: str, error: Exception) -> dict[str, object]:
    return {
        "id": source.id,
        "source_url": source.url,
        "title": source.title,
        "status": "fetch_failed",
        "note": source.note,
        "error": str(error),
        "excerpt": "",
        "retrieved_at": retrieved_at,
    }


def build_records(source_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    source_by_id = {source["id"]: source for source in source_rows}
    rows: list[dict[str, object]] = []
    for record in PREPARATION_RECORDS:
        source = source_by_id[str(record["source_id"])]
        rows.append(
            {
                **record,
                "source_title": source.get("title") or source.get("id") or "",
                "source_url": source.get("source_url") or "",
                "content_type": "herbal_preparation_guidance",
            }
        )
    return rows


def build_sft_examples(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        formula = str(record["formula_name"])
        rows.append(
            {
                "id": f"sft_{record['id']}",
                "source_record_id": record["id"],
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Bagaimana cara membuat {formula} yang lebih detail dan aman?"},
                    {"role": "assistant", "content": assistant_answer(record)},
                ],
            }
        )
    return rows


def assistant_answer(record: dict[str, object]) -> str:
    ingredients = ", ".join(as_list(record.get("ingredients"))) or "-"
    return (
        f"Untuk {record.get('formula_name')}, pastikan keluhan masih ringan dan tidak ada tanda bahaya.\n\n"
        f"Bahan: {ingredients}.\n"
        f"Cara pengolahan detail: {record.get('preparation')}.\n"
        f"Dosis/kisaran: {record.get('dosage')}.\n"
        f"Kewaspadaan: {record.get('safety_notes')}.\n"
        f"Sumber/kurasi: {record.get('source_title')}.\n\n"
        "Informasi ini bersifat edukasi, bukan diagnosis medis final atau pengganti konsultasi tenaga kesehatan."
    )


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


def is_ssl_verification_error(error: Exception) -> bool:
    if isinstance(error, ssl.SSLCertVerificationError):
        return True
    reason = getattr(error, "reason", None)
    return isinstance(reason, ssl.SSLCertVerificationError) or "CERTIFICATE_VERIFY_FAILED" in str(error)


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def make_excerpt(text: str, max_chars: int = 1200) -> str:
    normalized = normalize_text(text)
    if len(normalized) <= max_chars:
        return normalized
    cut = normalized[:max_chars].rsplit(" ", 1)[0]
    return f"{cut}..."


def as_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [item.strip() for item in value.split(";") if item.strip()]
    return []


if __name__ == "__main__":
    sys.exit(main())
