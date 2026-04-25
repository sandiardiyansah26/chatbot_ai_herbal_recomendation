from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class CaseEntry:
    id: str
    keluhan_ringan: str
    pertanyaan_anamnesis: str
    ramuan_rekomendasi: str
    bahan: list[str]
    cara_pengolahan: str
    dosis_penggunaan: str
    catatan_kewaspadaan: str
    sumber_ringkas: str

    @property
    def text(self) -> str:
        return " ".join(
            [
                self.keluhan_ringan,
                self.pertanyaan_anamnesis,
                self.ramuan_rekomendasi,
                " ".join(self.bahan),
                self.cara_pengolahan,
                self.catatan_kewaspadaan,
                self.sumber_ringkas,
            ]
        )


@dataclass(frozen=True)
class FormulaEntry:
    id: str
    nama_formula: str
    komposisi: list[str]
    gejala_target: str
    ringkasan_manfaat: str
    evidence_level: str
    catatan_keamanan: str
    sumber_utama: str

    @property
    def text(self) -> str:
        return " ".join(
            [
                self.nama_formula,
                " ".join(self.komposisi),
                self.gejala_target,
                self.ringkasan_manfaat,
                self.evidence_level,
                self.catatan_keamanan,
                self.sumber_utama,
            ]
        )


@dataclass(frozen=True)
class HerbEntry:
    id: str
    nama_lokal: str
    nama_latin: str
    gejala_target: str
    ringkasan_manfaat: str
    evidence_level: str
    catatan_keamanan: str
    sumber_utama: str

    @property
    def text(self) -> str:
        return " ".join(
            [
                self.nama_lokal,
                self.nama_latin,
                self.gejala_target,
                self.ringkasan_manfaat,
                self.evidence_level,
                self.catatan_keamanan,
                self.sumber_utama,
            ]
        )


@dataclass(frozen=True)
class TrainingRecord:
    id: str
    topic: str
    formula_name: str
    ingredients: list[str]
    symptoms: list[str]
    preparation: str
    dosage: str
    safety_notes: str
    evidence_level: str
    source_title: str
    source_url: str
    curation_method: str
    content_type: str = "herbal_guidance"
    overview: str = ""
    diagnosis_summary: str = ""
    prevention_steps: list[str] = field(default_factory=list)
    warning_signs: list[str] = field(default_factory=list)
    screening_questions: list[str] = field(default_factory=list)
    care_recommendation: str = ""

    @property
    def text(self) -> str:
        return " ".join(
            [
                self.content_type,
                self.topic,
                self.formula_name,
                " ".join(self.ingredients),
                " ".join(self.symptoms),
                self.overview,
                self.preparation,
                self.dosage,
                self.diagnosis_summary,
                " ".join(self.prevention_steps),
                " ".join(self.warning_signs),
                " ".join(self.screening_questions),
                self.care_recommendation,
                self.safety_notes,
                self.evidence_level,
                self.source_title,
                self.source_url,
                self.curation_method,
            ]
        )


@dataclass(frozen=True)
class AnamnesisEntry:
    id: str
    condition_group: str
    suspected_condition: str
    primary_symptoms: list[str]
    applicable_case_ids: list[str]
    required_questions: list[str]
    red_flag_questions: list[str]
    triage_action: str
    source_titles: list[str]
    source_urls: list[str]
    curation_method: str
    disclaimer: str

    @property
    def text(self) -> str:
        return " ".join(
            [
                self.condition_group,
                self.suspected_condition,
                " ".join(self.primary_symptoms),
                " ".join(self.applicable_case_ids),
                " ".join(self.required_questions),
                " ".join(self.red_flag_questions),
                self.triage_action,
                " ".join(self.source_titles),
                " ".join(self.source_urls),
                self.curation_method,
                self.disclaimer,
            ]
        )


@dataclass(frozen=True)
class KnowledgeChunk:
    id: str
    type: str
    title: str
    text: str
    source: str | None
    evidence_level: str | None
    case_id: str | None = None
    formula_id: str | None = None
    herb_id: str | None = None
    payload: CaseEntry | FormulaEntry | HerbEntry | TrainingRecord | AnamnesisEntry | None = None
    metadata: dict[str, str | list[str] | bool | None] = field(default_factory=dict)


class KnowledgeBase:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.cases = self._load_cases(data_dir / "focused_mild_ailment_herbal_dataset.csv")
        self.formulas = self._load_formulas(data_dir / "herbal_formulas.csv")
        self.herbs = self._load_herbs(data_dir / "herbal_references.csv")
        self.training_records = self._load_training_records(data_dir.parent / "traning")
        self.anamnesis_records = self._load_anamnesis_records(data_dir.parent / "anamnesis" / "anamnesis_questions.jsonl")
        self.chunks = self._build_chunks()

    def health(self) -> dict[str, int | str]:
        return {
            "data_dir": str(self.data_dir),
            "cases": len(self.cases),
            "formulas": len(self.formulas),
            "herbs": len(self.herbs),
            "training_records": len(self.training_records),
            "anamnesis_records": len(self.anamnesis_records),
            "chunks": len(self.chunks),
        }

    def case_by_id(self, case_id: str) -> CaseEntry | None:
        return next((case for case in self.cases if case.id == case_id), None)

    def formula_by_name_hint(self, formula_name: str) -> FormulaEntry | None:
        normalized = formula_name.lower().replace("/", " ")
        return next(
            (formula for formula in self.formulas if formula.nama_formula.lower() in normalized),
            None,
        )

    def _build_chunks(self) -> list[KnowledgeChunk]:
        chunks: list[KnowledgeChunk] = []
        for case in self.cases:
            chunks.append(
                KnowledgeChunk(
                    id=f"case:{case.id}",
                    type="case",
                    title=case.keluhan_ringan,
                    text=(
                        f"Keluhan ringan: {case.keluhan_ringan}. "
                        f"Pertanyaan anamnesis: {case.pertanyaan_anamnesis}. "
                        f"Ramuan rekomendasi: {case.ramuan_rekomendasi}. "
                        f"Bahan: {', '.join(case.bahan)}. "
                        f"Cara pengolahan: {case.cara_pengolahan}. "
                        f"Dosis: {case.dosis_penggunaan}. "
                        f"Kewaspadaan: {case.catatan_kewaspadaan}. "
                        f"Sumber: {case.sumber_ringkas}."
                    ),
                    source=case.sumber_ringkas,
                    evidence_level="curated_case",
                    case_id=case.id,
                    payload=case,
                    metadata={
                        "keluhan_ringan": case.keluhan_ringan,
                        "ramuan": case.ramuan_rekomendasi,
                        "bahan": case.bahan,
                    },
                )
            )
        for formula in self.formulas:
            chunks.append(
                KnowledgeChunk(
                    id=f"formula:{formula.id}",
                    type="formula",
                    title=formula.nama_formula,
                    text=(
                        f"Formula herbal: {formula.nama_formula}. "
                        f"Komposisi: {', '.join(formula.komposisi)}. "
                        f"Gejala target: {formula.gejala_target}. "
                        f"Manfaat ringkas: {formula.ringkasan_manfaat}. "
                        f"Level evidensi: {formula.evidence_level}. "
                        f"Kewaspadaan: {formula.catatan_keamanan}. "
                        f"Sumber: {formula.sumber_utama}."
                    ),
                    source=formula.sumber_utama,
                    evidence_level=formula.evidence_level,
                    formula_id=formula.id,
                    payload=formula,
                    metadata={
                        "gejala_target": formula.gejala_target,
                        "komposisi": formula.komposisi,
                    },
                )
            )
        for herb in self.herbs:
            chunks.append(
                KnowledgeChunk(
                    id=f"herb:{herb.id}",
                    type="herb",
                    title=herb.nama_lokal,
                    text=(
                        f"Tanaman herbal: {herb.nama_lokal}. "
                        f"Nama latin: {herb.nama_latin}. "
                        f"Gejala target: {herb.gejala_target}. "
                        f"Manfaat ringkas: {herb.ringkasan_manfaat}. "
                        f"Level evidensi: {herb.evidence_level}. "
                        f"Kewaspadaan: {herb.catatan_keamanan}. "
                        f"Sumber: {herb.sumber_utama}."
                    ),
                    source=herb.sumber_utama,
                    evidence_level=herb.evidence_level,
                    herb_id=herb.id,
                    payload=herb,
                    metadata={
                        "gejala_target": herb.gejala_target,
                        "nama_latin": herb.nama_latin,
                    },
                )
            )
        for record in self.training_records:
            if record.content_type == "disease_guidance":
                chunks.append(
                    KnowledgeChunk(
                        id=f"training:{record.id}",
                        type="training",
                        title=record.formula_name or record.topic,
                        text=(
                            f"Edukasi penyakit/triase: {record.formula_name or record.topic}. "
                            f"Topik: {record.topic}. "
                            f"Gejala kunci: {', '.join(record.symptoms)}. "
                            f"Ringkasan: {record.overview}. "
                            f"Pertanyaan skrining: {' | '.join(record.screening_questions)}. "
                            f"Pemeriksaan/diagnosis ringkas: {record.diagnosis_summary}. "
                            f"Pencegahan: {' | '.join(record.prevention_steps)}. "
                            f"Tanda bahaya: {' | '.join(record.warning_signs)}. "
                            f"Arahan: {record.care_recommendation or record.safety_notes}. "
                            f"Level evidensi: {record.evidence_level}. "
                            f"Sumber: {record.source_title} {record.source_url}. "
                            f"Metode kurasi: {record.curation_method}."
                        ),
                        source=record.source_url or record.source_title,
                        evidence_level=record.evidence_level,
                        payload=record,
                        metadata={
                            "content_type": record.content_type,
                            "topic": record.topic,
                            "formula_name": record.formula_name,
                            "symptoms": record.symptoms,
                            "prevention_steps": record.prevention_steps,
                            "warning_signs": record.warning_signs,
                            "screening_questions": record.screening_questions,
                            "curation_method": record.curation_method,
                        },
                    )
                )
                continue
            chunks.append(
                KnowledgeChunk(
                    id=f"training:{record.id}",
                    type="training",
                    title=record.formula_name or record.topic,
                    text=(
                        f"Data training herbal: {record.topic}. "
                        f"Formula/ramuan: {record.formula_name}. "
                        f"Bahan: {', '.join(record.ingredients)}. "
                        f"Gejala target: {', '.join(record.symptoms)}. "
                        f"Cara pengolahan: {record.preparation}. "
                        f"Dosis/kisaran penggunaan: {record.dosage}. "
                        f"Kewaspadaan: {record.safety_notes}. "
                        f"Level evidensi: {record.evidence_level}. "
                        f"Sumber: {record.source_title} {record.source_url}. "
                        f"Metode kurasi: {record.curation_method}."
                    ),
                    source=record.source_url or record.source_title,
                    evidence_level=record.evidence_level,
                    payload=record,
                    metadata={
                        "content_type": record.content_type,
                        "topic": record.topic,
                        "formula_name": record.formula_name,
                        "ingredients": record.ingredients,
                        "symptoms": record.symptoms,
                        "curation_method": record.curation_method,
                    },
                )
            )
        for record in self.anamnesis_records:
            chunks.append(
                KnowledgeChunk(
                    id=f"anamnesis:{record.id}",
                    type="anamnesis",
                    title=record.suspected_condition,
                    text=(
                        f"Dataset anamnesis: {record.condition_group}. "
                        f"Kondisi yang digali: {record.suspected_condition}. "
                        f"Gejala utama: {', '.join(record.primary_symptoms)}. "
                        f"Pertanyaan wajib: {' | '.join(record.required_questions)}. "
                        f"Pertanyaan red flag: {' | '.join(record.red_flag_questions)}. "
                        f"Tindakan triase: {record.triage_action}. "
                        f"Case terkait: {', '.join(record.applicable_case_ids)}. "
                        f"Sumber: {', '.join(record.source_titles)}."
                    ),
                    source="; ".join(record.source_urls),
                    evidence_level="anamnesis_guideline",
                    payload=record,
                    metadata={
                        "condition_group": record.condition_group,
                        "suspected_condition": record.suspected_condition,
                        "primary_symptoms": record.primary_symptoms,
                        "applicable_case_ids": record.applicable_case_ids,
                        "curation_method": record.curation_method,
                    },
                )
            )
        return chunks

    @staticmethod
    def _split_semicolon(value: str) -> list[str]:
        return [item.strip() for item in (value or "").split(";") if item.strip()]

    def _load_cases(self, path: Path) -> list[CaseEntry]:
        if not path.exists():
            raise FileNotFoundError(f"Dataset kasus keluhan ringan tidak ditemukan: {path}")
        with path.open(newline="", encoding="utf-8") as file:
            rows = csv.DictReader(file)
            return [
                CaseEntry(
                    id=row["id"],
                    keluhan_ringan=row["keluhan_ringan"],
                    pertanyaan_anamnesis=row["pertanyaan_anamnesis"],
                    ramuan_rekomendasi=row["ramuan_rekomendasi"],
                    bahan=self._split_semicolon(row["bahan"]),
                    cara_pengolahan=row["cara_pengolahan"],
                    dosis_penggunaan=row["dosis_penggunaan"],
                    catatan_kewaspadaan=row["catatan_kewaspadaan"],
                    sumber_ringkas=row["sumber_ringkas"],
                )
                for row in rows
            ]

    def _load_formulas(self, path: Path) -> list[FormulaEntry]:
        if not path.exists():
            return []
        with path.open(newline="", encoding="utf-8") as file:
            rows = csv.DictReader(file)
            return [
                FormulaEntry(
                    id=row["id"],
                    nama_formula=row["nama_formula"],
                    komposisi=self._split_semicolon(row["komposisi"]),
                    gejala_target=row["gejala_target"],
                    ringkasan_manfaat=row["ringkasan_manfaat"],
                    evidence_level=row["evidence_level"],
                    catatan_keamanan=row["catatan_keamanan"],
                    sumber_utama=row["sumber_utama"],
                )
                for row in rows
            ]

    def _load_herbs(self, path: Path) -> list[HerbEntry]:
        if not path.exists():
            return []
        with path.open(newline="", encoding="utf-8") as file:
            rows = csv.DictReader(file)
            return [
                HerbEntry(
                    id=row["id"],
                    nama_lokal=row["nama_lokal"],
                    nama_latin=row["nama_latin"],
                    gejala_target=row["gejala_target"],
                    ringkasan_manfaat=row["ringkasan_manfaat"],
                    evidence_level=row["evidence_level"],
                    catatan_keamanan=row["catatan_keamanan"],
                    sumber_utama=row["sumber_utama"],
                )
                for row in rows
            ]

    def _load_training_records(self, path: Path) -> list[TrainingRecord]:
        if not path.exists():
            return []

        if path.is_dir():
            record_paths = sorted(path.glob("*_training_records.jsonl"))
        else:
            record_paths = [path]

        records: list[TrainingRecord] = []
        for record_path in record_paths:
            with record_path.open(encoding="utf-8") as file:
                for line_number, line in enumerate(file, start=1):
                    if not line.strip():
                        continue
                    raw = json.loads(line)
                    default_id = f"{record_path.stem}_{line_number:04d}"
                    records.append(
                        TrainingRecord(
                            id=str(raw.get("id") or default_id),
                            topic=str(raw.get("topic") or "ramuan herbal"),
                            formula_name=str(raw.get("formula_name") or raw.get("title") or "Ramuan herbal"),
                            ingredients=self._ensure_list(raw.get("ingredients")),
                            symptoms=self._ensure_list(raw.get("symptoms")),
                            preparation=str(raw.get("preparation") or ""),
                            dosage=str(raw.get("dosage") or ""),
                            safety_notes=str(raw.get("safety_notes") or ""),
                            evidence_level=str(raw.get("evidence_level") or "curated_training"),
                            source_title=str(raw.get("source_title") or ""),
                            source_url=str(raw.get("source_url") or ""),
                            curation_method=str(raw.get("curation_method") or "scraped_and_curated"),
                            content_type=str(raw.get("content_type") or "herbal_guidance"),
                            overview=str(raw.get("overview") or ""),
                            diagnosis_summary=str(raw.get("diagnosis_summary") or ""),
                            prevention_steps=self._ensure_list(raw.get("prevention_steps")),
                            warning_signs=self._ensure_list(raw.get("warning_signs")),
                            screening_questions=self._ensure_list(raw.get("screening_questions")),
                            care_recommendation=str(raw.get("care_recommendation") or ""),
                        )
                    )
        return records

    def _load_anamnesis_records(self, path: Path) -> list[AnamnesisEntry]:
        if not path.exists():
            return []

        records: list[AnamnesisEntry] = []
        with path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                records.append(
                    AnamnesisEntry(
                        id=str(raw.get("id") or f"anamnesis_{line_number:04d}"),
                        condition_group=str(raw.get("condition_group") or ""),
                        suspected_condition=str(raw.get("suspected_condition") or ""),
                        primary_symptoms=self._ensure_list(raw.get("primary_symptoms")),
                        applicable_case_ids=self._ensure_list(raw.get("applicable_case_ids")),
                        required_questions=self._ensure_list(raw.get("required_questions")),
                        red_flag_questions=self._ensure_list(raw.get("red_flag_questions")),
                        triage_action=str(raw.get("triage_action") or ""),
                        source_titles=self._ensure_list(raw.get("source_titles")),
                        source_urls=self._ensure_list(raw.get("source_urls")),
                        curation_method=str(raw.get("curation_method") or "official_source_guided_manual_curation"),
                        disclaimer=str(raw.get("disclaimer") or ""),
                    )
                )
        return records

    @staticmethod
    def _ensure_list(value: object) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            return [item.strip() for item in value.replace("|", ";").split(";") if item.strip()]
        return []
