from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from math import log, sqrt
from pathlib import Path
from typing import Any, Union

from app.config import CHROMA_DB_DIR
from app.services.anamnesis import AnamnesisResult
from app.services.knowledge_base import (
    CaseEntry,
    AnamnesisEntry,
    FormulaEntry,
    HerbEntry,
    KnowledgeBase,
    KnowledgeChunk,
    TrainingRecord,
)
from app.services.text import tokenize

try:  # ChromaDB is optional at import-time so local tests can still use the fallback.
    import chromadb
except Exception:  # pragma: no cover - exercised only when dependency is absent/broken.
    chromadb = None


Payload = Union[CaseEntry, FormulaEntry, HerbEntry, TrainingRecord, AnamnesisEntry, KnowledgeChunk]


@dataclass(frozen=True)
class RetrievedItem:
    id: str
    type: str
    title: str
    score: float
    source: str | None
    payload: Payload
    evidence_level: str | None = None
    matched_terms: list[str] = field(default_factory=list)


class InMemoryVectorIndex:
    """Small TF-IDF vector index used as a local vector database.

    This keeps the prototype runnable without downloading embedding models.
    The pipeline mirrors the thesis RAG flow: chunking, indexing, vector search,
    metadata-aware re-ranking, then grounded response generation.
    """

    def __init__(self, chunks: list[KnowledgeChunk]):
        self.chunks = chunks
        tokenized_docs = [tokenize(chunk.text) for chunk in chunks]
        doc_freq: Counter[str] = Counter()
        for tokens in tokenized_docs:
            doc_freq.update(set(tokens))

        total_docs = max(len(chunks), 1)
        self.idf = {token: log((1 + total_docs) / (1 + freq)) + 1 for token, freq in doc_freq.items()}
        self.vectors = [self._vectorize_tokens(tokens) for tokens in tokenized_docs]
        self.norms = [self._norm(vector) for vector in self.vectors]

    def search(self, query: str, limit: int = 10) -> list[RetrievedItem]:
        query_tokens = tokenize(query)
        query_vector = self._vectorize_tokens(query_tokens)
        query_norm = self._norm(query_vector)
        if not query_vector or query_norm == 0:
            return []

        results: list[RetrievedItem] = []
        for chunk, vector, norm in zip(self.chunks, self.vectors, self.norms):
            if norm == 0:
                continue
            score = self._cosine(query_vector, query_norm, vector, norm)
            if score <= 0:
                continue
            matched_terms = sorted(set(query_tokens) & set(vector))
            results.append(
                RetrievedItem(
                    id=chunk.id,
                    type=chunk.type,
                    title=chunk.title,
                    score=round(score, 4),
                    source=chunk.source,
                    payload=chunk,
                    evidence_level=chunk.evidence_level,
                    matched_terms=matched_terms,
                )
            )
        return sorted(results, key=lambda item: item.score, reverse=True)[:limit]

    def _vectorize_tokens(self, tokens: list[str]) -> dict[str, float]:
        counts = Counter(tokens)
        if not counts:
            return {}
        max_tf = max(counts.values())
        return {
            token: (0.5 + 0.5 * (count / max_tf)) * self.idf.get(token, 1.0)
            for token, count in counts.items()
        }

    @staticmethod
    def _norm(vector: dict[str, float]) -> float:
        return sqrt(sum(value * value for value in vector.values()))

    @staticmethod
    def _cosine(
        query_vector: dict[str, float],
        query_norm: float,
        doc_vector: dict[str, float],
        doc_norm: float,
    ) -> float:
        dot = sum(value * doc_vector.get(token, 0.0) for token, value in query_vector.items())
        return dot / (query_norm * doc_norm)


class LocalHashingEmbedding:
    """Deterministic lightweight embedding for ChromaDB without model downloads.

    This is intentionally small for the thesis prototype: it gives us a real
    vector-store flow with ChromaDB while avoiding external embedding services.
    QLoRA/production phases can replace this with a sentence embedding model.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

    def embed(self, text: str) -> list[float]:
        terms = self._terms(text)
        vector = [0.0] * self.dimension
        if not terms:
            return vector

        for term in terms:
            digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            weight = 1.35 if "__" in term else 1.0
            vector[bucket] += sign * weight

        norm = sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    @staticmethod
    def _terms(text: str) -> list[str]:
        tokens = tokenize(text)
        bigrams = [f"{first}__{second}" for first, second in zip(tokens, tokens[1:])]
        return tokens + bigrams


class ChromaVectorIndex:
    def __init__(
        self,
        chunks: list[KnowledgeChunk],
        persist_dir: Path,
        collection_name: str = "herbal_knowledge",
    ):
        if chromadb is None:
            raise RuntimeError("ChromaDB belum tersedia di environment backend.")

        self.chunks = chunks
        self.chunk_by_id = {chunk.id: chunk for chunk in chunks}
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding = LocalHashingEmbedding()
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self._rebuild_collection()

    def search(self, query: str, limit: int = 10) -> list[RetrievedItem]:
        query_embedding = self.embedding.embed(query)
        if not any(query_embedding) or not self.chunks:
            return []

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(limit, len(self.chunks)),
            include=["metadatas", "distances"],
        )
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]

        query_terms = set(tokenize(query))
        results: list[RetrievedItem] = []
        for item_id, distance, metadata in zip(ids, distances, metadatas):
            chunk = self.chunk_by_id.get(item_id)
            if not chunk:
                continue
            score = self._distance_to_score(float(distance))
            matched_terms = sorted(query_terms & set(tokenize(chunk.text)))
            results.append(
                RetrievedItem(
                    id=chunk.id,
                    type=str((metadata or {}).get("type") or chunk.type),
                    title=str((metadata or {}).get("title") or chunk.title),
                    score=round(score, 4),
                    source=chunk.source,
                    payload=chunk,
                    evidence_level=chunk.evidence_level,
                    matched_terms=matched_terms,
                )
            )
        return results

    def _rebuild_collection(self):
        collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        if not self.chunks:
            return collection

        batch_size = 128
        for start in range(0, len(self.chunks), batch_size):
            batch = self.chunks[start : start + batch_size]
            collection.upsert(
                ids=[chunk.id for chunk in batch],
                documents=[chunk.text for chunk in batch],
                embeddings=self.embedding.embed_many([chunk.text for chunk in batch]),
                metadatas=[self._metadata(chunk) for chunk in batch],
            )
        return collection

    @staticmethod
    def _metadata(chunk: KnowledgeChunk) -> dict[str, str | int | float | bool]:
        metadata: dict[str, str | int | float | bool] = {
            "type": chunk.type,
            "title": chunk.title,
        }
        if chunk.source:
            metadata["source"] = chunk.source
        if chunk.evidence_level:
            metadata["evidence_level"] = chunk.evidence_level
        if chunk.case_id:
            metadata["case_id"] = chunk.case_id
        if chunk.formula_id:
            metadata["formula_id"] = chunk.formula_id
        if chunk.herb_id:
            metadata["herb_id"] = chunk.herb_id

        for key, value in chunk.metadata.items():
            if value is None:
                continue
            if isinstance(value, list):
                metadata[key] = "; ".join(str(item) for item in value)
            elif isinstance(value, (str, int, float, bool)):
                metadata[key] = value
            else:
                metadata[key] = str(value)
        return metadata

    @staticmethod
    def _distance_to_score(distance: float) -> float:
        return max(0.0, min(1.0, 1.0 - distance))


class RAGRetriever:
    def __init__(self, kb: KnowledgeBase, chroma_db_dir: Path | None = None, use_chroma: bool = True):
        self.kb = kb
        self.chroma_db_dir = chroma_db_dir or CHROMA_DB_DIR
        self.backend = "in_memory_tfidf_vector_index"
        self.fallback_reason: str | None = None
        if use_chroma:
            try:
                self.index = ChromaVectorIndex(kb.chunks, self.chroma_db_dir)
                self.backend = "chromadb_persistent_vector_store"
            except Exception as error:  # pragma: no cover - defensive fallback for local machines.
                self.fallback_reason = str(error)
                self.index = InMemoryVectorIndex(kb.chunks)
        else:
            self.index = InMemoryVectorIndex(kb.chunks)

    def health(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "chunks_indexed": len(self.kb.chunks),
            "chroma_db_dir": str(self.chroma_db_dir),
            "fallback_reason": self.fallback_reason,
        }

    def retrieve_cases(self, query: str, anamnesis: AnamnesisResult, limit: int = 3) -> list[RetrievedItem]:
        vector_results = self.index.search(query, limit=20)
        case_results = [item for item in vector_results if item.type == "case"]

        enriched: list[RetrievedItem] = []
        for item in case_results:
            chunk = item.payload
            if not isinstance(chunk, KnowledgeChunk) or not chunk.case_id:
                continue
            case = self.kb.case_by_id(chunk.case_id)
            if not case:
                continue
            score = item.score
            if case.id in anamnesis.hinted_case_ids:
                score += 0.8
            if "ringan" in query.lower() and "ringan" in case.keluhan_ringan:
                score += 0.08
            enriched.append(
                RetrievedItem(
                    id=case.id,
                    type="case",
                    title=case.keluhan_ringan,
                    score=round(score, 4),
                    source=case.sumber_ringkas,
                    payload=case,
                    evidence_level="curated_case",
                    matched_terms=item.matched_terms,
                )
            )
        existing_ids = {item.id for item in enriched}
        for hinted_case_id in anamnesis.hinted_case_ids:
            if hinted_case_id in existing_ids:
                continue
            hinted_case = self.kb.case_by_id(hinted_case_id)
            if not hinted_case:
                continue
            enriched.append(
                RetrievedItem(
                    id=hinted_case.id,
                    type="case",
                    title=hinted_case.keluhan_ringan,
                    score=0.85,
                    source=hinted_case.sumber_ringkas,
                    payload=hinted_case,
                    evidence_level="curated_case_hint",
                    matched_terms=sorted(set(tokenize(query)) & set(tokenize(hinted_case.text))),
                )
            )
        return sorted(enriched, key=lambda item: item.score, reverse=True)[:limit]

    def retrieve_context_for_case(
        self,
        query: str,
        case: CaseEntry,
        anamnesis: AnamnesisResult,
        limit: int = 6,
    ) -> list[RetrievedItem]:
        retrieval_query = " ".join([query, case.keluhan_ringan, case.ramuan_rekomendasi, " ".join(case.bahan)])
        candidates = self.index.search(retrieval_query, limit=30)
        reranked: list[RetrievedItem] = []
        formula_hint = self._normalize_hint(case.ramuan_rekomendasi)
        ingredient_hints = {self._normalize_hint(ingredient) for ingredient in case.bahan}

        for item in candidates:
            if item.type == "case":
                continue
            score = item.score
            title_hint = self._normalize_hint(item.title)
            if item.type == "formula" and (title_hint in formula_hint or formula_hint in title_hint):
                score += 0.9
            if item.type in {"formula", "herb", "training", "anamnesis"}:
                if title_hint in ingredient_hints:
                    score += 0.5
                if any(term in title_hint for term in ingredient_hints):
                    score += 0.25
            if item.type == "training" and (title_hint in formula_hint or formula_hint in title_hint):
                score += 0.45
            if item.type == "anamnesis" and self._anamnesis_matches_case(item.payload, case.id):
                score += 0.55
            if item.evidence_level in {"high", "medium"}:
                score += 0.08
            if anamnesis.detected_symptoms and any(term in item.matched_terms for term in anamnesis.detected_symptoms):
                score += 0.08

            reranked.append(
                RetrievedItem(
                    id=item.id,
                    type=item.type,
                    title=item.title,
                    score=round(score, 4),
                    source=item.source,
                    payload=item.payload,
                    evidence_level=item.evidence_level,
                    matched_terms=item.matched_terms,
                )
            )

        deduped: dict[str, RetrievedItem] = {}
        for item in sorted(reranked, key=lambda result: result.score, reverse=True):
            deduped.setdefault(item.id, item)
        return list(deduped.values())[:limit]

    def retrieve_guidance(
        self,
        query: str,
        anamnesis: AnamnesisResult,
        limit: int = 5,
    ) -> list[RetrievedItem]:
        candidates = self.index.search(query, limit=30)
        reranked: list[RetrievedItem] = []
        normalized_query = self._normalize_hint(query)

        for item in candidates:
            chunk = item.payload if isinstance(item.payload, KnowledgeChunk) else None
            if item.type == "training":
                if not chunk or chunk.metadata.get("content_type") != "disease_guidance":
                    continue
            elif item.type == "anamnesis":
                if not chunk:
                    continue
            else:
                continue

            score = item.score
            if item.type == "training" and item.evidence_level in {"clinical_guideline_reference", "medical_review_reference"}:
                score += 0.15
            if item.type == "anamnesis" and chunk and chunk.metadata.get("condition_group") == "penyakit_tropis":
                score += 0.12
            if anamnesis.detected_symptoms and any(term in item.matched_terms for term in anamnesis.detected_symptoms):
                score += 0.12
            title_hint = self._normalize_hint(item.title)
            if title_hint and title_hint in normalized_query:
                score += 0.2

            reranked.append(
                RetrievedItem(
                    id=item.id,
                    type=item.type,
                    title=item.title,
                    score=round(score, 4),
                    source=item.source,
                    payload=item.payload,
                    evidence_level=item.evidence_level,
                    matched_terms=item.matched_terms,
                )
            )

        deduped: dict[str, RetrievedItem] = {}
        for item in sorted(reranked, key=lambda result: result.score, reverse=True):
            deduped.setdefault(item.id, item)
        return list(deduped.values())[:limit]

    @staticmethod
    def _normalize_hint(value: str) -> str:
        return " ".join(tokenize(value))

    @staticmethod
    def _anamnesis_matches_case(payload: Payload, case_id: str) -> bool:
        if isinstance(payload, KnowledgeChunk):
            case_ids = payload.metadata.get("applicable_case_ids", [])
            return isinstance(case_ids, list) and case_id in case_ids
        return False


# Backward-compatible alias for older tests/imports.
LightweightRetriever = RAGRetriever
