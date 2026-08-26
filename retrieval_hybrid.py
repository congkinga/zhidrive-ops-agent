"""Hybrid retrieval: vector, BM25/TF-IDF and reciprocal rank fusion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rag_engine import RAGEngine
from vector_rag import VectorRAG


ROOT = Path(__file__).resolve().parent
DOC_FILES = [
    "SOUL.md",
    "README.md",
    "CASE_STUDY.md",
    "product/README.md",
    "docs/zhijia-industry-observation.md",
    "docs/agent-architecture.md",
]


class HybridRetriever:
    def __init__(
        self,
        root: Path = ROOT,
        doc_files: list[str] | None = None,
        vector: VectorRAG | None = None,
        lexical: RAGEngine | None = None,
    ):
        self.root = root
        self.doc_files = doc_files or DOC_FILES
        self.lexical = lexical or RAGEngine(self.root, self.doc_files)
        self.vector = vector or VectorRAG(self.root, self.doc_files)

    def _rrf(
        self,
        vector_results: list[dict[str, Any]],
        lexical_results: list[dict[str, Any]],
        top_k: int,
        k: int = 60,
    ) -> list[dict[str, Any]]:
        scores: dict[str, float] = {}
        docs: dict[str, dict[str, Any]] = {}

        for rank, item in enumerate(vector_results, start=1):
            key = f"{item.get('source', '')}::{item.get('text', '')}"
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            docs[key] = {**item, "vector_rank": rank}

        for rank, item in enumerate(lexical_results, start=1):
            key = f"{item.get('source', '')}::{item.get('text', '')}"
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            merged = docs.get(key, item)
            merged["lexical_rank"] = rank
            docs[key] = merged

        ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        output: list[dict[str, Any]] = []
        for key, score in ranked[:top_k]:
            item = docs[key]
            item["rrf_score"] = round(score, 6)
            output.append(item)
        return output

    def search(self, query: str, top_k: int = 6) -> list[dict[str, Any]]:
        vector_results = self.vector.search(query, top_k=max(top_k, 8))
        lexical_results = self.lexical.search(query, top_k=max(top_k, 8))
        return self._rrf(vector_results, lexical_results, top_k=top_k)

    def sources(self, query: str) -> list[str]:
        seen: list[str] = []
        for item in self.search(query):
            source = item.get("source", "")
            if source and source not in seen:
                seen.append(source)
        return seen
