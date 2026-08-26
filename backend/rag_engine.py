#!/usr/bin/env python3
"""轻量 RAG 检索引擎。

不依赖外部模型和深度学习库：
- Markdown 文档按标题和段落切块；
- 使用中文字符、中文二元组和英文词构建 TF-IDF 向量；
- 同时计算 BM25，做向量相似度与关键词相关性混合检索。

这能让 Agent 在没有本地 embedding 模型时仍然具备可解释的检索能力。
"""

from __future__ import annotations

import math
import re
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


@dataclass
class Chunk:
    chunk_id: str
    source: str
    section: str
    text: str


def tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens: list[str] = []
    tokens.extend(re.findall(r"[a-z0-9_]+", text))
    for span in re.findall(r"[\u4e00-\u9fff]+", text):
        tokens.extend(span)
        if len(span) >= 2:
            tokens.extend(span[i : i + 2] for i in range(len(span) - 1))
    return tokens


def _merge_paragraphs(
    paragraphs: list[tuple[str, str]], max_chars: int
) -> list[tuple[str, str]]:
    chunks: list[tuple[str, str]] = []
    current_section = ""
    current_text = ""
    for section, paragraph in paragraphs:
        candidate = paragraph if not current_text else f"{current_text}\n\n{paragraph}"
        if len(candidate) <= max_chars:
            current_text = candidate
            current_section = current_section or section
            continue
        if current_text:
            chunks.append((current_section, current_text))
        current_text = paragraph
        current_section = section
    if current_text:
        chunks.append((current_section, current_text))
    return chunks


def markdown_chunks(
    text: str, source: str, max_chars: int = 760
) -> list[tuple[str, str]]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs: list[tuple[str, str]] = []
    section = source
    block = ""

    def flush() -> None:
        nonlocal block
        if block.strip():
            paragraphs.append((section, block.strip()))
        block = ""

    for line in text.splitlines():
        if line.startswith("#"):
            flush()
            section = line.lstrip("#").strip() or source
            continue
        if not line.strip():
            flush()
            continue
        block = line if not block else f"{block}\n{line}"
        if len(block) >= max_chars:
            flush()

    flush()
    return _merge_paragraphs(paragraphs, max_chars)


class RAGEngine:
    def __init__(self, root: Path, doc_files: list[str], top_k: int = 6) -> None:
        self.root = Path(root)
        self.doc_files = doc_files
        self.top_k = top_k
        self.chunks: list[Chunk] = []
        self.docs_tokens: list[list[str]] = []
        self.vocab: list[str] = []
        self.vocab_index: dict[str, int] = {}
        self.matrix: list[list[float]] = []
        self.doc_norms: list[float] = []
        self.external_embeddings: list[list[float]] = []
        self.fit()

    def fit(self) -> None:
        raw_chunks: list[tuple[str, str, str]] = []
        for rel in self.doc_files:
            path = self.root / rel
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8")
            for section, chunk_text in markdown_chunks(text, rel):
                raw_chunks.append((rel, section, chunk_text))

        self.chunks = [
            Chunk(
                chunk_id=f"{source}:{index}",
                source=source,
                section=section,
                text=text,
            )
            for index, (source, section, text) in enumerate(raw_chunks)
        ]

        self.docs_tokens = [tokenize(chunk.text) for chunk in self.chunks]
        df: dict[str, int] = {}
        for tokens in self.docs_tokens:
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1

        self.vocab = [term for term, count in df.items() if count >= 1]
        self.vocab_index = {term: index for index, term in enumerate(self.vocab)}
        self.matrix = self._build_tfidf_matrix(df)
        self.doc_norms = [math.sqrt(sum(value * value for value in row)) for row in self.matrix]
        self._maybe_load_external_embeddings()
        self.save_index()

    def _maybe_load_external_embeddings(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_EMBEDDING_MODEL")
        if not api_key or not model or not self.chunks:
            return
        try:
            self.external_embeddings = self._embed_openai(
                [chunk.text for chunk in self.chunks],
                api_key,
                model,
            )
        except Exception:
            self.external_embeddings = []

    def _embed_openai(
        self,
        texts: list[str],
        api_key: str,
        model: str,
    ) -> list[list[float]]:
        base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        payload = {"model": model, "input": texts}
        request = Request(
            f"{base}/embeddings",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
        rows = [item["embedding"] for item in data["data"]]
        return [list(map(float, row)) for row in rows]

    def index_path(self) -> Path:
        return self.root / "data" / "index" / "rag_index.json"

    def save_index(self) -> None:
        payload = {
            "schema_version": 1,
            "doc_files": self.doc_files,
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                    "section": chunk.section,
                    "text": chunk.text,
                }
                for chunk in self.chunks
            ],
            "vocab": self.vocab,
            "matrix": self.matrix,
            "doc_norms": self.doc_norms,
        }
        self.index_path().parent.mkdir(parents=True, exist_ok=True)
        self.index_path().write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )

    def load_index(self) -> bool:
        path = self.index_path()
        if not path.exists():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != 1:
                return False
            self.doc_files = list(payload.get("doc_files", self.doc_files))
            self.chunks = [
                Chunk(
                    chunk_id=item["chunk_id"],
                    source=item["source"],
                    section=item["section"],
                    text=item["text"],
                )
                for item in payload.get("chunks", [])
            ]
            self.vocab = list(payload.get("vocab", []))
            self.vocab_index = {term: index for index, term in enumerate(self.vocab)}
            self.matrix = [list(row) for row in payload.get("matrix", [])]
            self.doc_norms = [float(value) for value in payload.get("doc_norms", [])]
            return True
        except (KeyError, TypeError, json.JSONDecodeError):
            return False

    def _build_tfidf_matrix(self, df: dict[str, int]) -> list[list[float]]:
        if not self.chunks:
            return [[0.0 for _ in self.vocab] for _ in self.chunks]
        matrix = [[0.0 for _ in self.vocab] for _ in self.chunks]
        n_docs = len(self.chunks)
        for row, tokens in enumerate(self.docs_tokens):
            counts: dict[str, int] = {}
            for term in tokens:
                counts[term] = counts.get(term, 0) + 1
            length = max(len(tokens), 1)
            for term, count in counts.items():
                if term not in self.vocab_index:
                    continue
                idf = math.log((1 + n_docs) / (1 + df[term])) + 1.0
                matrix[row][self.vocab_index[term]] = (count / length) * idf
        return matrix

    def _query_vector(self, query: str) -> list[float]:
        if self.matrix is None or not self.vocab:
            return []
        vector = [0.0 for _ in self.vocab]
        tokens = tokenize(query)
        counts: dict[str, int] = {}
        for term in tokens:
            counts[term] = counts.get(term, 0) + 1
        length = max(len(tokens), 1)
        n_docs = len(self.chunks)
        for term, count in counts.items():
            if term not in self.vocab_index:
                continue
            doc_freq = sum(1 for doc in self.docs_tokens if term in set(doc))
            idf = math.log((1 + n_docs) / (1 + max(doc_freq, 1))) + 1.0
            vector[self.vocab_index[term]] = (count / length) * idf
        return vector

    def _bm25_scores(self, query: str) -> list[float]:
        if not self.docs_tokens:
            return []
        scores = [0.0 for _ in self.docs_tokens]
        query_tokens = tokenize(query)
        if not query_tokens:
            return scores
        avgdl = float(sum(len(doc) for doc in self.docs_tokens) / len(self.docs_tokens))
        k1 = 1.5
        b = 0.75
        n_docs = len(self.docs_tokens)
        for term in set(query_tokens):
            doc_freq = sum(1 for doc in self.docs_tokens if term in set(doc))
            idf = math.log(1 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))
            for row, doc in enumerate(self.docs_tokens):
                tf = doc.count(term)
                if tf == 0:
                    continue
                length = len(doc)
                denominator = tf + k1 * (1 - b + b * length / max(avgdl, 1))
                scores[row] += idf * ((k1 + 1) * tf / denominator)
        return scores

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        if not self.chunks:
            return []
        top_k = top_k or self.top_k
        if self.external_embeddings:
            api_key = os.getenv("OPENAI_API_KEY")
            model = os.getenv("OPENAI_EMBEDDING_MODEL")
            if api_key and model:
                query_embedding = self._embed_openai([query], api_key, model)[0]
                norms = [
                    math.sqrt(sum(value * value for value in row))
                    for row in self.external_embeddings
                ]
                query_norm = math.sqrt(sum(value * value for value in query_embedding))
                cosine = [
                    sum(a * b for a, b in zip(row, query_embedding))
                    / (norm or 1)
                    / (query_norm or 1)
                    for row, norm in zip(self.external_embeddings, norms)
                ]
                ranked = sorted(range(len(cosine)), key=lambda index: cosine[index], reverse=True)[:top_k]
                return [
                    {
                        "chunk_id": self.chunks[index].chunk_id,
                        "source": self.chunks[index].source,
                        "section": self.chunks[index].section,
                        "text": self.chunks[index].text,
                        "score": round(float(cosine[index]), 4),
                    }
                    for index in ranked
                ]
        query_vec = self._query_vector(query)
        query_norm = math.sqrt(sum(value * value for value in query_vec))

        if query_norm > 0 and self.doc_norms is not None:
            cosine = [
                sum(a * b for a, b in zip(row, query_vec)) / (norm or 1) / query_norm
                for row, norm in zip(self.matrix, self.doc_norms)
            ]
            cosine_max = max(cosine) if cosine else 0.0
            cosine_norm = [value / cosine_max for value in cosine] if cosine_max > 0 else [0.0 for _ in cosine]
        else:
            cosine_norm = [0.0 for _ in self.chunks]

        bm25 = self._bm25_scores(query)
        bm25_max = max(bm25) if bm25 else 0.0
        bm25_norm = [value / bm25_max for value in bm25] if bm25_max > 0 else [0.0 for _ in bm25]

        combined = [0.72 * a + 0.28 * b for a, b in zip(cosine_norm, bm25_norm)]
        ranked = sorted(range(len(combined)), key=lambda index: combined[index], reverse=True)[:top_k]
        results: list[dict[str, Any]] = []
        for index in ranked:
            chunk = self.chunks[int(index)]
            results.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                    "section": chunk.section,
                    "text": chunk.text,
                    "score": round(float(combined[int(index)]), 4),
                }
            )
        return results

    def build_context(self, query: str, max_chars: int = 6500) -> str:
        results = self.search(query)
        if not results:
            return "没有检索到足够相关的项目资料。"
        parts: list[str] = []
        total = 0
        for item in results:
            block = f"[{item['source']} · {item['section']}]\n{item['text']}"
            if total + len(block) > max_chars:
                remaining = max_chars - total
                if remaining > 120:
                    parts.append(block[:remaining])
                break
            parts.append(block)
            total += len(block)
        return "\n\n---\n\n".join(parts)

    def sources(self, query: str) -> list[str]:
        seen: list[str] = []
        for item in self.search(query):
            if item["source"] not in seen:
                seen.append(item["source"])
        return seen
