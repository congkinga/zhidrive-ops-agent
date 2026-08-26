#!/usr/bin/env python3
"""ChromaDB + sentence-transformers 向量 RAG 后端。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")

import chromadb
from sentence_transformers import SentenceTransformer

from .rag_engine import markdown_chunks


MODEL_NAME = os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
COLLECTION_NAME = "zhidrive_ops"


class VectorRAG:
    def __init__(
        self,
        root: Path,
        doc_files: list[str],
        model_name: str = MODEL_NAME,
    ) -> None:
        self.root = Path(root)
        self.doc_files = doc_files
        self.model_name = model_name
        self.available = False
        self.error = ""
        self.model = None
        self.collection = None
        self.client = None
        self._initialize()

    def _initialize(self) -> None:
        try:
            self.client = chromadb.PersistentClient(
                path=str(self.root / "data" / "chroma")
            )
            self.collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            self.model = SentenceTransformer(self.model_name)
            self._rebuild_if_empty()
            self.available = True
        except Exception as exc:
            self.error = str(exc)
            self.available = False

    def _rebuild_if_empty(self) -> None:
        if not self.collection:
            return
        count = self.collection.count()
        if count > 0:
            return
        documents: list[str] = []
        metadatas: list[dict[str, str]] = []
        ids: list[str] = []
        index = 0
        for rel in self.doc_files:
            path = self.root / rel
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8")
            for section, chunk in markdown_chunks(text, rel):
                chunk_id = f"{rel}:{index}"
                ids.append(chunk_id)
                documents.append(chunk)
                metadatas.append({"source": rel, "section": section})
                index += 1
        if not documents:
            return
        embeddings = self.model.encode(
            documents,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=[list(map(float, vector)) for vector in embeddings],
        )

    def search(self, query: str, top_k: int = 6) -> list[dict[str, Any]]:
        if not self.available or self.model is None or self.collection is None:
            return []
        embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        result = self.collection.query(
            query_embeddings=[list(map(float, embedding))],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        output: list[dict[str, Any]] = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            output.append(
                {
                    "source": metadata.get("source", ""),
                    "section": metadata.get("section", ""),
                    "text": document,
                    "score": round(max(0.0, 1.0 - float(distance)), 4),
                }
            )
        return output

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
