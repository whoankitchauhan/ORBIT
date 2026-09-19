"""Semantic memory: embeddings plus similarity search.

ChromaDB is used when installed. When it is not, the same interface is served
by an embedded index that persists to JSON and scores with cosine similarity
over hashed-bag-of-words vectors combined with an inverse-document-frequency
weighting.

The fallback is not a keyword `LIKE` query in disguise. It projects tokens into
a fixed-dimension space with sub-word shingles, so "refund policy" still
retrieves a document about "refunds and returns" — near-miss matching is the
whole point of semantic memory, and a demo that only does exact matching would
misrepresent what the component does.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import settings

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Words, 4-character shingles of longer words, and adjacent-word bigrams.

    The shingles give the fallback its fuzziness: "replacement" and "replace"
    share several shingles even though they are different tokens.

    The bigrams do the opposite job. An identifier like "CUST-002" splits into
    the very common token "cust" and the short token "002", so on unigrams
    alone a record about CUST-004 scores almost as well as the one actually
    asked for. The bigram "cust_002" is rare and pins the match to the right
    record.
    """
    words = TOKEN_RE.findall(text.lower())
    tokens: list[str] = []
    for word in words:
        tokens.append(word)
        if len(word) > 5:
            tokens.extend(word[i : i + 4] for i in range(len(word) - 3))
    tokens.extend(f"{a}_{b}" for a, b in zip(words, words[1:]))
    return tokens


def embed(text: str, dim: int | None = None) -> list[float]:
    """Hash tokens into a fixed-width vector and L2-normalise it.

    Deterministic, dependency-free, and good enough for retrieval over a small
    corpus. Swap this for a sentence-transformer by changing this one function.
    """
    dim = dim or settings.embedding_dim
    vector = [0.0] * dim
    counts = Counter(tokenize(text))
    for token, count in counts.items():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        # Sub-linear term weighting stops one repeated word dominating.
        vector[index] += sign * (1.0 + math.log(count))
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [v / norm for v in vector]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


@dataclass
class MemoryRecord:
    id: str
    text: str
    source: str
    kind: str = "document"
    metadata: dict[str, Any] | None = None
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "kind": self.kind,
            "metadata": self.metadata or {},
            "score": round(self.score, 4),
        }


class EmbeddedVectorIndex:
    """JSON-backed vector index used when ChromaDB is absent."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._records: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self._records = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._records = {}

    def _persist(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._records), encoding="utf-8")
        tmp.replace(self.path)

    def add(self, record_id: str, text: str, metadata: dict[str, Any]) -> None:
        with self._lock:
            self._records[record_id] = {
                "text": text,
                "metadata": metadata,
                "vector": embed(text),
                "added_at": time.time(),
            }
            self._persist()

    def _idf(self) -> dict[str, float]:
        """Document frequency weighting, recomputed per query.

        The corpus here is small (tens to low hundreds of documents), so this
        stays cheap and avoids keeping a stale index on disk.
        """
        total = len(self._records) or 1
        frequency: Counter[str] = Counter()
        for record in self._records.values():
            frequency.update(set(TOKEN_RE.findall(record["text"].lower())))
        return {
            token: math.log(1 + total / (1 + count))
            for token, count in frequency.items()
        }

    def search(self, query: str, k: int, where: dict[str, Any] | None = None) -> list[MemoryRecord]:
        with self._lock:
            records = list(self._records.items())
        if not records:
            return []

        query_vector = embed(query)
        idf = self._idf()
        query_tokens = set(TOKEN_RE.findall(query.lower()))

        results: list[MemoryRecord] = []
        for record_id, record in records:
            metadata = record.get("metadata", {})
            if where and any(metadata.get(key) != value for key, value in where.items()):
                continue

            dense = cosine(query_vector, record["vector"])
            record_tokens = set(TOKEN_RE.findall(record["text"].lower()))
            overlap = query_tokens & record_tokens
            lexical = sum(idf.get(t, 0.0) for t in overlap) / (1 + math.log(1 + len(query_tokens)))
            # Blend dense similarity with idf-weighted overlap: the dense part
            # catches paraphrase, the lexical part anchors rare exact terms
            # like an order id that hashing would otherwise dilute.
            score = 0.65 * dense + 0.35 * min(1.0, lexical / 3.0)
            # Filter out completely unrelated documents with zero word overlap and low dense similarity
            if not overlap and dense < 0.38:
                continue
            if score < 0.18:
                continue
            results.append(
                MemoryRecord(
                    id=record_id,
                    text=record["text"],
                    source=metadata.get("source", "memory"),
                    kind=metadata.get("kind", "document"),
                    metadata=metadata,
                    score=score,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]

    def count(self) -> int:
        return len(self._records)

    def delete(self, ids: list[str]) -> None:
        """Remove records by their IDs."""
        with self._lock:
            for record_id in ids:
                self._records.pop(record_id, None)
            self._persist()

    def delete_by_kind(self, kind: str) -> int:
        """Delete all records of a given kind. Returns the number removed."""
        with self._lock:
            to_remove = [
                rid for rid, rec in self._records.items()
                if rec.get("metadata", {}).get("kind") == kind
            ]
            for rid in to_remove:
                del self._records[rid]
            if to_remove:
                self._persist()
        return len(to_remove)

    def clear(self) -> None:
        with self._lock:
            self._records = {}
            self._persist()


class VectorMemory:
    """Public semantic-memory interface used by agents and tools."""

    def __init__(self) -> None:
        self.backend = "chromadb" if settings.has_chroma else "embedded"
        self._collection: Any = None
        self._index: EmbeddedVectorIndex | None = None

        if self.backend == "chromadb":
            try:
                import chromadb  # lazy

                client = chromadb.PersistentClient(path=str(settings.chroma_path))
                self._collection = client.get_or_create_collection(
                    name=settings.chroma_collection,
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception:
                # A broken Chroma install must not take the platform down.
                self.backend = "embedded"

        if self.backend == "embedded":
            self._index = EmbeddedVectorIndex(settings.chroma_path / "embedded_index.json")

    # ---------------------------------------------------------------- write

    def add(
        self,
        text: str,
        source: str = "manual",
        kind: str = "document",
        metadata: dict[str, Any] | None = None,
        record_id: str | None = None,
    ) -> str:
        text = (text or "").strip()
        if not text:
            raise ValueError("cannot store empty text in semantic memory")

        record_id = record_id or hashlib.blake2b(
            f"{source}:{text}".encode("utf-8"), digest_size=8
        ).hexdigest()
        meta = {"source": source, "kind": kind, "added_at": time.time(), **(metadata or {})}

        if self.backend == "chromadb":
            self._collection.upsert(ids=[record_id], documents=[text], metadatas=[meta])
        else:
            assert self._index is not None
            self._index.add(record_id, text, meta)
        return record_id

    def add_many(self, items: list[dict[str, Any]]) -> int:
        added = 0
        for item in items:
            try:
                self.add(
                    text=item["text"],
                    source=item.get("source", "bulk"),
                    kind=item.get("kind", "document"),
                    metadata=item.get("metadata"),
                )
                added += 1
            except ValueError:
                continue
        return added

    def chunk_and_add(
        self, text: str, source: str, kind: str = "document", chunk_chars: int = 700
    ) -> int:
        """Split a long document on paragraph boundaries, then store the chunks.

        Retrieval quality depends far more on chunk boundaries than on the
        embedding function, so chunks are cut at paragraphs and only split
        mid-paragraph when one paragraph exceeds the budget on its own.
        """
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        chunks: list[str] = []
        buffer = ""
        for paragraph in paragraphs:
            if len(paragraph) > chunk_chars:
                if buffer:
                    chunks.append(buffer)
                    buffer = ""
                for i in range(0, len(paragraph), chunk_chars):
                    chunks.append(paragraph[i : i + chunk_chars])
            elif len(buffer) + len(paragraph) + 2 <= chunk_chars:
                buffer = f"{buffer}\n\n{paragraph}" if buffer else paragraph
            else:
                chunks.append(buffer)
                buffer = paragraph
        if buffer:
            chunks.append(buffer)

        for i, chunk in enumerate(chunks):
            self.add(chunk, source=source, kind=kind, metadata={"chunk": i})
        return len(chunks)

    # ----------------------------------------------------------------- read

    def search(
        self, query: str, k: int | None = None, where: dict[str, Any] | None = None
    ) -> list[MemoryRecord]:
        k = k or settings.retrieval_top_k
        if not (query or "").strip():
            return []

        if self.backend == "chromadb":
            try:
                response = self._collection.query(
                    query_texts=[query], n_results=k, where=where or None
                )
            except Exception:
                return []
            records: list[MemoryRecord] = []
            ids = (response.get("ids") or [[]])[0]
            documents = (response.get("documents") or [[]])[0]
            metadatas = (response.get("metadatas") or [[]])[0]
            distances = (response.get("distances") or [[]])[0]
            for i, record_id in enumerate(ids):
                meta = metadatas[i] or {}
                # Chroma reports cosine *distance*; convert so higher is better.
                score = 1.0 - float(distances[i]) if i < len(distances) else 0.0
                records.append(
                    MemoryRecord(
                        id=record_id,
                        text=documents[i],
                        source=meta.get("source", "memory"),
                        kind=meta.get("kind", "document"),
                        metadata=meta,
                        score=score,
                    )
                )
            return records

        assert self._index is not None
        return self._index.search(query, k, where)

    def count(self) -> int:
        if self.backend == "chromadb":
            try:
                return int(self._collection.count())
            except Exception:
                return 0
        assert self._index is not None
        return self._index.count()

    def delete_episodes(self) -> int:
        """Remove all episodic memory records. Returns count removed.

        Episodes are full task-outcome summaries that were written during runs.
        Clearing them is safe: the knowledge base (policies, documents) is
        not affected.
        """
        if self.backend == "chromadb":
            try:
                result = self._collection.get(where={"kind": "episode"})
                ids = result.get("ids", [])
                if ids:
                    self._collection.delete(ids=ids)
                return len(ids)
            except Exception:
                return 0
        assert self._index is not None
        return self._index.delete_by_kind("episode")

    def clear(self) -> None:
        if self.backend == "chromadb":
            ids = self._collection.get().get("ids", [])
            if ids:
                self._collection.delete(ids=ids)
        elif self._index is not None:
            self._index.clear()


_memory: VectorMemory | None = None


def get_memory() -> VectorMemory:
    global _memory
    if _memory is None:
        _memory = VectorMemory()
    return _memory
