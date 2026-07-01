"""Persist chunk embeddings and do point-in-time retrieval.

``build_chunk_index`` chunks + embeds ``text_documents`` into
``text_chunks``. ``ChunkIndex`` loads all embedded chunks into memory once
and answers nearest-neighbour queries with a **PIT filter** (only chunks
published on/before the as-of date), so retrieval can never surface text
that wasn't public yet. ``make_retriever`` wires an embedder + index into
the simple ``Callable[[date], list[str]]`` the SFT builder consumes.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np

from kalshi_train.db.connection import connect
from kalshi_train.rag.chunk import chunk_text
from kalshi_train.rag.embedder import Embedder

logger = logging.getLogger(__name__)
_DOC_BATCH = 40  # docs per embed+commit cycle, so long runs are resumable

DateLike = date | datetime | str


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _to_blob(vec: Sequence[float]) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def _from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def _as_iso_date(value: DateLike) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)[:10]


def build_chunk_index(
    embedder: Embedder,
    *,
    db_path: Path | None = None,
    document_types: Sequence[str] | None = None,
    rebuild: bool = False,
    chunk_chars: int = 1500,
) -> int:
    """Chunk + embed documents into ``text_chunks``. Returns chunks written.

    Skips documents that already have chunks unless ``rebuild`` is set.
    """
    with connect(db_path, read_only=True) as conn:
        params: list[object] = []
        where = ""
        if document_types:
            placeholders = ",".join("?" for _ in document_types)
            where = f"WHERE document_type IN ({placeholders})"
            params.extend(document_types)
        docs = conn.execute(
            f"SELECT doc_id, source, document_type, published_date, body "
            f"FROM text_documents {where}",
            params,
        ).fetchall()
        existing = {
            r[0] for r in conn.execute("SELECT DISTINCT doc_id FROM text_chunks").fetchall()
        }

    if rebuild:
        with connect(db_path) as conn:
            conn.execute("DELETE FROM text_chunks")
            conn.commit()
        existing = set()

    todo = [d for d in docs if d["doc_id"] not in existing]
    total_written = 0

    # Process in doc-batches with a commit after each, so a long/rate-limited
    # run is resumable: already-embedded docs are skipped on re-run.
    for start in range(0, len(todo), _DOC_BATCH):
        batch_docs = todo[start : start + _DOC_BATCH]
        meta: list[dict[str, object]] = []
        texts: list[str] = []
        for d in batch_docs:
            for idx, passage in enumerate(chunk_text(d["body"], chunk_chars=chunk_chars)):
                meta.append(
                    {
                        "chunk_id": f"{d['doc_id']}:{idx}",
                        "doc_id": d["doc_id"],
                        "source": d["source"],
                        "document_type": d["document_type"],
                        "published_date": _as_iso_date(d["published_date"]),
                        "chunk_index": idx,
                        "text": passage,
                    }
                )
                texts.append(passage)
        if not texts:
            continue

        vectors = embedder.embed(texts)
        now = _now_iso()
        rows = [
            (
                m["chunk_id"], m["doc_id"], m["source"], m["document_type"],
                m["published_date"], m["chunk_index"], m["text"],
                _to_blob(vec), embedder.model, embedder.dim, now,
            )
            for m, vec in zip(meta, vectors, strict=True)
        ]
        with connect(db_path) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO text_chunks (chunk_id, doc_id, source, "
                "document_type, published_date, chunk_index, text, embedding, "
                "embed_model, dim, ingested_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
            conn.commit()
        total_written += len(rows)
        logger.info(
            "embedded %d/%d docs (%d chunks so far)",
            min(start + _DOC_BATCH, len(todo)), len(todo), total_written,
        )

    return total_written


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    text: str
    doc_id: str
    document_type: str
    published_date: str
    score: float


class ChunkIndex:
    """In-memory view of embedded chunks for fast PIT nearest-neighbour search."""

    def __init__(
        self, *, db_path: Path | None = None, document_types: Sequence[str] | None = None
    ) -> None:
        where = "WHERE embedding IS NOT NULL"
        params: list[object] = []
        if document_types:
            placeholders = ",".join("?" for _ in document_types)
            where += f" AND document_type IN ({placeholders})"
            params.extend(document_types)
        with connect(db_path, read_only=True) as conn:
            rows = conn.execute(
                f"SELECT text, doc_id, document_type, published_date, embedding "
                f"FROM text_chunks {where}",
                params,
            ).fetchall()

        self.texts = [r["text"] for r in rows]
        self.doc_ids = [r["doc_id"] for r in rows]
        self.doc_types = [r["document_type"] for r in rows]
        self.pub_dates = np.array([str(r["published_date"])[:10] for r in rows])
        if rows:
            mat = np.vstack([_from_blob(r["embedding"]) for r in rows]).astype(np.float32)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            self.matrix = mat / np.clip(norms, 1e-8, None)
        else:
            self.matrix = np.zeros((0, 0), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.texts)

    def retrieve(
        self, query_vec: np.ndarray, as_of: DateLike, k: int = 4
    ) -> list[RetrievedChunk]:
        """Top-``k`` chunks published on/before ``as_of`` by cosine similarity."""
        if len(self) == 0:
            return []
        as_of_iso = _as_iso_date(as_of)
        mask = self.pub_dates <= as_of_iso
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        q = q / max(float(np.linalg.norm(q)), 1e-8)
        sims = self.matrix[idx] @ q
        order = np.argsort(-sims)[:k]
        return [
            RetrievedChunk(
                text=self.texts[idx[i]],
                doc_id=self.doc_ids[idx[i]],
                document_type=self.doc_types[idx[i]],
                published_date=str(self.pub_dates[idx[i]]),
                score=float(sims[i]),
            )
            for i in order
        ]


_RETRIEVAL_QUERY = (
    "U.S. monetary policy outlook and the case for changing the federal "
    "funds rate, given recent inflation, employment, and growth."
)


def make_retriever(
    embedder: Embedder, index: ChunkIndex, *, k: int = 4
) -> Callable[[DateLike], list[str]]:
    """Return a ``as_of -> [passage, ...]`` retriever for the SFT builder.

    The query is constant, so we embed it once; each call only re-applies
    the point-in-time date filter. Passages are prefixed with their source +
    date so the model knows what it's reading (and that it predates the
    decision).
    """
    qvec = np.asarray(embedder.embed([_RETRIEVAL_QUERY])[0], dtype=np.float32)

    def _retrieve(as_of: DateLike) -> list[str]:
        hits = index.retrieve(qvec, as_of, k=k)
        return [f"[{h.document_type} {h.published_date}] {h.text}".strip() for h in hits]

    return _retrieve


__all__ = [
    "ChunkIndex",
    "RetrievedChunk",
    "build_chunk_index",
    "make_retriever",
]
