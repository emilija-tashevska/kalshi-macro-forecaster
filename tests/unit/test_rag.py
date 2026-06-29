"""Unit tests for the RAG layer (chunking, indexing, PIT retrieval) and
its integration into the SFT builder. A keyword-count fake embedder makes
similarity deterministic without an API key.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np

from kalshi_train.data.fomc_calendar import fomc_meeting_dates
from kalshi_train.db.connection import connect
from kalshi_train.db.ingest import (
    Observation,
    SeriesDefinition,
    TextDocument,
    upsert_document,
    upsert_observation,
    upsert_series_definition,
)
from kalshi_train.rag.chunk import chunk_text
from kalshi_train.rag.store import ChunkIndex, build_chunk_index, make_retriever
from kalshi_train.sft.dataset import build_sft_dataset

_VOCAB = ("inflation", "employment", "rate", "growth", "recession")


class _FakeEmbedder:
    """Deterministic keyword-count embeddings; cosine reflects word overlap."""

    model = "fake-embed"
    dim = len(_VOCAB)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(t.lower().count(w)) for w in _VOCAB] for t in texts]


# ── chunking ───────────────────────────────────────────────────────────


def test_chunk_text_splits_long_and_keeps_short() -> None:
    text = "\n\n".join(["Paragraph about inflation. " * 20 for _ in range(5)])
    chunks = chunk_text(text, chunk_chars=400, overlap_chars=50)
    assert len(chunks) > 1
    assert all(len(c) <= 400 + 50 + 10 for c in chunks)


def test_chunk_text_empty_returns_empty() -> None:
    assert chunk_text("") == []
    assert chunk_text("   ") == []


# ── index + retrieval ──────────────────────────────────────────────────


def _seed_docs(db_path: Path) -> None:
    docs = [
        ("2020-01-15", "inflation inflation inflation prices rising"),
        ("2021-06-15", "employment employment jobs labor market strong"),
        ("2022-09-15", "recession recession growth slowing sharply downturn"),
    ]
    with connect(db_path) as conn:
        for i, (pub, body) in enumerate(docs):
            upsert_document(
                conn,
                TextDocument(
                    source="fed",
                    document_type="fed_speech",
                    title=f"Speech {i}",
                    published_date=pub,
                    body=(body + " ") * 20,  # long enough to chunk
                    url=f"https://x/{i}.htm",
                ),
            )
        conn.commit()


def test_build_chunk_index_and_pit_retrieval(tmp_db: Path) -> None:
    _seed_docs(tmp_db)
    n = build_chunk_index(_FakeEmbedder(), db_path=tmp_db, chunk_chars=300)
    assert n > 0

    index = ChunkIndex(db_path=tmp_db)
    assert len(index) == n

    emb = _FakeEmbedder()
    q = emb.embed(["inflation outlook"])[0]
    qv = np.asarray(q, dtype=np.float32)

    # As of 2022 everything is visible; the inflation doc should rank first.
    hits_all = index.retrieve(qv, "2022-12-31", k=3)
    assert hits_all
    assert "inflation" in hits_all[0].text.lower()

    # PIT filter: as of 2020-06 only the Jan-2020 doc is knowable.
    hits_early = index.retrieve(qv, "2020-06-01", k=5)
    assert hits_early
    assert all(h.published_date <= "2020-06-01" for h in hits_early)


def test_build_chunk_index_skips_already_indexed(tmp_db: Path) -> None:
    _seed_docs(tmp_db)
    first = build_chunk_index(_FakeEmbedder(), db_path=tmp_db)
    second = build_chunk_index(_FakeEmbedder(), db_path=tmp_db)  # no new docs
    assert first > 0
    assert second == 0


def test_make_retriever_returns_pit_passages(tmp_db: Path) -> None:
    _seed_docs(tmp_db)
    build_chunk_index(_FakeEmbedder(), db_path=tmp_db)
    index = ChunkIndex(db_path=tmp_db)
    retrieve = make_retriever(_FakeEmbedder(), index, k=2)
    passages = retrieve(date(2021, 12, 31))
    assert passages
    # Tagged with type+date; none published after the as-of date.
    assert all(p.startswith("[fed_speech ") for p in passages)
    assert all(p.split()[1].rstrip("]") <= "2021-12-31" for p in passages)


# ── SFT integration ────────────────────────────────────────────────────


def test_sft_build_with_retriever_augments_prompts(
    tmp_db: Path, tmp_path: Path, monkeypatch
) -> None:
    # Minimal Fed-cut DB so snapshots can be built.
    dates = [
        "2020-03-15", "2020-04-28", "2020-06-10", "2020-07-29", "2020-09-16",
        "2020-11-05", "2020-12-16", "2021-01-27", "2021-03-17", "2021-04-28",
    ]
    cal = tmp_path / "fomc.txt"
    cal.write_text("\n".join(dates) + "\n")
    rates = dict.fromkeys(dates, 0.25)
    rates["2020-03-15"] = 1.75
    with connect(tmp_db) as conn:
        upsert_series_definition(
            conn, SeriesDefinition("DFEDTARU", "FRED", "Target upper", "daily", revises=False)
        )
        for d, r in rates.items():
            upsert_observation(conn, Observation("DFEDTARU", d, d, f"{d}T18:00:00+00:00", r))
        conn.commit()
    monkeypatch.setattr(
        "kalshi_train.targets.fed_cut.fomc_meeting_dates",
        lambda start, end, **kw: fomc_meeting_dates(
            start, end, calendar_path=cal, prefer_db=False, db_path=tmp_db
        ),
    )

    def fake_retriever(as_of: date) -> list[str]:
        return [f"[fed_speech 2019-01-01] inflation remains elevated as of {as_of}"]

    report = build_sft_dataset(
        start="2020-01-01",
        end="2021-12-31",
        horizons=(7, 1),
        db_path=tmp_db,
        output_dir=tmp_path / "sft",
        report_path=tmp_path / "p4.md",
        retriever=fake_retriever,
    )
    assert report.sanity_passed

    line = (tmp_path / "sft" / "train.jsonl").read_text().splitlines()[0]
    ex = json.loads(line)
    user = ex["messages"][1]["content"]
    assert "Relevant Fed communications" in user
    assert ex["metadata"]["n_passages"] == 1
