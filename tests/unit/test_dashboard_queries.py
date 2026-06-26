"""Unit tests for the Phase 1.7 dashboard query functions.

We seed a small synthetic DB and assert the reporting frames summarize it
correctly, and that every query is safe (non-raising) on an empty DB.
"""

from __future__ import annotations

from pathlib import Path

from kalshi_train.dashboard import queries as q
from kalshi_train.db.connection import connect
from kalshi_train.db.ingest import (
    EventRow,
    KalshiMarketRow,
    Observation,
    SeriesDefinition,
    TextDocument,
    bulk_insert_events,
    bulk_insert_kalshi_markets,
    upsert_document,
    upsert_observation,
    upsert_series_definition,
)


def _seed(db_path: Path) -> None:
    with connect(db_path) as conn:
        # Two series; one with obs, one empty.
        upsert_series_definition(
            conn, SeriesDefinition("CPIAUCSL", "FRED", "CPI", "monthly", revises=True)
        )
        upsert_series_definition(
            conn, SeriesDefinition("EMPTYSER", "FRED", "Empty", "monthly", revises=True)
        )
        upsert_observation(
            conn, Observation("CPIAUCSL", "2024-01-01", "2024-02-10", "2024-02-10", 300.0)
        )
        upsert_observation(
            conn, Observation("CPIAUCSL", "2024-01-01", "2024-03-10", "2024-03-10", 301.0)
        )
        # Text doc
        upsert_document(
            conn,
            TextDocument(
                source="fed",
                document_type="fomc_statement",
                title="t",
                published_date="2024-01-31",
                body="x" * 500,
                url="http://x/1",
            ),
        )
        # Kalshi market
        bulk_insert_kalshi_markets(
            conn,
            [
                KalshiMarketRow(
                    ticker="FED-1",
                    event_ticker="FED",
                    template_id="fed_decision",
                    result="yes",
                )
            ],
        )
        # Calendar event with consensus + surprise
        bulk_insert_events(
            conn,
            [
                EventRow(
                    event_id="GDPC1:2024-01-01",
                    event_name="Real GDP",
                    release_date="2024-04-25",
                    consensus_value=99.0,
                    actual_value=100.0,
                    surprise=1.0,
                )
            ],
        )
        conn.commit()


def test_queries_safe_on_empty_db(tmp_db: Path) -> None:
    # Every function must return a DataFrame (possibly empty) without raising.
    assert q.numeric_coverage(tmp_db).empty
    assert q.text_coverage(tmp_db).empty
    assert q.kalshi_coverage(tmp_db).empty
    assert q.calendar_coverage(tmp_db).empty
    assert q.empty_series(tmp_db).empty
    # table_counts lists schema tables; data tables are empty, but
    # question_templates is seeded at schema init.
    tc = q.table_counts(tmp_db).set_index("table")
    assert "text_documents" in tc.index
    assert tc.loc["text_documents", "rows"] == 0
    assert tc.loc["series_observations", "rows"] == 0
    assert tc.loc["question_templates", "rows"] == 7


def test_numeric_coverage_counts_periods_and_vintages(tmp_db: Path) -> None:
    _seed(tmp_db)
    cov = q.numeric_coverage(tmp_db).set_index("series_id")
    assert cov.loc["CPIAUCSL", "n_periods"] == 1
    assert cov.loc["CPIAUCSL", "n_rows"] == 2  # two vintages
    assert cov.loc["CPIAUCSL", "first_obs"] == "2024-01-01"
    assert cov.loc["EMPTYSER", "n_periods"] == 0


def test_empty_series_flags_unpopulated(tmp_db: Path) -> None:
    _seed(tmp_db)
    empties = set(q.empty_series(tmp_db)["series_id"])
    assert empties == {"EMPTYSER"}


def test_text_coverage_summarizes(tmp_db: Path) -> None:
    _seed(tmp_db)
    text = q.text_coverage(tmp_db).set_index("document_type")
    assert text.loc["fomc_statement", "n_docs"] == 1
    assert text.loc["fomc_statement", "avg_chars"] == 500


def test_kalshi_and_calendar_coverage(tmp_db: Path) -> None:
    _seed(tmp_db)
    k = q.kalshi_coverage(tmp_db).set_index("template_id")
    assert k.loc["fed_decision", "n_markets"] == 1
    assert k.loc["fed_decision", "n_resolved"] == 1

    cal = q.calendar_coverage(tmp_db).set_index("event_name")
    assert cal.loc["Real GDP", "n_events"] == 1
    assert cal.loc["Real GDP", "n_consensus"] == 1
    assert cal.loc["Real GDP", "n_surprise"] == 1


def test_ingest_log_returns_frame(tmp_db: Path) -> None:
    # Empty but well-formed.
    assert list(q.ingest_log(tmp_db).columns) == [
        "run_id",
        "source",
        "target",
        "status",
        "rows_added",
        "started_at",
        "finished_at",
    ]
