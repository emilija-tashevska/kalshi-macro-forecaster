"""Unit tests for the Phase 1.6 event-calendar ingestion.

We seed a tiny synthetic ``series_observations`` table and verify:

  - release events use the *first* vintage (the print), not revisions
  - surprise = actual - consensus only when a consensus series exists
  - FOMC events classify cut / hold / hike against the prior meeting
  - the whole ingest is idempotent (re-running keeps counts stable)
  - once populated, fomc_calendar prefers the DB over the static file
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from kalshi_train.data.calendar_registry import ReleaseEvent
from kalshi_train.data.fomc_calendar import fomc_meeting_dates
from kalshi_train.data.ingest_calendar import (
    build_fomc_events,
    build_release_events,
    run_calendar_ingest,
)
from kalshi_train.db.connection import connect
from kalshi_train.db.ingest import (
    Observation,
    SeriesDefinition,
    upsert_observation,
    upsert_series_definition,
)


def _seed_series(db_path: Path, defn: SeriesDefinition, obs: list[Observation]) -> None:
    with connect(db_path) as conn:
        upsert_series_definition(conn, defn)
        for o in obs:
            upsert_observation(conn, o)
        conn.commit()


def _seed_gdp_and_consensus(db_path: Path) -> None:
    # GDPC1: advance print 100.0 (Apr), revised up to 102.0 (Jun).
    _seed_series(
        db_path,
        SeriesDefinition("GDPC1", "FRED", "Real GDP", "quarterly", revises=True),
        [
            Observation("GDPC1", "2024-01-01", "2024-04-25", "2024-04-25", 100.0),
            Observation("GDPC1", "2024-01-01", "2024-06-27", "2024-06-27", 102.0),
        ],
    )
    # SPF same-quarter nowcast: 99.0.
    _seed_series(
        db_path,
        SeriesDefinition(
            "SPF_RGDP_MEDIAN_NOWCAST", "SPF", "SPF RGDP nowcast", "quarterly"
        ),
        [Observation("SPF_RGDP_MEDIAN_NOWCAST", "2024-01-01", "2024-02-15", "2024-02-15", 99.0)],
    )


def _seed_cpi(db_path: Path) -> None:
    _seed_series(
        db_path,
        SeriesDefinition("CPIAUCSL", "FRED", "CPI", "monthly", revises=True),
        [Observation("CPIAUCSL", "2024-08-01", "2024-09-11", "2024-09-11", 314.0)],
    )


def _seed_fed_rates(db_path: Path, rates: dict[str, float]) -> None:
    obs = [
        Observation("DFEDTARU", d, d, d, v)  # daily, non-revising
        for d, v in rates.items()
    ]
    _seed_series(
        db_path,
        SeriesDefinition("DFEDTARU", "FRED", "Fed Funds Target Upper", "daily", revises=False),
        obs,
    )


def _write_calendar(tmp_path: Path, dates: list[str]) -> Path:
    p = tmp_path / "fomc.txt"
    p.write_text("\n".join(dates) + "\n")
    return p


# ── release events ────────────────────────────────────────────────────


def test_release_event_uses_first_print_not_revision(tmp_db: Path) -> None:
    _seed_gdp_and_consensus(tmp_db)
    gdp = ReleaseEvent(
        "GDPC1", "Real GDP", template_id="gdp",
        consensus_series_id="SPF_RGDP_MEDIAN_NOWCAST",
    )
    with connect(tmp_db, read_only=True) as conn:
        events, results = build_release_events(conn, releases=(gdp,))

    assert len(events) == 1
    ev = events[0]
    assert ev.actual_value == 100.0  # advance print, not the 102.0 revision
    assert ev.consensus_value == 99.0
    assert ev.surprise == 100.0 - 99.0
    assert ev.template_id == "gdp"
    assert ev.event_id == "GDPC1:2024-01-01"
    assert results[0].events == 1
    assert results[0].with_consensus == 1


def test_release_event_without_consensus_leaves_surprise_null(tmp_db: Path) -> None:
    _seed_cpi(tmp_db)
    cpi = ReleaseEvent("CPIAUCSL", "CPI (headline)", template_id="cpi_yoy")
    with connect(tmp_db, read_only=True) as conn:
        events, results = build_release_events(conn, releases=(cpi,))

    assert len(events) == 1
    assert events[0].actual_value == 314.0
    assert events[0].consensus_value is None
    assert events[0].surprise is None
    assert results[0].with_consensus == 0


def test_release_event_skips_series_not_in_db(tmp_db: Path) -> None:
    missing = ReleaseEvent("DOES_NOT_EXIST", "Nothing")
    with connect(tmp_db, read_only=True) as conn:
        events, results = build_release_events(conn, releases=(missing,))
    assert events == []
    assert results[0].events == 0
    assert results[0].success is True


# ── FOMC events ───────────────────────────────────────────────────────


def test_build_fomc_events_classifies_decisions(tmp_db: Path, tmp_path: Path) -> None:
    dates = ["2024-01-31", "2024-03-20", "2024-05-01"]
    _seed_fed_rates(tmp_db, {"2024-01-31": 5.5, "2024-03-20": 5.25, "2024-05-01": 5.25})
    cal = _write_calendar(tmp_path, dates)

    events, result = build_fomc_events(
        "2024-01-01", "2024-12-31", db_path=tmp_db, calendar_path=cal
    )

    assert result.events == 3
    by_date = {e.event_id: e for e in events}
    assert "Decision: unknown" in by_date["fed_decision:2024-01-31"].notes  # no prior
    assert by_date["fed_decision:2024-01-31"].actual_value == 5.5
    assert "Decision: cut" in by_date["fed_decision:2024-03-20"].notes
    assert "Decision: hold" in by_date["fed_decision:2024-05-01"].notes
    assert all(e.template_id == "fed_decision" for e in events)


# ── full orchestrator ─────────────────────────────────────────────────


def test_run_calendar_ingest_is_idempotent(tmp_db: Path, tmp_path: Path) -> None:
    _seed_gdp_and_consensus(tmp_db)
    _seed_cpi(tmp_db)
    _seed_fed_rates(tmp_db, {"2024-01-31": 5.5, "2024-03-20": 5.25})
    cal = _write_calendar(tmp_path, ["2024-01-31", "2024-03-20"])

    def _count() -> int:
        with connect(tmp_db, read_only=True) as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM event_calendar").fetchone()
            return int(row["c"])

    kwargs = {"start": "2024-01-01", "end": "2024-12-31", "db_path": tmp_db, "calendar_path": cal}
    r1 = run_calendar_ingest(**kwargs)
    first = _count()
    r2 = run_calendar_ingest(**kwargs)
    second = _count()

    assert first == second  # no duplicate rows on re-run
    assert r1.total_events == r2.total_events
    # 1 GDP + 1 CPI + 2 FOMC = 4
    assert first == 4
    assert r1.total_with_consensus == 1


def test_run_calendar_ingest_records_audit_run(tmp_db: Path, tmp_path: Path) -> None:
    _seed_fed_rates(tmp_db, {"2024-01-31": 5.5})
    cal = _write_calendar(tmp_path, ["2024-01-31"])
    run_calendar_ingest(start="2024-01-01", end="2024-12-31", db_path=tmp_db, calendar_path=cal)
    with connect(tmp_db, read_only=True) as conn:
        run = conn.execute(
            "SELECT source, status, rows_added FROM ingest_runs WHERE source='calendar'"
        ).fetchone()
    assert run is not None
    assert run["status"] == "ok"
    assert run["rows_added"] == 1


def test_fomc_calendar_prefers_db_after_ingest(tmp_db: Path, tmp_path: Path) -> None:
    """Closes the Phase 2 loop: once event_calendar has fed_decision rows,
    fomc_meeting_dates(prefer_db=True) reads them instead of the file."""
    _seed_fed_rates(tmp_db, {"2024-01-31": 5.5, "2024-03-20": 5.25})
    cal = _write_calendar(tmp_path, ["2024-01-31", "2024-03-20"])
    run_calendar_ingest(start="2024-01-01", end="2024-12-31", db_path=tmp_db, calendar_path=cal)

    # A DIFFERENT static file (empty) proves the dates came from the DB.
    sub = tmp_path / "sub"
    sub.mkdir(exist_ok=True)
    empty_cal = _write_calendar(sub, [])
    got = fomc_meeting_dates(
        "2024-01-01", "2024-12-31", db_path=tmp_db, calendar_path=empty_cal, prefer_db=True
    )
    assert got == (date(2024, 1, 31), date(2024, 3, 20))
