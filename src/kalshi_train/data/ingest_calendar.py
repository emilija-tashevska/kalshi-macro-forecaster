"""Phase 1.6 — economic-event calendar ingestion.

This orchestrator populates ``event_calendar`` from data we already hold,
so it needs **no new API key** and is fully deterministic:

1. **Economic releases** (``build_release_events``). For every series in
   ``calendar_registry.CALENDAR_RELEASES`` that exists in
   ``series_observations``, each period's *first print* (earliest vintage
   for that ``observation_date``) becomes one calendar row. Where a
   same-frequency consensus series is registered (real GDP → SPF nowcast),
   we attach ``consensus_value`` and compute ``surprise = actual - consensus``.

2. **FOMC decisions** (``build_fomc_events``). From the FOMC meeting
   schedule (the static file shipped with the repo), one row per meeting.
   The "actual" is the post-meeting Fed funds target upper bound
   (``DFEDTARU``) as known on the meeting date, and we classify the
   decision as cut / hold / hike by comparing to the prior meeting.

Running this also closes the loop with Phase 2: once ``event_calendar``
holds ``fed_decision`` rows, ``fomc_calendar.fomc_meeting_dates`` prefers
them over the static fallback.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

from kalshi_train.data.calendar_registry import (
    CALENDAR_RELEASES,
    FOMC_EVENT_NAME,
    FOMC_RATE_SERIES,
    FOMC_TEMPLATE_ID,
    ReleaseEvent,
)
from kalshi_train.data.fomc_calendar import DEFAULT_CALENDAR_PATH, fomc_meeting_dates
from kalshi_train.db.connection import connect
from kalshi_train.db.ingest import (
    EventRow,
    IngestRun,
    bulk_insert_events,
    record_ingest_run,
)
from kalshi_train.db.point_in_time import pit_value

logger = logging.getLogger(__name__)

DateLike = date | datetime | str


# ── Report dataclasses ────────────────────────────────────────────────


@dataclass(slots=True)
class CalendarGroupResult:
    """Outcome for one logical group of events (a series, or "FOMC")."""

    name: str
    events: int = 0
    with_consensus: int = 0
    success: bool = True
    error: str | None = None


@dataclass(slots=True)
class CalendarIngestReport:
    started_at: datetime
    finished_at: datetime | None
    results: list[CalendarGroupResult] = field(default_factory=list)

    @property
    def total_events(self) -> int:
        return sum(r.events for r in self.results)

    @property
    def total_with_consensus(self) -> int:
        return sum(r.with_consensus for r in self.results)

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.results if not r.success)


# ── DB read helpers ───────────────────────────────────────────────────


def _first_prints(
    conn: sqlite3.Connection, series_id: str
) -> list[tuple[str, str, float]]:
    """Return ``(observation_date, release_date, value)`` for each period's
    *first* reported vintage of ``series_id``.

    The first print is the row with the minimum ``vintage_date`` for a
    given ``observation_date`` — i.e. "the number as the market first saw
    it", before any revisions. Rows whose value is NULL are skipped.
    """
    rows = conn.execute(
        """
        SELECT o.observation_date AS observation_date,
               o.release_date     AS release_date,
               o.value            AS value
        FROM series_observations o
        WHERE o.series_id = :sid
          AND o.value IS NOT NULL
          AND o.vintage_date = (
              SELECT MIN(o2.vintage_date)
              FROM series_observations o2
              WHERE o2.series_id = :sid
                AND o2.observation_date = o.observation_date
          )
        ORDER BY o.observation_date
        """,
        {"sid": series_id},
    ).fetchall()
    return [(r["observation_date"], r["release_date"], float(r["value"])) for r in rows]


def _first_print_by_obs_date(
    conn: sqlite3.Connection, series_id: str
) -> dict[str, float]:
    """First-print value of ``series_id`` keyed by ``observation_date``.

    Used to look up the consensus forecast for the same period as a
    release. SPF series are single-vintage, so "first print" is just the
    stored value.
    """
    return {obs: val for obs, _rel, val in _first_prints(conn, series_id)}


# ── Builders ──────────────────────────────────────────────────────────


def build_release_events(
    conn: sqlite3.Connection,
    releases: tuple[ReleaseEvent, ...] = CALENDAR_RELEASES,
) -> tuple[list[EventRow], list[CalendarGroupResult]]:
    """Build calendar rows for every registered economic release present."""
    events: list[EventRow] = []
    results: list[CalendarGroupResult] = []

    for rel in releases:
        try:
            prints = _first_prints(conn, rel.series_id)
        except sqlite3.Error as exc:  # pragma: no cover - defensive
            logger.exception("Reading first prints for %s failed", rel.series_id)
            results.append(
                CalendarGroupResult(name=rel.series_id, success=False, error=str(exc))
            )
            continue

        consensus_by_obs: dict[str, float] = {}
        if rel.consensus_series_id:
            consensus_by_obs = _first_print_by_obs_date(conn, rel.consensus_series_id)

        n_consensus = 0
        for obs_date, release_date, value in prints:
            consensus = consensus_by_obs.get(obs_date)
            surprise = (value - consensus) if consensus is not None else None
            if consensus is not None:
                n_consensus += 1
            events.append(
                EventRow(
                    event_id=f"{rel.series_id}:{obs_date}",
                    event_name=rel.event_name,
                    series_id=rel.series_id,
                    template_id=rel.template_id,
                    release_date=release_date,
                    observation_date=obs_date,
                    consensus_value=consensus,
                    actual_value=value,
                    surprise=surprise,
                    notes=rel.notes,
                )
            )

        results.append(
            CalendarGroupResult(
                name=rel.series_id,
                events=len(prints),
                with_consensus=n_consensus,
            )
        )
        logger.info(
            "  → %s: %d release events (%d with consensus)",
            rel.series_id,
            len(prints),
            n_consensus,
        )

    return events, results


def _classify_decision(rate_before: float | None, rate_after: float | None) -> str:
    if rate_before is None or rate_after is None:
        return "unknown"
    if rate_after < rate_before:
        return "cut"
    if rate_after > rate_before:
        return "hike"
    return "hold"


def build_fomc_events(
    start: DateLike,
    end: DateLike,
    *,
    db_path: Path | None = None,
    calendar_path: Path = DEFAULT_CALENDAR_PATH,
) -> tuple[list[EventRow], CalendarGroupResult]:
    """Build one ``fed_decision`` calendar row per FOMC meeting in range.

    We read the meeting schedule from the static file (``prefer_db=False``
    so we don't depend on the very table we're populating) and resolve the
    post-meeting target rate via the point-in-time interface, exactly as
    the Phase 2 Fed-cut target does.
    """
    meetings = fomc_meeting_dates(
        start, end, db_path=db_path, calendar_path=calendar_path, prefer_db=False
    )
    events: list[EventRow] = []
    prev_rate: float | None = None

    for meeting in meetings:
        rate_after = pit_value(FOMC_RATE_SERIES, meeting, db_path=db_path)
        decision = _classify_decision(prev_rate, rate_after)
        note = (
            f"Decision: {decision}."
            + (f" Target upper {prev_rate}→{rate_after}." if rate_after is not None else "")
        )
        events.append(
            EventRow(
                event_id=f"{FOMC_TEMPLATE_ID}:{meeting.isoformat()}",
                event_name=FOMC_EVENT_NAME,
                series_id=FOMC_RATE_SERIES,
                template_id=FOMC_TEMPLATE_ID,
                release_date=meeting,
                observation_date=meeting,
                consensus_value=None,
                actual_value=rate_after,
                surprise=None,
                notes=note,
            )
        )
        if rate_after is not None:
            prev_rate = rate_after

    result = CalendarGroupResult(name="FOMC", events=len(events))
    logger.info("  → FOMC: %d decision events", len(events))
    return events, result


# ── Top-level orchestrator ────────────────────────────────────────────


def run_calendar_ingest(
    *,
    start: DateLike = "2000-01-01",
    end: DateLike | None = None,
    db_path: Path | None = None,
    calendar_path: Path = DEFAULT_CALENDAR_PATH,
    include_releases: bool = True,
    include_fomc: bool = True,
) -> CalendarIngestReport:
    """Populate ``event_calendar`` from internal data. Idempotent."""
    started_at = datetime.now(tz=UTC)
    end = end or date.today().isoformat()

    audit_id = 0
    with connect(db_path) as conn:
        audit_id = record_ingest_run(
            conn,
            IngestRun(
                source="calendar",
                target=f"{start}..{end}",
                started_at=started_at.isoformat(),
                status="running",
            ),
        )
        conn.commit()

    all_events: list[EventRow] = []
    results: list[CalendarGroupResult] = []

    if include_releases:
        with connect(db_path, read_only=True) as conn:
            rel_events, rel_results = build_release_events(conn)
        all_events.extend(rel_events)
        results.extend(rel_results)

    if include_fomc:
        fomc_events, fomc_result = build_fomc_events(
            start, end, db_path=db_path, calendar_path=calendar_path
        )
        all_events.extend(fomc_events)
        results.append(fomc_result)

    with connect(db_path) as conn:
        # event_calendar.series_id is a FK into series_definitions. FOMC
        # events reference DFEDTARU, which may not be ingested yet on a
        # fresh DB — drop the link (NULL) rather than violate the FK.
        known = {
            row["series_id"]
            for row in conn.execute("SELECT series_id FROM series_definitions")
        }
        for ev in all_events:
            if ev.series_id is not None and ev.series_id not in known:
                ev.series_id = None
        bulk_insert_events(conn, all_events)
        conn.commit()

    finished_at = datetime.now(tz=UTC)
    report = CalendarIngestReport(
        started_at=started_at, finished_at=finished_at, results=results
    )

    with connect(db_path) as conn:
        conn.execute(
            """
            UPDATE ingest_runs
               SET finished_at = ?, status = ?, rows_added = ?, error_message = ?
             WHERE run_id = ?
            """,
            (
                finished_at.isoformat(),
                "ok" if report.n_failed == 0 else "partial",
                report.total_events,
                "; ".join(
                    f"{r.name}: {r.error}" for r in report.results if not r.success
                )[:2000],
                audit_id,
            ),
        )
        conn.commit()

    logger.info(
        "Calendar ingest complete: %d events (%d with consensus), %d groups failed",
        report.total_events,
        report.total_with_consensus,
        report.n_failed,
    )
    return report


__all__ = [
    "CalendarGroupResult",
    "CalendarIngestReport",
    "build_fomc_events",
    "build_release_events",
    "run_calendar_ingest",
]
