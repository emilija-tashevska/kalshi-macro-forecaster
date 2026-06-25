"""FOMC meeting calendar — decision dates used to build Fed-cut labels.

Phase 1.6 will eventually populate ``event_calendar`` with the same
information. Until then (and as a fallback), we ship a static file at
``data/static/fomc_meeting_dates.txt`` sourced from the Fed's historical
archive plus recent calendar pages.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

import pandas as pd

from kalshi_train.config import PROJECT_ROOT, settings
from kalshi_train.db.connection import connect

DateLike = date | datetime | str

DEFAULT_CALENDAR_PATH = PROJECT_ROOT / "data" / "static" / "fomc_meeting_dates.txt"


def _parse_date(value: DateLike) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(value.replace("Z", "+00:00")).date()


def _load_static_calendar(path: Path = DEFAULT_CALENDAR_PATH) -> tuple[date, ...]:
    if not path.exists():
        raise FileNotFoundError(
            f"FOMC calendar not found at {path}. Expected one ISO date per line."
        )
    dates: list[date] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        dates.append(date.fromisoformat(stripped))
    return tuple(sorted(set(dates)))


def _load_calendar_from_db(
    db_path: Path | None,
    start: date,
    end: date,
) -> tuple[date, ...]:
    """Load FOMC dates from ``event_calendar`` when Phase 1.6 has run.

    Returns ``()`` (so callers fall back to the static file) when the DB
    file does not exist yet or the table hasn't been created — this keeps
    ``prefer_db=True`` safe to use before any ingestion has happened.
    """
    path = db_path or settings.kalshi_train_db_path
    if not Path(path).exists():
        return ()
    try:
        with connect(db_path, read_only=True) as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT date(release_date) AS meeting_date
                FROM event_calendar
                WHERE template_id = 'fed_decision'
                  AND date(release_date) BETWEEN :start AND :end
                ORDER BY meeting_date
                """,
                {"start": start.isoformat(), "end": end.isoformat()},
            ).fetchall()
    except sqlite3.OperationalError:
        return ()
    if not rows:
        return ()
    return tuple(date.fromisoformat(r["meeting_date"]) for r in rows)


@lru_cache(maxsize=4)
def _cached_static_dates(path_str: str) -> tuple[date, ...]:
    return _load_static_calendar(Path(path_str))


def fomc_meeting_dates(
    start: DateLike,
    end: DateLike,
    *,
    db_path: Path | None = None,
    calendar_path: Path = DEFAULT_CALENDAR_PATH,
    prefer_db: bool = True,
) -> tuple[date, ...]:
    """Return sorted FOMC decision dates in ``[start, end]`` inclusive.

    When ``prefer_db`` is True and ``event_calendar`` contains fed_decision
    rows, those take precedence. Otherwise we fall back to the static file.
    """
    start_d = _parse_date(start)
    end_d = _parse_date(end)
    if start_d > end_d:
        raise ValueError(f"start {start_d} is after end {end_d}")

    if prefer_db:
        db_dates = _load_calendar_from_db(db_path, start_d, end_d)
        if db_dates:
            return db_dates

    static = _cached_static_dates(str(calendar_path.resolve()))
    return tuple(d for d in static if start_d <= d <= end_d)


def previous_business_day(value: DateLike) -> date:
    """Return the prior NYSE business day (simple Mon-Fri calendar)."""
    d = _parse_date(value)
    prior = pd.Timestamp(d).normalize() - pd.tseries.offsets.BDay(1)
    result: date = prior.date()
    return result
