"""Phase 1.4 — text-corpus ingestion (Federal Reserve communications).

We ingest the three highest-signal, machine-addressable Fed sources:

  - ``fomc_statement`` — the post-meeting policy statement
  - ``fomc_minutes``   — the detailed minutes (released ~3 weeks later)
  - ``beige_book``     — the regional economic conditions summary (8/yr)

Rather than crawl index pages (whose layout the Fed changes often), we
build *deterministic candidate URLs* and probe them. Statement and
minutes URLs are derived from the FOMC meeting schedule (reusing the
Phase 1.6 calendar); Beige Book URLs are probed across all months. The
client returns ``None`` for URLs that don't exist yet, so probing is
safe and self-limiting.

Speeches, SEP projections, and non-Fed central banks (ECB/BoE) are
deliberately out of scope for this first pass; the source-dispatch design
below makes adding them a matter of one more URL builder + parser.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from kalshi_train.data.fomc_calendar import DEFAULT_CALENDAR_PATH, fomc_meeting_dates
from kalshi_train.data.sources.fed_text import BASE_URL, FedTextClient, build_document
from kalshi_train.db.connection import connect
from kalshi_train.db.ingest import (
    IngestRun,
    TextDocument,
    bulk_insert_documents,
    record_ingest_run,
)

logger = logging.getLogger(__name__)

DateLike = date | datetime | str

SOURCE = "fed"
DOC_FOMC_STATEMENT = "fomc_statement"
DOC_FOMC_MINUTES = "fomc_minutes"
DOC_BEIGE_BOOK = "beige_book"
ALL_DOC_TYPES = (DOC_FOMC_STATEMENT, DOC_FOMC_MINUTES, DOC_BEIGE_BOOK)

# Minutes publish roughly three weeks after the meeting. We don't parse
# the exact release date out of the HTML (brittle); this approximation is
# only used for ``published_date`` and is documented as such.
_MINUTES_PUBLICATION_LAG = timedelta(days=21)


@dataclass(frozen=True, slots=True)
class TextCandidate:
    """One logical document and the ordered URLs to try for it.

    The Fed's URL scheme changed across eras and 2-day meetings publish on
    the *last* day (which the calendar may list as the first), so each
    document carries several candidate URLs. The orchestrator fetches them
    in order and keeps the first that yields a real page.
    """

    document_type: str
    urls: tuple[str, ...]
    published_date: date
    effective_date: date | None = None
    title: str | None = None
    min_body_chars: int = 200


@dataclass(slots=True)
class TextSourceResult:
    document_type: str
    fetched: int = 0
    stored: int = 0
    missing: int = 0
    success: bool = True
    error: str | None = None


@dataclass(slots=True)
class TextIngestReport:
    started_at: datetime
    finished_at: datetime | None
    results: list[TextSourceResult] = field(default_factory=list)

    @property
    def total_stored(self) -> int:
        return sum(r.stored for r in self.results)

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.results if not r.success)


# ── URL builders ───────────────────────────────────────────────────────


def fomc_statement_url(meeting: date) -> str:
    """Modern (primary) statement URL for a meeting date."""
    return f"/newsevents/pressreleases/monetary{meeting:%Y%m%d}a.htm"


def fomc_minutes_url(meeting: date) -> str:
    """Modern (primary) minutes URL for a meeting date."""
    return f"/monetarypolicy/fomcminutes{meeting:%Y%m%d}.htm"


def beige_book_url(year: int, month: int) -> str:
    """Modern (2017+) Beige Book URL for a year/month."""
    return f"/monetarypolicy/beigebook{year:04d}{month:02d}.htm"


def beige_book_urls(year: int, month: int) -> tuple[str, ...]:
    """Beige Book URL variants: modern (2017+) then legacy (2011-2016).

    Pre-2011 issues are JS-rendered single-page apps (a server GET returns
    only a table of contents), so they're not recoverable this way and are
    filtered out by the body-length minimum.
    """
    return (
        f"/monetarypolicy/beigebook{year:04d}{month:02d}.htm",
        f"/monetarypolicy/beigebook/beigebook{year:04d}{month:02d}.htm",
    )


# Beige Book TOC stubs (pre-2011, JS-rendered) are ~1.3k chars; real
# reports are >=3.5k. This threshold keeps reports and drops the stubs.
BEIGE_MIN_BODY_CHARS = 2000


def _meeting_days(meeting: date) -> tuple[date, ...]:
    """The two possible publication days: the listed date and the next day.

    The calendar lists meeting *start* dates; statements/minutes are
    published on the last day, which for 2-day meetings is start + 1.
    """
    return (meeting, meeting + timedelta(days=1))


def fomc_statement_urls(meeting: date) -> tuple[str, ...]:
    """All statement URL variants to try, modern scheme first."""
    days = _meeting_days(meeting)
    modern = [f"/newsevents/pressreleases/monetary{d:%Y%m%d}a.htm" for d in days]
    legacy = [f"/boarddocs/press/monetary/{d:%Y}/{d:%Y%m%d}/default.htm" for d in days]
    # Some 2001 intermeeting actions live under press/general.
    general = [f"/boarddocs/press/general/{d:%Y}/{d:%Y%m%d}/default.htm" for d in days]
    return tuple(modern + legacy + general)


def fomc_minutes_urls(meeting: date) -> tuple[str, ...]:
    """All minutes URL variants to try, modern scheme first."""
    days = _meeting_days(meeting)
    modern = [f"/monetarypolicy/fomcminutes{d:%Y%m%d}.htm" for d in days]
    legacy = [f"/fomc/minutes/{d:%Y%m%d}.htm" for d in days]
    return tuple(modern + legacy)


def build_candidates(
    *,
    start: DateLike,
    end: DateLike,
    document_types: Sequence[str],
    db_path: Path | None = None,
    calendar_path: Path = DEFAULT_CALENDAR_PATH,
) -> list[TextCandidate]:
    """Generate candidate URLs for the requested document types in range."""
    start_d = _as_date(start)
    end_d = _as_date(end)
    candidates: list[TextCandidate] = []

    meetings: tuple[date, ...] = ()
    if DOC_FOMC_STATEMENT in document_types or DOC_FOMC_MINUTES in document_types:
        meetings = fomc_meeting_dates(
            start_d, end_d, db_path=db_path, calendar_path=calendar_path
        )

    if DOC_FOMC_STATEMENT in document_types:
        candidates += [
            TextCandidate(
                document_type=DOC_FOMC_STATEMENT,
                urls=fomc_statement_urls(m),
                published_date=m,
                effective_date=m,
                title=f"FOMC Statement — {m:%B %d, %Y}",
            )
            for m in meetings
        ]

    if DOC_FOMC_MINUTES in document_types:
        candidates += [
            TextCandidate(
                document_type=DOC_FOMC_MINUTES,
                urls=fomc_minutes_urls(m),
                published_date=m + _MINUTES_PUBLICATION_LAG,
                effective_date=m,
                title=f"FOMC Minutes — {m:%B %d, %Y}",
            )
            for m in meetings
        ]

    if DOC_BEIGE_BOOK in document_types:
        for year in range(start_d.year, end_d.year + 1):
            for month in range(1, 13):
                pub = date(year, month, 1)
                if not (start_d <= pub <= end_d):
                    continue
                candidates.append(
                    TextCandidate(
                        document_type=DOC_BEIGE_BOOK,
                        urls=beige_book_urls(year, month),
                        published_date=pub,
                        title=f"Beige Book — {pub:%B %Y}",
                        min_body_chars=BEIGE_MIN_BODY_CHARS,
                    )
                )

    return candidates


# ── Orchestrator ───────────────────────────────────────────────────────


async def _fetch_first(
    client: FedTextClient, cand: TextCandidate, res: TextSourceResult
) -> TextDocument | None:
    """Try each candidate URL in order; return the first real document.

    A network error on one URL is recorded but doesn't abort the others.
    """
    for url in cand.urls:
        try:
            html = await client.fetch(url)
        except Exception as exc:
            logger.warning("Fetch failed for %s: %s", url, exc)
            res.success = False
            res.error = str(exc)
            continue
        if html is None:
            continue
        abs_url = url if url.startswith("http") else f"{BASE_URL}{url}"
        doc = build_document(
            html,
            source=SOURCE,
            document_type=cand.document_type,
            url=abs_url,
            published_date=cand.published_date,
            effective_date=cand.effective_date,
            title=cand.title,
            min_body_chars=cand.min_body_chars,
        )
        if doc is not None:
            return doc
    return None


async def run_text_ingest(
    *,
    start: DateLike = "2000-01-01",
    end: DateLike | None = None,
    document_types: Sequence[str] = ALL_DOC_TYPES,
    db_path: Path | None = None,
    calendar_path: Path = DEFAULT_CALENDAR_PATH,
    client: FedTextClient | None = None,
    limit: int | None = None,
) -> TextIngestReport:
    """Probe + ingest the requested Fed document types into ``text_documents``."""
    started_at = datetime.now(tz=UTC)
    end = end or date.today().isoformat()

    candidates = build_candidates(
        start=start,
        end=end,
        document_types=document_types,
        db_path=db_path,
        calendar_path=calendar_path,
    )
    if limit is not None:
        candidates = candidates[:limit]

    audit_id = 0
    with connect(db_path) as conn:
        audit_id = record_ingest_run(
            conn,
            IngestRun(
                source="text:fed",
                target=",".join(document_types),
                started_at=started_at.isoformat(),
                status="running",
            ),
        )
        conn.commit()

    results: dict[str, TextSourceResult] = {
        dt: TextSourceResult(document_type=dt) for dt in document_types
    }
    docs: list[TextDocument] = []

    owns_client = client is None
    client_ctx = FedTextClient() if client is None else client
    try:
        if owns_client:
            await client_ctx.__aenter__()
        for cand in candidates:
            res = results[cand.document_type]
            doc = await _fetch_first(client_ctx, cand, res)
            if doc is not None:
                docs.append(doc)
                res.fetched += 1
                res.stored += 1
            else:
                res.missing += 1
    finally:
        if owns_client:
            await client_ctx.__aexit__(None, None, None)

    with connect(db_path) as conn:
        bulk_insert_documents(conn, docs)
        conn.commit()

    finished_at = datetime.now(tz=UTC)
    report = TextIngestReport(
        started_at=started_at, finished_at=finished_at, results=list(results.values())
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
                report.total_stored,
                "; ".join(
                    f"{r.document_type}: {r.error}" for r in report.results if not r.success
                )[:2000],
                audit_id,
            ),
        )
        conn.commit()

    logger.info("Text ingest complete: %d documents stored", report.total_stored)
    return report


# ── helpers ────────────────────────────────────────────────────────────


def _as_date(value: DateLike) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


__all__ = [
    "ALL_DOC_TYPES",
    "TextCandidate",
    "TextIngestReport",
    "TextSourceResult",
    "build_candidates",
    "run_text_ingest",
]
