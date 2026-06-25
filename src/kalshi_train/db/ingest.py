"""Low-level DB ingestion helpers.

These wrap the SQL inserts so callers don't write raw SQL. They are
deliberately small and dumb — actual data-source clients live in
``kalshi_train.data.sources`` and call these functions to persist what
they fetch.

Two flavors exist for ``series_observations``:

- ``upsert_observation`` — single row, the path used by interactive
  scripts and tests.
- ``bulk_insert_observations`` — many rows at once with proper
  conflict handling, used by real ingestors that pull thousands of rows
  per call.

We keep timestamps in ISO-8601 strings (UTC). SQLite has no native
datetime type and ISO strings sort lexicographically, which is the
cleanest portable representation.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields
from datetime import UTC, date, datetime
from typing import Any

DateLike = date | datetime | str


def _to_iso_date(value: DateLike) -> str:
    """Normalize a date-ish argument into an ISO date string ('YYYY-MM-DD').

    Accepts ``date``, ``datetime``, or pre-formatted strings. Rejects
    anything that doesn't parse, to fail fast instead of silently
    accepting bad input that would later corrupt our PIT queries.
    """
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    # Accept either a full ISO timestamp or a bare date string.
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()
    except ValueError as e:
        raise ValueError(f"Could not parse date from {value!r}: {e}") from e


def _to_iso_datetime(value: DateLike) -> str:
    """Normalize a date-ish argument into an ISO-8601 datetime string."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).isoformat()
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=UTC).isoformat()
    # String input — accept either date or full timestamp form.
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as e:
        raise ValueError(f"Could not parse datetime from {value!r}: {e}") from e
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat()


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


# ── series_definitions ────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SeriesDefinition:
    """Metadata for a single numeric series.

    `revises=True` means the series may be restated after release (CPI,
    GDP, NFP, etc.) and we should track vintages. `revises=False` means
    the value is fixed at release (e.g. Treasury yields — the closing
    yield on a specific day never changes).
    """

    series_id: str
    source: str
    title: str
    frequency: str
    units: str = ""
    seasonal_adjustment: str = ""
    revises: bool = False
    category: str = ""
    notes: str = ""


def upsert_series_definition(conn: sqlite3.Connection, defn: SeriesDefinition) -> None:
    """Insert or update a series definition. Idempotent."""
    conn.execute(
        """
        INSERT INTO series_definitions (
            series_id, source, title, units, frequency, seasonal_adjustment,
            revises, category, notes, last_ingested_at
        ) VALUES (
            :series_id, :source, :title, :units, :frequency, :seasonal_adjustment,
            :revises, :category, :notes, :now
        )
        ON CONFLICT(series_id) DO UPDATE SET
            source              = excluded.source,
            title               = excluded.title,
            units               = excluded.units,
            frequency           = excluded.frequency,
            seasonal_adjustment = excluded.seasonal_adjustment,
            revises             = excluded.revises,
            category            = excluded.category,
            notes               = excluded.notes,
            last_ingested_at    = excluded.last_ingested_at
        """,
        {
            "series_id": defn.series_id,
            "source": defn.source,
            "title": defn.title,
            "units": defn.units,
            "frequency": defn.frequency,
            "seasonal_adjustment": defn.seasonal_adjustment,
            "revises": int(defn.revises),
            "category": defn.category,
            "notes": defn.notes,
            "now": _now_iso(),
        },
    )


# ── series_observations ───────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Observation:
    """A single (series, period, vintage) triple with a value.

    Vintages of the same observation_date coexist; the composite primary
    key is (series_id, observation_date, vintage_date).
    """

    series_id: str
    observation_date: DateLike
    release_date: DateLike
    vintage_date: DateLike
    value: float | None
    value_text: str | None = None


def upsert_observation(conn: sqlite3.Connection, obs: Observation) -> None:
    """Upsert one observation. Two rows differing only in vintage_date coexist."""
    conn.execute(
        """
        INSERT INTO series_observations (
            series_id, observation_date, vintage_date, release_date,
            value, value_text, ingested_at
        ) VALUES (
            :series_id, :observation_date, :vintage_date, :release_date,
            :value, :value_text, :ingested_at
        )
        ON CONFLICT(series_id, observation_date, vintage_date) DO UPDATE SET
            release_date = excluded.release_date,
            value        = excluded.value,
            value_text   = excluded.value_text,
            ingested_at  = excluded.ingested_at
        """,
        {
            "series_id": obs.series_id,
            "observation_date": _to_iso_date(obs.observation_date),
            "vintage_date": _to_iso_date(obs.vintage_date),
            "release_date": _to_iso_datetime(obs.release_date),
            "value": obs.value,
            "value_text": obs.value_text if obs.value_text is not None
                          else (None if obs.value is None else f"{obs.value}"),
            "ingested_at": _now_iso(),
        },
    )


def bulk_insert_observations(
    conn: sqlite3.Connection, observations: Iterable[Observation]
) -> int:
    """Insert many observations at once. Returns count of rows attempted."""
    rows = [
        {
            "series_id": o.series_id,
            "observation_date": _to_iso_date(o.observation_date),
            "vintage_date": _to_iso_date(o.vintage_date),
            "release_date": _to_iso_datetime(o.release_date),
            "value": o.value,
            "value_text": o.value_text if o.value_text is not None
                          else (None if o.value is None else f"{o.value}"),
            "ingested_at": _now_iso(),
        }
        for o in observations
    ]
    if not rows:
        return 0
    conn.executemany(
        """
        INSERT INTO series_observations (
            series_id, observation_date, vintage_date, release_date,
            value, value_text, ingested_at
        ) VALUES (
            :series_id, :observation_date, :vintage_date, :release_date,
            :value, :value_text, :ingested_at
        )
        ON CONFLICT(series_id, observation_date, vintage_date) DO UPDATE SET
            release_date = excluded.release_date,
            value        = excluded.value,
            value_text   = excluded.value_text,
            ingested_at  = excluded.ingested_at
        """,
        rows,
    )
    return len(rows)


# ── event_calendar ─────────────────────────────────────────────────────


@dataclass(slots=True)
class EventRow:
    """One scheduled economic event with optional consensus / actual / surprise.

    ``event_id`` is a stable natural key so re-ingesting the same event
    updates in place rather than duplicating. We build it from the source
    series and the period it describes (e.g. ``"CPIAUCSL:2024-09-01"``) or,
    for FOMC decisions, ``"fed_decision:2024-09-18"``.

    ``surprise`` is intentionally *not* auto-derived here — the caller
    decides whether ``actual - consensus`` is a meaningful comparison for
    that event (it only is when both are in the same units / frequency).
    """

    event_id: str
    event_name: str
    release_date: DateLike
    series_id: str | None = None
    template_id: str | None = None
    observation_date: DateLike | None = None
    consensus_value: float | None = None
    actual_value: float | None = None
    surprise: float | None = None
    country: str = "US"
    notes: str = ""


def _event_params(row: EventRow) -> dict[str, Any]:
    return {
        "event_id": row.event_id,
        "event_name": row.event_name,
        "series_id": row.series_id,
        "template_id": row.template_id,
        "release_date": _to_iso_datetime(row.release_date),
        "observation_date": (
            _to_iso_date(row.observation_date)
            if row.observation_date is not None
            else None
        ),
        "consensus_value": row.consensus_value,
        "actual_value": row.actual_value,
        "surprise": row.surprise,
        "country": row.country,
        "notes": row.notes,
        "ingested_at": _now_iso(),
    }


_EVENT_UPSERT_SQL = """
    INSERT INTO event_calendar (
        event_id, event_name, series_id, template_id, release_date,
        observation_date, consensus_value, actual_value, surprise,
        country, notes, ingested_at
    ) VALUES (
        :event_id, :event_name, :series_id, :template_id, :release_date,
        :observation_date, :consensus_value, :actual_value, :surprise,
        :country, :notes, :ingested_at
    )
    ON CONFLICT(event_id) DO UPDATE SET
        event_name       = excluded.event_name,
        series_id        = excluded.series_id,
        template_id      = excluded.template_id,
        release_date     = excluded.release_date,
        observation_date = excluded.observation_date,
        consensus_value  = excluded.consensus_value,
        actual_value     = excluded.actual_value,
        surprise         = excluded.surprise,
        country          = excluded.country,
        notes            = excluded.notes,
        ingested_at      = excluded.ingested_at
"""


def upsert_event(conn: sqlite3.Connection, row: EventRow) -> None:
    """Insert or update a single calendar event. Idempotent on ``event_id``."""
    conn.execute(_EVENT_UPSERT_SQL, _event_params(row))


def bulk_insert_events(conn: sqlite3.Connection, rows: Iterable[EventRow]) -> int:
    """Insert many calendar events at once. Returns count of rows attempted."""
    params = [_event_params(r) for r in rows]
    if not params:
        return 0
    conn.executemany(_EVENT_UPSERT_SQL, params)
    return len(params)


# ── ingest_runs (audit) ────────────────────────────────────────────────


@dataclass(slots=True)
class IngestRun:
    """One ingest invocation's audit record."""

    source: str
    target: str
    started_at: str = field(default_factory=_now_iso)
    finished_at: str | None = None
    status: str = "running"
    rows_added: int = 0
    rows_updated: int = 0
    error_message: str = ""


def record_ingest_run(conn: sqlite3.Connection, run: IngestRun) -> int:
    """Insert an ingest-run row; returns the auto-generated run_id."""
    cur = conn.execute(
        """
        INSERT INTO ingest_runs (
            source, target, started_at, finished_at, status,
            rows_added, rows_updated, error_message
        ) VALUES (
            :source, :target, :started_at, :finished_at, :status,
            :rows_added, :rows_updated, :error_message
        )
        """,
        {
            "source": run.source,
            "target": run.target,
            "started_at": run.started_at,
            "finished_at": run.finished_at,
            "status": run.status,
            "rows_added": run.rows_added,
            "rows_updated": run.rows_updated,
            "error_message": run.error_message,
        },
    )
    return int(cur.lastrowid or 0)


# ── metadata ───────────────────────────────────────────────────────────


def set_metadata(conn: sqlite3.Connection, key: str, value: str | Mapping[str, Any]) -> None:
    """Upsert a key in the generic metadata table. Dicts are JSON-encoded."""
    encoded = json.dumps(value) if isinstance(value, Mapping) else value
    conn.execute(
        "INSERT INTO metadata(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, encoded),
    )


def get_metadata(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
    if row is None:
        return None
    return str(row[0])


# ── text_documents (Phase 1.4) ─────────────────────────────────────────


def sha256_hex(text: str) -> str:
    """Stable hex digest used for ``doc_id`` and ``body_hash``."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class TextDocument:
    """One scraped document destined for ``text_documents``.

    ``doc_id`` defaults to a hash of ``(source, url)`` — stable across
    re-scrapes of the same URL, so re-ingesting updates in place. The FTS
    index is kept in sync automatically by triggers in ``schema.sql``.
    """

    source: str
    document_type: str
    title: str
    published_date: DateLike
    body: str
    url: str = ""
    author: str = ""
    effective_date: DateLike | None = None
    metadata: Mapping[str, Any] | None = None
    doc_id: str | None = None

    def resolved_doc_id(self) -> str:
        if self.doc_id:
            return self.doc_id
        basis = self.url or f"{self.source}|{self.document_type}|{self.title}"
        return sha256_hex(f"{self.source}|{basis}")


def _document_params(doc: TextDocument) -> dict[str, Any]:
    return {
        "doc_id": doc.resolved_doc_id(),
        "source": doc.source,
        "document_type": doc.document_type,
        "title": doc.title,
        "author": doc.author,
        "published_date": _to_iso_date(doc.published_date),
        "effective_date": (
            _to_iso_date(doc.effective_date) if doc.effective_date is not None else None
        ),
        "body": doc.body,
        "body_hash": sha256_hex(doc.body),
        "url": doc.url,
        "metadata_json": json.dumps(dict(doc.metadata)) if doc.metadata else "{}",
        "ingested_at": _now_iso(),
    }


_DOCUMENT_UPSERT_SQL = """
    INSERT INTO text_documents (
        doc_id, source, document_type, title, author, published_date,
        effective_date, body, body_hash, url, metadata_json, ingested_at
    ) VALUES (
        :doc_id, :source, :document_type, :title, :author, :published_date,
        :effective_date, :body, :body_hash, :url, :metadata_json, :ingested_at
    )
    ON CONFLICT(doc_id) DO UPDATE SET
        source        = excluded.source,
        document_type = excluded.document_type,
        title         = excluded.title,
        author        = excluded.author,
        published_date= excluded.published_date,
        effective_date= excluded.effective_date,
        body          = excluded.body,
        body_hash     = excluded.body_hash,
        url           = excluded.url,
        metadata_json = excluded.metadata_json,
        ingested_at   = excluded.ingested_at
"""


def upsert_document(conn: sqlite3.Connection, doc: TextDocument) -> None:
    """Insert or update one text document. Idempotent on ``doc_id``."""
    conn.execute(_DOCUMENT_UPSERT_SQL, _document_params(doc))


def bulk_insert_documents(conn: sqlite3.Connection, docs: Iterable[TextDocument]) -> int:
    """Insert many documents. Returns count attempted.

    We loop rather than ``executemany`` so the per-row FTS sync triggers
    fire cleanly and a single malformed row can't poison the batch.
    """
    n = 0
    for doc in docs:
        conn.execute(_DOCUMENT_UPSERT_SQL, _document_params(doc))
        n += 1
    return n


# ── kalshi_markets / kalshi_price_history (Phase 1.5) ──────────────────


@dataclass(slots=True)
class KalshiMarketRow:
    """Flattened Kalshi market metadata for ``kalshi_markets``."""

    ticker: str
    event_ticker: str
    series_ticker: str = ""
    market_type: str = ""
    title: str = ""
    subtitle: str = ""
    yes_sub_title: str = ""
    no_sub_title: str = ""
    rules_primary: str = ""
    rules_secondary: str = ""
    open_time: str | None = None
    close_time: str | None = None
    created_time: str | None = None
    settlement_time: str | None = None
    status: str = ""
    result: str = ""
    settlement_value_dollars: str | None = None
    template_id: str | None = None
    strike_value: float | None = None
    strike_direction: str = ""
    last_price_dollars: str = "0.0000"
    volume_fp: str = "0.00"
    open_interest_fp: str = "0.00"


_KALSHI_MARKET_UPSERT_SQL = """
    INSERT INTO kalshi_markets (
        ticker, event_ticker, series_ticker, market_type, title, subtitle,
        yes_sub_title, no_sub_title, rules_primary, rules_secondary,
        open_time, close_time, created_time, settlement_time, status, result,
        settlement_value_dollars, template_id, strike_value, strike_direction,
        last_price_dollars, volume_fp, open_interest_fp, ingested_at, last_refreshed_at
    ) VALUES (
        :ticker, :event_ticker, :series_ticker, :market_type, :title, :subtitle,
        :yes_sub_title, :no_sub_title, :rules_primary, :rules_secondary,
        :open_time, :close_time, :created_time, :settlement_time, :status, :result,
        :settlement_value_dollars, :template_id, :strike_value, :strike_direction,
        :last_price_dollars, :volume_fp, :open_interest_fp, :now, :now
    )
    ON CONFLICT(ticker) DO UPDATE SET
        event_ticker             = excluded.event_ticker,
        series_ticker            = excluded.series_ticker,
        market_type              = excluded.market_type,
        title                    = excluded.title,
        subtitle                 = excluded.subtitle,
        yes_sub_title            = excluded.yes_sub_title,
        no_sub_title             = excluded.no_sub_title,
        rules_primary            = excluded.rules_primary,
        rules_secondary          = excluded.rules_secondary,
        open_time                = excluded.open_time,
        close_time               = excluded.close_time,
        created_time             = excluded.created_time,
        settlement_time          = excluded.settlement_time,
        status                   = excluded.status,
        result                   = excluded.result,
        settlement_value_dollars = excluded.settlement_value_dollars,
        template_id              = excluded.template_id,
        strike_value             = excluded.strike_value,
        strike_direction         = excluded.strike_direction,
        last_price_dollars       = excluded.last_price_dollars,
        volume_fp                = excluded.volume_fp,
        open_interest_fp         = excluded.open_interest_fp,
        last_refreshed_at        = excluded.last_refreshed_at
"""


def upsert_kalshi_market(conn: sqlite3.Connection, market: KalshiMarketRow) -> None:
    """Insert or update one Kalshi market. ``ingested_at`` is preserved on
    update (ON CONFLICT keeps the original row's value untouched except for
    the columns listed), while ``last_refreshed_at`` advances."""
    params = {f.name: getattr(market, f.name) for f in fields(market)}
    params["now"] = _now_iso()
    conn.execute(_KALSHI_MARKET_UPSERT_SQL, params)


def bulk_insert_kalshi_markets(
    conn: sqlite3.Connection, markets: Iterable[KalshiMarketRow]
) -> int:
    n = 0
    for m in markets:
        upsert_kalshi_market(conn, m)
        n += 1
    return n


@dataclass(slots=True)
class KalshiPriceRow:
    """One candlestick row for ``kalshi_price_history``."""

    ticker: str
    period_end_ts: int
    period_end_date: str
    open_dollars: str | None = None
    high_dollars: str | None = None
    low_dollars: str | None = None
    close_dollars: str | None = None
    mean_dollars: str | None = None
    yes_bid_close: str | None = None
    yes_ask_close: str | None = None
    volume_fp: str | None = None
    open_interest_fp: str | None = None


def bulk_insert_kalshi_prices(
    conn: sqlite3.Connection, rows: Iterable[KalshiPriceRow]
) -> int:
    params = [{f.name: getattr(r, f.name) for f in fields(r)} for r in rows]
    if not params:
        return 0
    conn.executemany(
        """
        INSERT INTO kalshi_price_history (
            ticker, period_end_ts, period_end_date, open_dollars, high_dollars,
            low_dollars, close_dollars, mean_dollars, yes_bid_close, yes_ask_close,
            volume_fp, open_interest_fp
        ) VALUES (
            :ticker, :period_end_ts, :period_end_date, :open_dollars, :high_dollars,
            :low_dollars, :close_dollars, :mean_dollars, :yes_bid_close, :yes_ask_close,
            :volume_fp, :open_interest_fp
        )
        ON CONFLICT(ticker, period_end_ts) DO UPDATE SET
            period_end_date  = excluded.period_end_date,
            open_dollars     = excluded.open_dollars,
            high_dollars     = excluded.high_dollars,
            low_dollars      = excluded.low_dollars,
            close_dollars    = excluded.close_dollars,
            mean_dollars     = excluded.mean_dollars,
            yes_bid_close    = excluded.yes_bid_close,
            yes_ask_close    = excluded.yes_ask_close,
            volume_fp        = excluded.volume_fp,
            open_interest_fp = excluded.open_interest_fp
        """,
        params,
    )
    return len(params)


# ── polymarket (Phase 1.5) ─────────────────────────────────────────────


@dataclass(slots=True)
class PolymarketMarketRow:
    """Flattened Polymarket market for ``polymarket_markets``."""

    condition_id: str
    slug: str = ""
    question: str = ""
    description: str = ""
    template_id: str | None = None
    strike_value: float | None = None
    strike_direction: str = ""
    start_date: str | None = None
    end_date: str | None = None
    resolved: bool = False
    outcome: str = ""
    metadata: Mapping[str, Any] | None = None


def upsert_polymarket_market(
    conn: sqlite3.Connection, market: PolymarketMarketRow
) -> None:
    conn.execute(
        """
        INSERT INTO polymarket_markets (
            condition_id, slug, question, description, template_id, strike_value,
            strike_direction, start_date, end_date, resolved, outcome,
            metadata_json, ingested_at
        ) VALUES (
            :condition_id, :slug, :question, :description, :template_id, :strike_value,
            :strike_direction, :start_date, :end_date, :resolved, :outcome,
            :metadata_json, :ingested_at
        )
        ON CONFLICT(condition_id) DO UPDATE SET
            slug             = excluded.slug,
            question         = excluded.question,
            description      = excluded.description,
            template_id      = excluded.template_id,
            strike_value     = excluded.strike_value,
            strike_direction = excluded.strike_direction,
            start_date       = excluded.start_date,
            end_date         = excluded.end_date,
            resolved         = excluded.resolved,
            outcome          = excluded.outcome,
            metadata_json    = excluded.metadata_json,
            ingested_at      = excluded.ingested_at
        """,
        {
            "condition_id": market.condition_id,
            "slug": market.slug,
            "question": market.question,
            "description": market.description,
            "template_id": market.template_id,
            "strike_value": market.strike_value,
            "strike_direction": market.strike_direction,
            "start_date": market.start_date,
            "end_date": market.end_date,
            "resolved": int(market.resolved),
            "outcome": market.outcome,
            "metadata_json": json.dumps(dict(market.metadata)) if market.metadata else "{}",
            "ingested_at": _now_iso(),
        },
    )


def bulk_insert_polymarket_markets(
    conn: sqlite3.Connection, markets: Iterable[PolymarketMarketRow]
) -> int:
    n = 0
    for m in markets:
        upsert_polymarket_market(conn, m)
        n += 1
    return n
