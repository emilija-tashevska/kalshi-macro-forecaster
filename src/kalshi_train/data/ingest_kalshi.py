"""Phase 1.5 — Kalshi macro-market ingestion.

Pulls Kalshi markets, keeps only those that match one of our seven macro
question templates (via ``kalshi_macro.classify``), and stores their
metadata in ``kalshi_markets`` plus daily candlestick history in
``kalshi_price_history`` — the "market-implied probability" series we
benchmark our models against.

Two market-discovery modes:

- ``series_tickers`` given → query ``/markets?series_ticker=...`` per
  series (fast, targeted; the normal path).
- otherwise → paginate **all** markets and classify each (exhaustive but
  heavy; useful for discovery / one-off backfills).

Price history is optional (``with_prices``) because it costs one extra
request per market.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from kalshi_train.data.kalshi_macro import classify
from kalshi_train.data.sources.kalshi import KalshiClient
from kalshi_train.data.sources.kalshi_models import Candlestick, KalshiMarketModel
from kalshi_train.db.connection import connect
from kalshi_train.db.ingest import (
    IngestRun,
    KalshiMarketRow,
    KalshiPriceRow,
    bulk_insert_kalshi_markets,
    bulk_insert_kalshi_prices,
    record_ingest_run,
)

logger = logging.getLogger(__name__)

DAILY_INTERVAL = 1440


@dataclass(slots=True)
class KalshiIngestReport:
    started_at: datetime
    finished_at: datetime | None
    markets_seen: int = 0
    markets_stored: int = 0
    price_rows: int = 0
    by_template: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


# ── Mapping helpers ────────────────────────────────────────────────────


def _to_ts(iso: str | None) -> int | None:
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp())


def market_to_row(model: KalshiMarketModel) -> KalshiMarketRow | None:
    """Classify a market and, if macro, flatten it into a DB row."""
    cls = classify(
        ticker=model.ticker,
        event_ticker=model.event_ticker,
        series_ticker=model.series_ticker,
        title=model.title,
        subtitle=model.subtitle,
        yes_sub_title=model.yes_sub_title,
    )
    if cls is None:
        return None
    return KalshiMarketRow(
        ticker=model.ticker,
        event_ticker=model.event_ticker,
        series_ticker=model.derived_series_ticker(),
        market_type=model.market_type,
        title=model.title,
        subtitle=model.subtitle,
        yes_sub_title=model.yes_sub_title,
        no_sub_title=model.no_sub_title,
        rules_primary=model.rules_primary,
        rules_secondary=model.rules_secondary,
        open_time=model.open_time,
        close_time=model.close_time,
        settlement_time=model.settlement_time,
        status=model.status,
        result=model.result,
        settlement_value_dollars=model.settlement_value_dollars(),
        template_id=cls.template_id,
        strike_value=cls.strike_value,
        strike_direction=cls.strike_direction,
        last_price_dollars=model.last_price_dollars(),
        volume_fp=model.volume_str(),
        open_interest_fp=model.open_interest_str(),
    )


def candles_to_rows(ticker: str, candles: list[Candlestick]) -> list[KalshiPriceRow]:
    rows: list[KalshiPriceRow] = []
    for c in candles:
        end_date = datetime.fromtimestamp(c.end_period_ts, tz=UTC).date().isoformat()
        vol = c.volume_fp or (str(c.volume) if c.volume is not None else None)
        oi = c.open_interest_fp or (
            str(c.open_interest) if c.open_interest is not None else None
        )
        rows.append(
            KalshiPriceRow(
                ticker=ticker,
                period_end_ts=c.end_period_ts,
                period_end_date=end_date,
                open_dollars=c.price.get_open(),
                high_dollars=c.price.get_high(),
                low_dollars=c.price.get_low(),
                close_dollars=c.price.get_close(),
                mean_dollars=c.price.get_mean(),
                yes_bid_close=c.yes_bid.get_close(),
                yes_ask_close=c.yes_ask.get_close(),
                volume_fp=vol,
                open_interest_fp=oi,
            )
        )
    return rows


# ── Discovery ──────────────────────────────────────────────────────────


async def _iter_markets(
    client: KalshiClient,
    *,
    series_tickers: list[str] | None,
    status: str | None,
    max_markets: int | None,
) -> AsyncIterator[KalshiMarketModel]:
    """Yield parsed market models, either per-series or across all markets."""
    base_params: dict[str, object] = {}
    if status:
        base_params["status"] = status

    targets: list[dict[str, object]]
    if series_tickers:
        targets = [{**base_params, "series_ticker": st} for st in series_tickers]
    else:
        targets = [dict(base_params)]

    seen = 0
    for params in targets:
        async for batch in client.paginate_batches("/markets", "markets", params):
            for raw in batch:
                try:
                    yield KalshiMarketModel.model_validate(raw)
                except Exception as exc:
                    logger.debug("Skipping unparseable market: %s", exc)
                    continue
                seen += 1
                if max_markets is not None and seen >= max_markets:
                    return


# ── Orchestrator ───────────────────────────────────────────────────────


async def _fetch_candles(
    client: KalshiClient, row: KalshiMarketRow, start_ts: int, end_ts: int
) -> list[Candlestick]:
    """Fetch daily candlesticks, preferring the series-scoped endpoint.

    The bare ``/historical`` path 404s for current markets, so we use the
    series-scoped endpoint whenever a series ticker is available.
    """
    if row.series_ticker:
        return await client.get_live_candlesticks(
            row.series_ticker, row.ticker, start_ts, end_ts, DAILY_INTERVAL
        )
    return await client.get_historical_candlesticks(
        row.ticker, start_ts, end_ts, DAILY_INTERVAL
    )


async def _ingest_prices(
    client: KalshiClient,
    rows: list[KalshiMarketRow],
    *,
    db_path: Path | None,
    report: KalshiIngestReport,
) -> None:
    """Fetch + store candlestick history for each market, resiliently."""
    for row in rows:
        start_ts = _to_ts(row.open_time)
        end_ts = _to_ts(row.close_time) or _to_ts(row.settlement_time)
        if start_ts is None or end_ts is None or end_ts <= start_ts:
            continue
        try:
            candles = await _fetch_candles(client, row, start_ts, end_ts)
        except Exception as exc:
            logger.warning("Candles failed for %s: %s", row.ticker, exc)
            report.errors.append(f"{row.ticker}: {exc}")
            continue
        price_rows = candles_to_rows(row.ticker, candles)
        if price_rows:
            with connect(db_path) as conn:
                report.price_rows += bulk_insert_kalshi_prices(conn, price_rows)
                conn.commit()


async def run_kalshi_ingest(
    *,
    series_tickers: list[str] | None = None,
    status: str | None = "settled",
    with_prices: bool = True,
    max_markets: int | None = None,
    db_path: Path | None = None,
    client: KalshiClient | None = None,
) -> KalshiIngestReport:
    """Ingest macro Kalshi markets (and optionally their price history)."""
    started_at = datetime.now(tz=UTC)
    report = KalshiIngestReport(started_at=started_at, finished_at=None)

    audit_id = 0
    with connect(db_path) as conn:
        audit_id = record_ingest_run(
            conn,
            IngestRun(
                source="kalshi",
                target=",".join(series_tickers) if series_tickers else "all",
                started_at=started_at.isoformat(),
                status="running",
            ),
        )
        conn.commit()

    macro_rows: list[KalshiMarketRow] = []
    owns_client = client is None
    client_ctx = KalshiClient() if client is None else client
    try:
        if owns_client:
            await client_ctx.__aenter__()

        async for model in _iter_markets(
            client_ctx,
            series_tickers=series_tickers,
            status=status,
            max_markets=max_markets,
        ):
            report.markets_seen += 1
            row = market_to_row(model)
            if row is None:
                continue
            macro_rows.append(row)
            report.by_template[row.template_id or "?"] = (
                report.by_template.get(row.template_id or "?", 0) + 1
            )

        with connect(db_path) as conn:
            bulk_insert_kalshi_markets(conn, macro_rows)
            conn.commit()
        report.markets_stored = len(macro_rows)

        if with_prices:
            await _ingest_prices(client_ctx, macro_rows, db_path=db_path, report=report)
    finally:
        if owns_client:
            await client_ctx.__aexit__(None, None, None)

    report.finished_at = datetime.now(tz=UTC)
    with connect(db_path) as conn:
        conn.execute(
            """
            UPDATE ingest_runs
               SET finished_at = ?, status = ?, rows_added = ?, error_message = ?
             WHERE run_id = ?
            """,
            (
                report.finished_at.isoformat(),
                "ok" if not report.errors else "partial",
                report.markets_stored,
                "; ".join(report.errors)[:2000],
                audit_id,
            ),
        )
        conn.commit()

    logger.info(
        "Kalshi ingest: %d seen, %d macro stored, %d price rows",
        report.markets_seen, report.markets_stored, report.price_rows,
    )
    return report


__all__ = [
    "KalshiIngestReport",
    "candles_to_rows",
    "market_to_row",
    "run_kalshi_ingest",
]
