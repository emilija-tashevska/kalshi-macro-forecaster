"""Phase 1.5 — Polymarket macro-market ingestion.

Paginates the Gamma ``/markets`` feed, keeps only markets whose question
matches one of our macro templates (reusing the title-keyword side of
``kalshi_macro.classify``), and stores metadata + resolution in
``polymarket_markets``. This gives us a second market-implied source and
some pre-Kalshi history for cross-referencing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from kalshi_train.data.kalshi_macro import classify
from kalshi_train.data.sources.polymarket import PolymarketClient, PolymarketMarketModel
from kalshi_train.db.connection import connect
from kalshi_train.db.ingest import (
    IngestRun,
    PolymarketMarketRow,
    bulk_insert_polymarket_markets,
    record_ingest_run,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PolymarketIngestReport:
    started_at: datetime
    finished_at: datetime | None
    markets_seen: int = 0
    markets_stored: int = 0
    by_template: dict[str, int] = field(default_factory=dict)


def market_to_row(model: PolymarketMarketModel) -> PolymarketMarketRow | None:
    """Classify by question/description; flatten macro markets to a row."""
    cls = classify(ticker="", title=model.question, subtitle=model.description)
    if cls is None:
        return None
    cond = model.resolved_condition_id()
    if not cond:
        return None
    return PolymarketMarketRow(
        condition_id=cond,
        slug=model.slug,
        question=model.question,
        description=(model.description or "")[:2000],
        template_id=cls.template_id,
        strike_value=cls.strike_value,
        strike_direction=cls.strike_direction,
        start_date=model.startDate,
        end_date=model.endDate,
        resolved=model.closed,
        outcome=model.resolved_outcome(),
    )


async def run_polymarket_ingest(
    *,
    params: dict[str, object] | None = None,
    max_markets: int | None = None,
    db_path: Path | None = None,
    client: PolymarketClient | None = None,
) -> PolymarketIngestReport:
    """Ingest macro Polymarket markets into ``polymarket_markets``."""
    started_at = datetime.now(tz=UTC)
    report = PolymarketIngestReport(started_at=started_at, finished_at=None)

    audit_id = 0
    with connect(db_path) as conn:
        audit_id = record_ingest_run(
            conn,
            IngestRun(
                source="polymarket",
                target="markets",
                started_at=started_at.isoformat(),
                status="running",
            ),
        )
        conn.commit()

    rows: list[PolymarketMarketRow] = []
    owns_client = client is None
    client_ctx = PolymarketClient() if client is None else client
    try:
        if owns_client:
            await client_ctx.__aenter__()
        async for batch in client_ctx.paginate_markets(params):
            for model in batch:
                report.markets_seen += 1
                row = market_to_row(model)
                if row is None:
                    continue
                rows.append(row)
                report.by_template[row.template_id or "?"] = (
                    report.by_template.get(row.template_id or "?", 0) + 1
                )
                if max_markets is not None and len(rows) >= max_markets:
                    break
            if max_markets is not None and len(rows) >= max_markets:
                break
    finally:
        if owns_client:
            await client_ctx.__aexit__(None, None, None)

    with connect(db_path) as conn:
        bulk_insert_polymarket_markets(conn, rows)
        conn.commit()
    report.markets_stored = len(rows)

    report.finished_at = datetime.now(tz=UTC)
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE ingest_runs SET finished_at = ?, status = ?, rows_added = ? WHERE run_id = ?",
            (report.finished_at.isoformat(), "ok", report.markets_stored, audit_id),
        )
        conn.commit()

    logger.info(
        "Polymarket ingest: %d seen, %d macro stored",
        report.markets_seen, report.markets_stored,
    )
    return report


__all__ = ["PolymarketIngestReport", "market_to_row", "run_polymarket_ingest"]
