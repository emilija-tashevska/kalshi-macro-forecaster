"""Phase 1.5 — Polymarket macro-market ingestion.

Pulls Gamma ``/markets`` and keeps those matching our macro templates
(via ``kalshi_macro.classify``), storing metadata + resolution in
``polymarket_markets``.

Discovery is **tag-based**: Polymarket has no series tickers, but it tags
markets by topic (Economy, GDP, recession, Macro Inflation, US Jobs, …).
Filtering by those tag IDs — rather than scanning the whole feed — is the
only reliable way to reach *closed/resolved* macro markets, since the
default feed is dominated by sports/crypto and the API caps pagination
depth. Tag IDs were discovered via the Gamma ``/tags`` endpoint.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

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

# Gamma tag IDs for macro topics (from the /tags endpoint). Filtering by
# these surfaces resolved macro markets the flat feed buries.
DEFAULT_MACRO_TAG_IDS: Final[tuple[int, ...]] = (
    100328,  # Economy (Fed rate-cut markets live here)
    101800,  # Economic Policy
    102000,  # Macro Indicators
    101249,  # Macro Inflation (CPI)
    370,     # GDP
    100201,  # recession
    1625,    # US jobs (payrolls)
)


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


def _targets(
    tag_ids: list[int] | None, closed: bool | None, params: dict[str, object] | None
) -> list[dict[str, object]]:
    """Build the list of Gamma query param sets to paginate (one per tag)."""
    base = dict(params or {})
    if closed is not None:
        base["closed"] = "true" if closed else "false"
    if tag_ids:
        return [{**base, "tag_id": t} for t in tag_ids]
    return [base]


async def run_polymarket_ingest(
    *,
    tag_ids: list[int] | None = None,
    closed: bool | None = True,
    params: dict[str, object] | None = None,
    max_markets: int | None = None,
    db_path: Path | None = None,
    client: PolymarketClient | None = None,
) -> PolymarketIngestReport:
    """Ingest macro Polymarket markets into ``polymarket_markets``.

    By default pulls **closed/resolved** markets across the macro tag set.
    Pass ``closed=False`` for live markets or ``closed=None`` for both, and
    ``tag_ids=[]`` to scan the flat feed instead of by tag.
    """
    started_at = datetime.now(tz=UTC)
    report = PolymarketIngestReport(started_at=started_at, finished_at=None)
    if tag_ids is None:
        tag_ids = list(DEFAULT_MACRO_TAG_IDS)

    audit_id = 0
    with connect(db_path) as conn:
        audit_id = record_ingest_run(
            conn,
            IngestRun(
                source="polymarket",
                target=f"tags={tag_ids or 'feed'} closed={closed}",
                started_at=started_at.isoformat(),
                status="running",
            ),
        )
        conn.commit()

    # Dedup across tags by condition_id (a market can carry several tags).
    by_cond: dict[str, PolymarketMarketRow] = {}
    owns_client = client is None
    client_ctx = PolymarketClient() if client is None else client
    try:
        if owns_client:
            await client_ctx.__aenter__()
        for target in _targets(tag_ids, closed, params):
            async for batch in client_ctx.paginate_markets(target):
                for model in batch:
                    report.markets_seen += 1
                    row = market_to_row(model)
                    if row is None or row.condition_id in by_cond:
                        continue
                    by_cond[row.condition_id] = row
                    report.by_template[row.template_id or "?"] = (
                        report.by_template.get(row.template_id or "?", 0) + 1
                    )
                if max_markets is not None and len(by_cond) >= max_markets:
                    break
            if max_markets is not None and len(by_cond) >= max_markets:
                break
    finally:
        if owns_client:
            await client_ctx.__aexit__(None, None, None)

    rows = list(by_cond.values())
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


__all__ = [
    "DEFAULT_MACRO_TAG_IDS",
    "PolymarketIngestReport",
    "market_to_row",
    "run_polymarket_ingest",
]
