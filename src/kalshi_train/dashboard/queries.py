"""Read-only reporting queries for the data-quality dashboard.

Every function returns a pandas DataFrame and is safe to call against an
empty database (returns an empty / zero-filled frame rather than raising),
so the dashboard renders cleanly at any stage of ingestion.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from kalshi_train.db.connection import connect


def _read(
    sql: str,
    db_path: Path | None = None,
    params: dict[str, object] | None = None,
) -> pd.DataFrame:
    with connect(db_path, read_only=True) as conn:
        return pd.read_sql_query(sql, conn, params=params or {})


def table_counts(db_path: Path | None = None) -> pd.DataFrame:
    """Row count for every real table (excludes sqlite_* and FTS shadows)."""
    with connect(db_path, read_only=True) as conn:
        names = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' AND name NOT LIKE '%_fts%' "
                "ORDER BY name"
            ).fetchall()
        ]
        rows = [
            {"table": n, "rows": conn.execute(f"SELECT COUNT(*) FROM {n}").fetchone()[0]}
            for n in names
        ]
    return pd.DataFrame(rows, columns=["table", "rows"])


def numeric_coverage(db_path: Path | None = None) -> pd.DataFrame:
    """Per-series numeric coverage: periods, vintages, date range, staleness."""
    return _read(
        """
        SELECT d.series_id,
               d.source,
               d.category,
               d.frequency,
               COUNT(DISTINCT o.observation_date)                          AS n_periods,
               COUNT(o.observation_date)                                   AS n_rows,
               MIN(o.observation_date)                                     AS first_obs,
               MAX(o.observation_date)                                     AS last_obs,
               CAST(julianday('now') - julianday(MAX(o.observation_date))
                    AS INTEGER)                                            AS days_since_last
        FROM series_definitions d
        LEFT JOIN series_observations o ON o.series_id = d.series_id
        GROUP BY d.series_id
        ORDER BY d.source, d.category, d.series_id
        """,
        db_path,
    )


def empty_series(db_path: Path | None = None) -> pd.DataFrame:
    """Series defined but with zero observations (ingest gaps to chase)."""
    return _read(
        """
        SELECT d.series_id, d.source, d.category
        FROM series_definitions d
        LEFT JOIN series_observations o ON o.series_id = d.series_id
        WHERE o.series_id IS NULL
        ORDER BY d.series_id
        """,
        db_path,
    )


def text_coverage(db_path: Path | None = None) -> pd.DataFrame:
    """Document counts and date range per text document_type."""
    return _read(
        """
        SELECT document_type,
               COUNT(*)            AS n_docs,
               MIN(published_date) AS first_pub,
               MAX(published_date) AS last_pub,
               CAST(AVG(LENGTH(body)) AS INTEGER) AS avg_chars
        FROM text_documents
        GROUP BY document_type
        ORDER BY document_type
        """,
        db_path,
    )


def text_by_year(db_path: Path | None = None) -> pd.DataFrame:
    """Document counts per (year, document_type) for a coverage heatmap."""
    return _read(
        """
        SELECT substr(published_date, 1, 4) AS year,
               document_type,
               COUNT(*)                     AS n_docs
        FROM text_documents
        GROUP BY year, document_type
        ORDER BY year
        """,
        db_path,
    )


def kalshi_coverage(db_path: Path | None = None) -> pd.DataFrame:
    """Kalshi market counts per template, with resolution + price coverage."""
    return _read(
        """
        SELECT m.template_id,
               COUNT(*)                                                   AS n_markets,
               SUM(CASE WHEN m.result IN ('yes','no') THEN 1 ELSE 0 END)  AS n_resolved,
               COUNT(DISTINCT p.ticker)                                   AS n_with_prices
        FROM kalshi_markets m
        LEFT JOIN kalshi_price_history p ON p.ticker = m.ticker
        GROUP BY m.template_id
        ORDER BY n_markets DESC
        """,
        db_path,
    )


def polymarket_coverage(db_path: Path | None = None) -> pd.DataFrame:
    """Polymarket market counts per template, with resolution coverage."""
    return _read(
        """
        SELECT template_id,
               COUNT(*)                                    AS n_markets,
               SUM(CASE WHEN resolved = 1 THEN 1 ELSE 0 END) AS n_resolved
        FROM polymarket_markets
        GROUP BY template_id
        ORDER BY n_markets DESC
        """,
        db_path,
    )


def calendar_coverage(db_path: Path | None = None) -> pd.DataFrame:
    """Event-calendar counts per event, with consensus / surprise coverage."""
    return _read(
        """
        SELECT event_name,
               COUNT(*)                                            AS n_events,
               SUM(CASE WHEN consensus_value IS NOT NULL THEN 1 ELSE 0 END) AS n_consensus,
               SUM(CASE WHEN surprise IS NOT NULL THEN 1 ELSE 0 END)        AS n_surprise,
               MIN(release_date)                                   AS first_event,
               MAX(release_date)                                   AS last_event
        FROM event_calendar
        GROUP BY event_name
        ORDER BY n_events DESC
        """,
        db_path,
    )


def ingest_log(db_path: Path | None = None, limit: int = 25) -> pd.DataFrame:
    """Most recent ingest runs (audit trail)."""
    return _read(
        """
        SELECT run_id, source, target, status, rows_added, started_at, finished_at
        FROM ingest_runs
        ORDER BY run_id DESC
        LIMIT :limit
        """,
        db_path,
        {"limit": limit},
    )


__all__ = [
    "calendar_coverage",
    "empty_series",
    "ingest_log",
    "kalshi_coverage",
    "numeric_coverage",
    "polymarket_coverage",
    "table_counts",
    "text_by_year",
    "text_coverage",
]
