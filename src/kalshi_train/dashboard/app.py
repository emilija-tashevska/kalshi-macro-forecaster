"""Phase 1.7 — read-only data-quality dashboard (Streamlit).

Run it with::

    make dashboard
    # or: uv run streamlit run src/kalshi_train/dashboard/app.py

It surfaces, at a glance, what's in the database and where the gaps are:
table sizes, per-series numeric coverage + staleness, text-corpus coverage
by year, market coverage, calendar consensus/surprise availability, the
ingest audit log, and an interactive point-in-time spot-check.
"""

from __future__ import annotations

from datetime import date

import streamlit as st

from kalshi_train.config import settings
from kalshi_train.dashboard import queries as q
from kalshi_train.db.point_in_time import VintagePolicy, pit_value

st.set_page_config(page_title="Kalshi Train — Data Quality", layout="wide")


def _table_or_info(frame: object, info: str) -> None:
    """Render a DataFrame, or an info message when it's empty."""
    if getattr(frame, "empty", True):
        st.info(info)
    else:
        st.dataframe(frame, width="stretch", hide_index=True)


def _render_overview() -> None:
    counts = q.table_counts()
    total = int(counts["rows"].sum()) if not counts.empty else 0
    st.metric("Total rows across all tables", f"{total:,}")
    st.header("Tables")
    st.dataframe(counts, width="stretch", hide_index=True)


def _render_numeric() -> None:
    st.header("Numeric series coverage")
    numeric = q.numeric_coverage()
    _table_or_info(numeric, "No numeric series yet. Run `ingest fred` / `ingest spf`.")
    empties = q.empty_series()
    if not empties.empty:
        st.warning(f"{len(empties)} defined series have **zero** observations:")
        st.dataframe(empties, width="stretch", hide_index=True)


def _render_text() -> None:
    st.header("Text corpus coverage")
    text = q.text_coverage()
    _table_or_info(text, "No documents yet. Run `kalshi-train ingest text`.")
    by_year = q.text_by_year()
    if not by_year.empty:
        pivot = by_year.pivot(index="year", columns="document_type", values="n_docs").fillna(0)
        st.bar_chart(pivot)


def _render_markets() -> None:
    st.header("Markets coverage")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Kalshi")
        _table_or_info(q.kalshi_coverage(), "No Kalshi markets. Run `ingest kalshi`.")
    with col2:
        st.subheader("Polymarket")
        _table_or_info(q.polymarket_coverage(), "No Polymarket markets. Run `ingest polymarket`.")


def _render_pit() -> None:
    st.header("Point-in-time spot check")
    st.caption("Resolve a series' value as it was knowable on a historical date.")
    numeric = q.numeric_coverage()
    numeric_ids = numeric["series_id"].tolist() if not numeric.empty else []
    if not numeric_ids:
        st.info("Ingest numeric series to enable the spot check.")
        return
    c1, c2, c3 = st.columns([2, 2, 1])
    series_id = c1.selectbox("Series", numeric_ids)
    as_of = c2.date_input("As-of date", value=date(2024, 11, 7))
    policy = c3.selectbox("Policy", [p.value for p in VintagePolicy])
    if st.button("Resolve"):
        val = pit_value(series_id, as_of, policy=VintagePolicy(policy))
        if val is None:
            st.warning(f"No value of {series_id} was knowable on {as_of}.")
        else:
            st.success(f"{series_id} as of {as_of} = {val}")


def main() -> None:
    st.title("Kalshi Train — Data Quality Dashboard")
    st.caption(f"Database: `{settings.kalshi_train_db_path}`  ·  read-only")
    _render_overview()
    _render_numeric()
    _render_text()
    _render_markets()
    st.header("Event calendar")
    _table_or_info(q.calendar_coverage(), "No calendar events. Run `ingest calendar`.")
    _render_pit()
    st.header("Ingest log")
    _table_or_info(q.ingest_log(), "No ingest runs recorded yet.")


if __name__ == "__main__":
    main()
