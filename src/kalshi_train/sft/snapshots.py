"""Generate point-in-time lookback snapshots for SFT examples.

For each resolved FOMC meeting we emit one snapshot per lookback horizon
(e.g. 90/60/30/14/7/3/1 days before the meeting). Each snapshot's features
are computed via the PIT layer **as of that horizon date**, so a 90-days-
out snapshot only ever sees data knowable 90 days before the decision.

This multiplies a few hundred meetings into thousands of training
examples while teaching the model how confidence should grow as the
meeting approaches.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.tseries.offsets import BDay

from kalshi_train.features.fed_cut import FED_CUT_FEATURE_SPECS
from kalshi_train.targets.fed_cut import FedCutExample, build_fed_cut_examples

DEFAULT_HORIZONS: tuple[int, ...] = (90, 60, 30, 14, 7, 3, 1)


def business_day_on_or_before(d: date) -> date:
    """Roll ``d`` back to the nearest business day on or before it."""
    rolled = BDay().rollback(pd.Timestamp(d))
    return date(rolled.year, rolled.month, rolled.day)


def _meetings_since_last_cut(meeting: date, history: list[FedCutExample]) -> float:
    streak = 0.0
    for prior in reversed(history):
        if prior.meeting_date >= meeting:
            continue
        if prior.label == 1:
            break
        streak += 1.0
    return streak


def _prior_meeting_was_cut(meeting: date, history: list[FedCutExample]) -> float | None:
    for prior in reversed(history):
        if prior.meeting_date < meeting:
            return float(prior.label)
    return None


def build_fed_cut_snapshots(
    *,
    start: str = "2000-01-01",
    end: str | None = None,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    db_path: Path | None = None,
) -> pd.DataFrame:
    """Return one row per (meeting, horizon) with PIT features + metadata.

    Columns: ``resolution_id, meeting_date, as_of_date, horizon_days,
    label`` plus every Fed-cut feature (and the two context features). The
    leakage invariant ``as_of_date < meeting_date`` holds for every row.
    """
    end_date = end or date.today().isoformat()
    examples = build_fed_cut_examples(start=start, end=end_date, db_path=db_path)
    examples = sorted(examples, key=lambda e: e.meeting_date)

    rows: list[dict[str, Any]] = []
    for ex in examples:
        context = {
            "meetings_since_last_cut": _meetings_since_last_cut(ex.meeting_date, examples),
            "prior_meeting_was_cut": _prior_meeting_was_cut(ex.meeting_date, examples),
        }
        for horizon in horizons:
            as_of = business_day_on_or_before(ex.meeting_date - timedelta(days=horizon))
            if as_of >= ex.meeting_date:  # safety; never include the decision day
                continue
            snap = replace(ex, as_of_date=as_of)
            row: dict[str, Any] = {
                "resolution_id": ex.resolution_id,
                "meeting_date": ex.meeting_date,
                "as_of_date": as_of,
                "horizon_days": horizon,
                "label": ex.label,
            }
            for spec in FED_CUT_FEATURE_SPECS:
                row[spec.name] = spec.compute(snap, db_path)
            row.update(context)
            rows.append(row)

    return pd.DataFrame(rows)


__all__ = ["DEFAULT_HORIZONS", "build_fed_cut_snapshots", "business_day_on_or_before"]
