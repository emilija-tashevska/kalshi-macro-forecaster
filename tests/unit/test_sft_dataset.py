"""Unit tests for Phase 4 SFT dataset construction."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from kalshi_train.data.fomc_calendar import fomc_meeting_dates
from kalshi_train.db.connection import connect
from kalshi_train.db.ingest import (
    Observation,
    SeriesDefinition,
    upsert_observation,
    upsert_series_definition,
)
from kalshi_train.llm.prompt import parse_probability
from kalshi_train.sft.dataset import (
    build_sft_dataset,
    calibrated_probability,
    group_temporal_split,
)
from kalshi_train.sft.snapshots import build_fed_cut_snapshots, business_day_on_or_before

# ── calibrated target ──────────────────────────────────────────────────


def test_calibrated_probability_anchors_far_horizon_to_base_rate() -> None:
    # At the longest horizon (>= span), weight is 0 → target == base rate.
    assert calibrated_probability(1, 90, 0.07) == pytest.approx(0.07, abs=1e-9)
    assert calibrated_probability(0, 120, 0.07) == pytest.approx(0.07, abs=1e-9)


def test_calibrated_probability_moves_toward_outcome_near_horizon() -> None:
    base = 0.1
    p_far = calibrated_probability(1, 60, base)
    p_near = calibrated_probability(1, 1, base)
    assert base < p_far < p_near <= 0.98  # ramps up toward a positive outcome
    n_far = calibrated_probability(0, 60, base)
    n_near = calibrated_probability(0, 1, base)
    assert 0.02 <= n_near < n_far < base  # ramps down toward a negative outcome


def test_calibrated_probability_respects_floor_ceiling() -> None:
    assert calibrated_probability(1, 0, 0.99) <= 0.98
    assert calibrated_probability(0, 0, 0.01) >= 0.02


# ── split ──────────────────────────────────────────────────────────────


def test_group_temporal_split_is_chronological_and_partitions() -> None:
    meetings = [date(2020, 1, 1), date(2021, 1, 1), date(2022, 1, 1), date(2023, 1, 1)]
    mapping = group_temporal_split(meetings, train_frac=0.5, val_frac=0.25)
    assert mapping[date(2020, 1, 1)] == "train"
    assert mapping[date(2023, 1, 1)] == "test"
    # earliest is train, latest is test (temporal order preserved)
    splits_in_order = [mapping[m] for m in meetings]
    assert splits_in_order[0] == "train"
    assert splits_in_order[-1] == "test"


def test_business_day_on_or_before_skips_weekend() -> None:
    # 2024-09-15 is a Sunday → rolls back to Friday the 13th.
    assert business_day_on_or_before(date(2024, 9, 15)) == date(2024, 9, 13)


# ── end-to-end ─────────────────────────────────────────────────────────


@pytest.fixture
def fed_db(tmp_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    dates = [
        "2020-03-15", "2020-04-28", "2020-06-10", "2020-07-29", "2020-09-16",
        "2020-11-05", "2020-12-16", "2021-01-27", "2021-03-17", "2021-04-28",
        "2021-06-16", "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
        "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27",
        "2023-09-20", "2024-09-18", "2024-11-07", "2024-12-18",
    ]
    cal = tmp_path / "fomc.txt"
    cal.write_text("\n".join(dates) + "\n")
    rates = dict.fromkeys(dates, 0.25)
    rates["2020-03-15"] = 1.75  # cut into 2020-04
    rates["2024-09-18"] = 5.00
    rates["2024-11-07"] = 4.75  # cut
    rates["2024-12-18"] = 4.50  # cut
    rates["2022-03-16"] = 0.50
    rates["2022-05-04"] = 1.00
    rates["2022-06-15"] = 1.75
    rates["2022-07-27"] = 2.50
    rates["2023-09-20"] = 5.50
    with connect(tmp_db) as conn:
        upsert_series_definition(
            conn,
            SeriesDefinition("DFEDTARU", "FRED", "Target upper", "daily", revises=False),
        )
        # Provide a daily-ish series so PIT features resolve at any horizon.
        for d, r in rates.items():
            upsert_observation(conn, Observation("DFEDTARU", d, d, f"{d}T18:00:00+00:00", r))
        conn.commit()

    monkeypatch.setattr(
        "kalshi_train.targets.fed_cut.fomc_meeting_dates",
        lambda start, end, **kw: fomc_meeting_dates(
            start, end, calendar_path=cal, prefer_db=False, db_path=tmp_db
        ),
    )
    return tmp_db


def test_snapshots_respect_leakage_and_horizons(fed_db: Path) -> None:
    horizons = (30, 7, 1)
    snaps = build_fed_cut_snapshots(
        start="2020-01-01", end="2024-12-31", horizons=horizons, db_path=fed_db
    )
    assert not snaps.empty
    # leakage invariant
    assert (snaps["as_of_date"] < snaps["meeting_date"]).all()
    # one row per (meeting, horizon)
    assert set(snaps["horizon_days"].unique()) == set(horizons)


def test_build_sft_dataset_writes_splits_and_passes_sanity(
    fed_db: Path, tmp_path: Path
) -> None:
    out = tmp_path / "sft"
    report = build_sft_dataset(
        start="2020-01-01",
        end="2024-12-31",
        horizons=(30, 7, 1),
        db_path=fed_db,
        output_dir=out,
        report_path=tmp_path / "phase4.md",
    )
    assert report.sanity_passed, report.sanity_notes
    assert report.n_examples == report.n_meetings * 3
    assert sum(report.split_counts.values()) == report.n_examples

    # Files exist and are valid chat JSONL.
    for split in ("train", "val", "test"):
        path = out / f"{split}.jsonl"
        assert path.exists()
    train_lines = (out / "train.jsonl").read_text().strip().splitlines()
    assert train_lines
    ex = json.loads(train_lines[0])
    roles = [m["role"] for m in ex["messages"]]
    assert roles == ["system", "user", "assistant"]
    # completion parses back to the calibrated target
    parsed = parse_probability(ex["messages"][-1]["content"])
    assert parsed == pytest.approx(ex["metadata"]["target_prob"], abs=0.01)


def test_build_sft_dataset_group_disjoint(fed_db: Path, tmp_path: Path) -> None:
    report = build_sft_dataset(
        start="2020-01-01",
        end="2024-12-31",
        horizons=(14, 1),
        db_path=fed_db,
        output_dir=tmp_path / "sft",
        report_path=tmp_path / "p4.md",
    )
    out = tmp_path / "sft"
    meetings_by_split = {}
    for split in ("train", "val", "test"):
        ids = set()
        for line in (out / f"{split}.jsonl").read_text().strip().splitlines():
            ids.add(json.loads(line)["metadata"]["resolution_id"])
        meetings_by_split[split] = ids
    assert not (meetings_by_split["train"] & meetings_by_split["test"])
    assert not (meetings_by_split["train"] & meetings_by_split["val"])
    assert report.sanity_passed
