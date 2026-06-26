"""Unit tests for the Phase 3 vanilla-LLM baseline.

A fake LLM client returns canned replies so we cover prompt rendering,
probability parsing, response caching, and the end-to-end orchestrator
(metrics + report) without any network or API key.
"""

from __future__ import annotations

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
from kalshi_train.llm.client import CachedLLM
from kalshi_train.llm.prompt import parse_probability, render_fed_cut_prompt
from kalshi_train.targets.fed_cut import build_fed_cut_examples
from kalshi_train.training.phase3_llm import run_phase3_llm

# ── probability parsing ────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Reasoning here.\nPROBABILITY: 0.23", 0.23),
        ("PROBABILITY: 25%", 0.25),
        ("I think about 0.6", 0.6),
        ("Roughly a 40% chance.", 0.40),
        ("PROBABILITY = 1.0", 1.0),
        ("PROBABILITY: 1.5", 1.0),  # clamped
        ("PROBABILITY: -0.2", 0.0),  # clamped
    ],
)
def test_parse_probability(text: str, expected: float) -> None:
    assert parse_probability(text) == pytest.approx(expected)


def test_parse_probability_none_when_absent() -> None:
    assert parse_probability("no numbers here at all") is None
    assert parse_probability("") is None


# ── prompt rendering ───────────────────────────────────────────────────


def test_render_prompt_includes_features_and_format() -> None:
    row = {
        "meeting_date": date(2024, 9, 18),
        "as_of_date": date(2024, 9, 17),
        "ff_target_upper": 5.5,
        "unrate": 4.2,
        "cpi_yoy": 0.025,
        "vix": None,  # should be omitted
    }
    system, user = render_fed_cut_prompt(row)
    assert "PROBABILITY:" in system
    assert "2024-09-18" in user
    assert "Fed funds target upper bound: 5.5" in user
    assert "U-3 unemployment rate: 4.2" in user
    assert "VIX close" not in user  # None feature omitted


# ── cache ──────────────────────────────────────────────────────────────


class _CountingClient:
    model = "fake-model"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        return self.reply


def test_cached_llm_avoids_repeat_calls(tmp_path: Path) -> None:
    inner = _CountingClient("PROBABILITY: 0.3")
    cached = CachedLLM(inner, cache_dir=tmp_path / "llm")
    a = cached.complete("sys", "user-1")
    b = cached.complete("sys", "user-1")  # same prompt → served from disk
    c = cached.complete("sys", "user-2")  # different prompt → new call
    assert a == b == "PROBABILITY: 0.3"
    assert c == "PROBABILITY: 0.3"
    assert inner.calls == 2  # not 3


# ── orchestrator ───────────────────────────────────────────────────────


class _FakeLLM:
    """Returns a fixed probability for every prompt."""

    model = "fake-llm"

    def __init__(self, prob: float) -> None:
        self._prob = prob
        self.seen: list[str] = []

    def complete(self, system: str, user: str) -> str:
        self.seen.append(user)
        return f"My estimate.\nPROBABILITY: {self._prob}"


@pytest.fixture
def fed_db(tmp_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """DB with DFEDTARU + enough FOMC dates to form a multi-class test split."""
    dates = [
        "2020-03-15", "2020-04-28", "2020-06-10", "2020-07-29", "2020-09-16",
        "2020-11-05", "2020-12-16", "2021-01-27", "2021-03-17", "2021-04-28",
        "2021-06-16", "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
        "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27",
        "2023-09-20", "2023-11-01", "2024-09-18", "2024-11-07", "2024-12-18",
    ]
    cal = tmp_path / "fomc.txt"
    cal.write_text("\n".join(dates) + "\n")
    # Rates: cuts in 2020-03 and the 2024 meetings → both classes in the
    # (recent) test split.
    rates = {
        "2020-03-15": 1.75, "2020-04-28": 0.25, "2020-06-10": 0.25, "2020-07-29": 0.25,
        "2020-09-16": 0.25, "2020-11-05": 0.25, "2020-12-16": 0.25, "2021-01-27": 0.25,
        "2021-03-17": 0.25, "2021-04-28": 0.25, "2021-06-16": 0.25, "2021-07-28": 0.25,
        "2021-09-22": 0.25, "2021-11-03": 0.25, "2021-12-15": 0.25, "2022-01-26": 0.25,
        "2022-03-16": 0.50, "2022-05-04": 1.00, "2022-06-15": 1.75, "2022-07-27": 2.50,
        "2023-09-20": 5.50, "2023-11-01": 5.50, "2024-09-18": 5.00, "2024-11-07": 4.75,
        "2024-12-18": 4.50,
    }
    with connect(tmp_db) as conn:
        upsert_series_definition(
            conn,
            SeriesDefinition("DFEDTARU", "FRED", "Target upper", "daily", revises=False),
        )
        for d, r in rates.items():
            upsert_observation(
                conn, Observation("DFEDTARU", d, d, f"{d}T18:00:00+00:00", r)
            )
        conn.commit()

    monkeypatch.setattr(
        "kalshi_train.targets.fed_cut.fomc_meeting_dates",
        lambda start, end, **kw: fomc_meeting_dates(
            start, end, calendar_path=cal, prefer_db=False, db_path=tmp_db
        ),
    )
    return tmp_db


def test_run_phase3_llm_end_to_end(fed_db: Path, tmp_path: Path) -> None:
    # Sanity: the test split must contain both classes for metrics to be defined.
    examples = build_fed_cut_examples(start="2020-01-01", end="2024-12-31", db_path=fed_db)
    assert sum(e.label for e in examples) >= 2

    report_path = tmp_path / "phase3.md"
    result = run_phase3_llm(
        client=_FakeLLM(0.2),
        start="2020-01-01",
        end="2024-12-31",
        db_path=fed_db,
        report_path=report_path,
        plot_dir=tmp_path / "figs",
    )
    assert result.model == "fake-llm"
    assert result.n_test > 0
    assert result.n_parse_failures == 0
    assert f"llm:{result.model}" in result.test_metrics
    assert "xgboost" in result.test_metrics
    assert report_path.exists()
    assert "Vanilla LLM baseline" in report_path.read_text()


def test_run_phase3_llm_counts_parse_failures(fed_db: Path, tmp_path: Path) -> None:
    class _Garbage:
        model = "garbage"

        def complete(self, system: str, user: str) -> str:
            return "I have no idea, sorry."

    result = run_phase3_llm(
        client=_Garbage(),
        start="2020-01-01",
        end="2024-12-31",
        db_path=fed_db,
        report_path=tmp_path / "p3.md",
        plot_dir=tmp_path / "figs",
    )
    # Every reply unparseable → all fell back to the base rate.
    assert result.n_parse_failures == result.n_test
