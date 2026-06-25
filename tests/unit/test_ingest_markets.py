"""Unit tests for Phase 1.5 Kalshi + Polymarket ingestion.

Fake clients return canned API payloads so we cover the macro classifier,
metadata flattening, candlestick→price mapping, resolution parsing, and
idempotency — without a network.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from kalshi_train.data.ingest_kalshi import market_to_row, run_kalshi_ingest
from kalshi_train.data.ingest_polymarket import run_polymarket_ingest
from kalshi_train.data.kalshi_macro import (
    CPI,
    FED,
    RECESSION,
    classify,
    parse_strike,
)
from kalshi_train.data.sources.kalshi_models import Candlestick, KalshiMarketModel
from kalshi_train.data.sources.polymarket import PolymarketMarketModel
from kalshi_train.db.connection import connect

# ── classifier ─────────────────────────────────────────────────────────


def test_parse_strike_directions() -> None:
    assert parse_strike("Above 3.0%") == (3.0, "above")
    assert parse_strike("≤ 2.5%") == (2.5, "below")
    assert parse_strike("3.00% to 3.25%") == (3.0, "between")
    assert parse_strike("no number here") == (None, "")


def test_classify_by_series_prefix() -> None:
    c = classify(ticker="FED-24MAR-T5.00", event_ticker="FED-24MAR", title="Fed decision")
    assert c is not None
    assert c.template_id == FED


def test_classify_by_title_keyword() -> None:
    c = classify(ticker="WEIRDTICKER-1", title="What will CPI inflation be?", subtitle="Above 3%")
    assert c is not None
    assert c.template_id == CPI
    assert c.strike_value == 3.0
    assert c.strike_direction == "above"


def test_classify_recession_keyword() -> None:
    c = classify(ticker="X", title="US recession in 2025?")
    assert c is not None
    assert c.template_id == RECESSION


def test_classify_returns_none_for_non_macro() -> None:
    assert classify(ticker="NBA-LAL-WIN", title="Will the Lakers win tonight?") is None


def test_market_to_row_skips_non_macro() -> None:
    m = KalshiMarketModel(ticker="NBA-LAL", title="Lakers win?")
    assert market_to_row(m) is None


# ── Kalshi orchestrator ────────────────────────────────────────────────


def _candle(ts: int, close: int) -> Candlestick:
    return Candlestick.model_validate(
        {
            "end_period_ts": ts,
            "price": {
                "close": close,
                "open": close - 1,
                "high": close + 1,
                "low": close - 2,
                "mean": close,
            },
            "yes_bid": {"close": close - 1},
            "yes_ask": {"close": close + 1},
            "volume": 100,
            "open_interest": 200,
        }
    )


class _FakeKalshiClient:
    def __init__(
        self, markets: list[dict[str, Any]], candles: dict[str, list[Candlestick]]
    ) -> None:
        self._markets = markets
        self._candles = candles
        self.candle_calls: list[str] = []

    async def __aenter__(self) -> _FakeKalshiClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def paginate_batches(
        self, path: str, key: str, params: dict[str, Any] | None = None, batch_size: int = 1000
    ) -> AsyncIterator[list[dict[str, Any]]]:
        yield self._markets

    async def get_historical_candlesticks(
        self, ticker: str, start_ts: int, end_ts: int, period_interval: int = 1440
    ) -> list[Candlestick]:
        self.candle_calls.append(ticker)
        return self._candles.get(ticker, [])

    async def get_live_candlesticks(
        self,
        series_ticker: str,
        ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = 1440,
    ) -> list[Candlestick]:
        self.candle_calls.append(ticker)
        return self._candles.get(ticker, [])


def _kalshi_markets() -> list[dict[str, Any]]:
    return [
        {
            "ticker": "FED-24MAR-T5.00",
            "event_ticker": "FED-24MAR",
            "title": "Will the Fed cut rates in March 2024?",
            "yes_sub_title": "Below 5.00%",
            "open_time": "2024-01-01T00:00:00Z",
            "close_time": "2024-03-20T00:00:00Z",
            "status": "settled",
            "result": "no",
            "last_price": 23,
            "volume": 1000,
            "open_interest": 500,
        },
        {
            "ticker": "CPIYOY-24FEB-A3.0",
            "event_ticker": "CPIYOY-24FEB",
            "title": "CPI year-over-year February 2024",
            "yes_sub_title": "Above 3.0%",
            "open_time": "2024-01-15T00:00:00Z",
            "close_time": "2024-03-12T00:00:00Z",
            "status": "settled",
            "result": "yes",
            "last_price": 77,
        },
        {  # non-macro: must be skipped
            "ticker": "NBA-LAL-WIN",
            "event_ticker": "NBA-LAL",
            "title": "Will the Lakers win?",
            "status": "settled",
        },
    ]


async def test_run_kalshi_ingest_stores_macro_with_prices(tmp_db: Path) -> None:
    candles = {
        "FED-24MAR-T5.00": [_candle(1710892800, 23)],
        "CPIYOY-24FEB-A3.0": [_candle(1710201600, 77)],
    }
    client = _FakeKalshiClient(_kalshi_markets(), candles)

    report = await run_kalshi_ingest(db_path=tmp_db, client=client, status="settled")

    assert report.markets_seen == 3
    assert report.markets_stored == 2  # NBA dropped
    assert report.by_template == {FED: 1, CPI: 1}
    assert report.price_rows == 2

    with connect(tmp_db, read_only=True) as conn:
        mkts = conn.execute("SELECT COUNT(*) AS c FROM kalshi_markets").fetchone()["c"]
        prices = conn.execute("SELECT COUNT(*) AS c FROM kalshi_price_history").fetchone()["c"]
        fed = conn.execute(
            "SELECT template_id, strike_value, strike_direction FROM kalshi_markets WHERE ticker=?",
            ("FED-24MAR-T5.00",),
        ).fetchone()
    assert mkts == 2
    assert prices == 2
    assert fed["template_id"] == FED
    assert fed["strike_value"] == 5.0
    assert fed["strike_direction"] == "below"


async def test_run_kalshi_ingest_idempotent(tmp_db: Path) -> None:
    candles = {"FED-24MAR-T5.00": [_candle(1710892800, 23)], "CPIYOY-24FEB-A3.0": []}
    await run_kalshi_ingest(db_path=tmp_db, client=_FakeKalshiClient(_kalshi_markets(), candles))
    await run_kalshi_ingest(db_path=tmp_db, client=_FakeKalshiClient(_kalshi_markets(), candles))
    with connect(tmp_db, read_only=True) as conn:
        mkts = conn.execute("SELECT COUNT(*) AS c FROM kalshi_markets").fetchone()["c"]
        prices = conn.execute("SELECT COUNT(*) AS c FROM kalshi_price_history").fetchone()["c"]
    assert mkts == 2
    assert prices == 1  # one candlestick, not duplicated


async def test_run_kalshi_ingest_without_prices(tmp_db: Path) -> None:
    client = _FakeKalshiClient(_kalshi_markets(), {})
    report = await run_kalshi_ingest(db_path=tmp_db, client=client, with_prices=False)
    assert report.price_rows == 0
    assert client.candle_calls == []


# ── Polymarket orchestrator ────────────────────────────────────────────


class _FakePolyClient:
    def __init__(self, pages: list[list[PolymarketMarketModel]]) -> None:
        self._pages = pages

    async def __aenter__(self) -> _FakePolyClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def paginate_markets(
        self, params: dict[str, Any] | None = None, *, page_size: int = 100
    ) -> AsyncIterator[list[PolymarketMarketModel]]:
        for page in self._pages:
            yield page


def _poly_models() -> list[PolymarketMarketModel]:
    return [
        PolymarketMarketModel.model_validate(
            {
                "conditionId": "0xabc",
                "slug": "fed-cut-march-2024",
                "question": "Will the Fed cut interest rates in March 2024?",
                "description": "Resolves yes if the FOMC lowers the target range.",
                "closed": True,
                "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0", "1"]',
                "startDate": "2024-01-01",
                "endDate": "2024-03-20",
            }
        ),
        PolymarketMarketModel.model_validate(
            {
                "conditionId": "0xdef",
                "question": "Will Manchester United win the league?",
                "closed": False,
            }
        ),
    ]


async def test_run_polymarket_ingest_stores_macro(tmp_db: Path) -> None:
    client = _FakePolyClient([_poly_models()])
    report = await run_polymarket_ingest(db_path=tmp_db, client=client)

    assert report.markets_seen == 2
    assert report.markets_stored == 1
    assert report.by_template == {FED: 1}

    with connect(tmp_db, read_only=True) as conn:
        row = conn.execute(
            "SELECT template_id, resolved, outcome FROM polymarket_markets WHERE condition_id=?",
            ("0xabc",),
        ).fetchone()
    assert row["template_id"] == FED
    assert row["resolved"] == 1
    assert row["outcome"] == "no"  # priced ["0","1"] → "No" wins
