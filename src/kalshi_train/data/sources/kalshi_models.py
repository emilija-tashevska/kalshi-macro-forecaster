"""Pydantic models for the subset of Kalshi API responses we care about.

Adapted from the author's earlier ``black-swan-event-intelligence``
project. Trimmed to the candlestick + market fields that matter for
historical training data; black-swan-specific models removed.

Kalshi prices come in two forms:
  - integer "cents" (0-100), e.g. 23
  - dollar string with 4 decimals, e.g. "0.2300"

The ``_to_dollar_str`` helper normalizes both into the canonical
4-decimal string we store in SQLite.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


def _to_dollar_str(v: str | int | float | None) -> str | None:
    """Normalize cent ints or dollar strings into 4-decimal dollar strings."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return f"{v / 100:.4f}"
    return str(v)


class CandlestickPrice(BaseModel):
    """Price section of a Kalshi candlestick. Lots of optional fields
    because the API has changed shape over time; we tolerate either form.
    """

    model_config = ConfigDict(extra="ignore")

    open: str | int | float | None = None
    high: str | int | float | None = None
    low: str | int | float | None = None
    close: str | int | float | None = None
    mean: str | int | float | None = None
    previous: str | int | float | None = None

    open_dollars: str | None = None
    high_dollars: str | None = None
    low_dollars: str | None = None
    close_dollars: str | None = None
    mean_dollars: str | None = None
    previous_dollars: str | None = None

    def get_close(self) -> str | None:
        return self.close_dollars or _to_dollar_str(self.close)

    def get_mean(self) -> str | None:
        return self.mean_dollars or _to_dollar_str(self.mean)

    def get_open(self) -> str | None:
        return self.open_dollars or _to_dollar_str(self.open)

    def get_high(self) -> str | None:
        return self.high_dollars or _to_dollar_str(self.high)

    def get_low(self) -> str | None:
        return self.low_dollars or _to_dollar_str(self.low)


class CandlestickBidAsk(BaseModel):
    """Bid or ask section of a candlestick."""

    model_config = ConfigDict(extra="ignore")

    open: str | int | float | None = None
    high: str | int | float | None = None
    low: str | int | float | None = None
    close: str | int | float | None = None

    open_dollars: str | None = None
    high_dollars: str | None = None
    low_dollars: str | None = None
    close_dollars: str | None = None

    def get_close(self) -> str | None:
        return self.close_dollars or _to_dollar_str(self.close)


class Candlestick(BaseModel):
    """One period's candlestick for a Kalshi market."""

    model_config = ConfigDict(extra="ignore")

    end_period_ts: int
    price: CandlestickPrice
    yes_bid: CandlestickBidAsk
    yes_ask: CandlestickBidAsk
    volume: str | int | float | None = None
    volume_fp: str | None = None
    open_interest: str | int | float | None = None
    open_interest_fp: str | None = None


class KalshiMarketModel(BaseModel):
    """A Kalshi market object as returned by ``/markets`` (subset).

    ``extra="ignore"`` keeps us tolerant of the many fields we don't
    persist and of the API's gradual shape changes.
    """

    model_config = ConfigDict(extra="ignore")

    ticker: str
    event_ticker: str = ""
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
    expiration_time: str | None = None
    settlement_time: str | None = None
    status: str = ""
    result: str = ""
    last_price: str | int | float | None = None
    settlement_value: str | int | float | None = None
    volume: str | int | float | None = None
    open_interest: str | int | float | None = None

    def derived_series_ticker(self) -> str:
        """Kalshi market objects don't always carry ``series_ticker``; it's
        the token before the first dash of the event ticker (``FED-24JAN`` →
        ``FED``)."""
        if self.series_ticker:
            return self.series_ticker
        base = self.event_ticker or self.ticker
        return base.split("-", 1)[0]

    def last_price_dollars(self) -> str:
        return _to_dollar_str(self.last_price) or "0.0000"

    def settlement_value_dollars(self) -> str | None:
        return _to_dollar_str(self.settlement_value)

    def volume_str(self) -> str:
        return "0.00" if self.volume is None else f"{float(self.volume):.2f}"

    def open_interest_str(self) -> str:
        return "0.00" if self.open_interest is None else f"{float(self.open_interest):.2f}"
