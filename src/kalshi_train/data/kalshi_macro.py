"""Map Kalshi markets to our seven macro question templates.

The Black Swan project filtered markets *out* (to drop sports). Here we
do the inverse: an **allowlist** that filters markets *in* and tags each
with the `question_templates.template_id` it belongs to.

Matching is two-tier:

1. **Series prefix** — the leading token of the market/event ticker
   (``FED-24JAN-...`` → ``FED``). This is the strongest, cheapest signal.
2. **Title keywords** — a regex fallback for markets whose ticker doesn't
   match a known series but whose title clearly names a macro target.

We also parse a numeric **strike** out of the market's subtitle (e.g.
"Above 3.0%", "≥ 3.25%", "3.00% to 3.25%") so downstream resolution logic
can compare against the released number.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# template_ids defined in schema.sql's question_templates seed.
FED = "fed_decision"
CPI = "cpi_yoy"
NFP = "nfp"
UNEMP = "unemployment"
GDP = "gdp"
YIELD10 = "yield_10y"
RECESSION = "recession_12m"


# Leading-token (series) → template. Kalshi has used several tickers per
# topic over time; modern series carry a ``KX`` prefix (e.g. ``KXFED``).
# We list both the current and legacy forms plus close variants.
SERIES_PREFIX_MAP: dict[str, str] = {
    # Fed rate decision
    "KXFED": FED,
    "KXFEDDECISION": FED,
    "FED": FED,
    "FEDDECISION": FED,
    "FEDRATE": FED,
    "FOMC": FED,
    # CPI / inflation
    "KXCPI": CPI,
    "KXCPIYOY": CPI,
    "KXCPICORE": CPI,
    "CPI": CPI,
    "CPIYOY": CPI,
    "ACPI": CPI,
    "CPICORE": CPI,
    "COREPCE": CPI,
    # Payrolls
    "KXPAYROLLS": NFP,
    "KXNFP": NFP,
    "PAYROLLS": NFP,
    "NFP": NFP,
    "JOBS": NFP,
    # Unemployment
    "KXU3": UNEMP,
    "KXUNRATE": UNEMP,
    "KXJOBLESS": UNEMP,
    "UNRATE": UNEMP,
    "UE": UNEMP,
    "JOBLESS": UNEMP,
    # GDP
    "KXGDP": GDP,
    "KXRGDP": GDP,
    "GDP": GDP,
    "GDPQ": GDP,
    "RGDP": GDP,
    # 10-year Treasury yield
    "KX10YY": YIELD10,
    "KXUS10Y": YIELD10,
    "US10Y": YIELD10,
    "10YY": YIELD10,
    "TNX": YIELD10,
    # Recession
    "KXRECESSION": RECESSION,
    "KXRECSS": RECESSION,
    "RECSS": RECESSION,
    "RECESSION": RECESSION,
}


# Series tickers to query by default when the caller doesn't specify any.
# These are the live (KX-prefixed) Kalshi macro series confirmed to exist;
# querying them directly is far cheaper than paginating all of Kalshi
# (which is overwhelmingly sports markets).
DEFAULT_MACRO_SERIES_TICKERS: tuple[str, ...] = (
    "KXFED",
    "KXFEDDECISION",
    "KXCPI",
    "KXCPIYOY",
    "KXU3",
    "KXPAYROLLS",
    "KXGDP",
)


# Ordered title-keyword fallbacks. First match wins.
TITLE_KEYWORDS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bfed(eral)?\b.*\b(rate|funds|hike|cut|fomc)\b", re.I), FED),
    (re.compile(r"\bfomc\b", re.I), FED),
    (re.compile(r"\bcore\s+pce\b", re.I), CPI),
    (re.compile(r"\b(cpi|consumer price|inflation)\b", re.I), CPI),
    (re.compile(r"\b(nonfarm|non-farm|payrolls?|jobs report)\b", re.I), NFP),
    (re.compile(r"\b(unemployment|jobless)\b", re.I), UNEMP),
    (re.compile(r"\b(gdp|gross domestic product)\b", re.I), GDP),
    (re.compile(r"\b(10[\-\s]?year|10y)\b.*\b(yield|treasury|note)\b", re.I), YIELD10),
    (re.compile(r"\brecession\b", re.I), RECESSION),
)

ABOVE = "above"
BELOW = "below"
BETWEEN = "between"

_ABOVE_RE = re.compile(r"(?:>=|≥|>|above|or higher|or more|at least)", re.I)
_BELOW_RE = re.compile(r"(?:<=|≤|<|below|or lower|or less|under)", re.I)
_BETWEEN_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*%?\s*(?:to|-|\u2013|and)\s*(-?\d+(?:\.\d+)?)", re.I
)
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass(frozen=True, slots=True)
class MarketClassification:
    template_id: str
    strike_value: float | None
    strike_direction: str


def parse_strike(text: str) -> tuple[float | None, str]:
    """Extract ``(strike_value, direction)`` from a market subtitle.

    Direction is one of ``above`` / ``below`` / ``between`` / ``""``.
    For ranges we keep the lower bound as the strike and tag ``between``.
    Returns ``(None, "")`` when no number is present.
    """
    if not text:
        return None, ""

    between = _BETWEEN_RE.search(text)
    if between:
        return float(between.group(1)), BETWEEN

    number = _NUMBER_RE.search(text)
    value = float(number.group()) if number else None

    if _ABOVE_RE.search(text):
        return value, ABOVE
    if _BELOW_RE.search(text):
        return value, BELOW
    return value, ""


def _prefix(ticker: str) -> str:
    return ticker.split("-", 1)[0].upper() if ticker else ""


def classify(
    *,
    ticker: str,
    event_ticker: str = "",
    series_ticker: str = "",
    title: str = "",
    subtitle: str = "",
    yes_sub_title: str = "",
) -> MarketClassification | None:
    """Return a classification if the market is one of our macro targets.

    Tries the series prefix first (of ``series_ticker``, then
    ``event_ticker``, then ``ticker``), then falls back to title keywords.
    Returns ``None`` for non-macro markets (the vast majority on Kalshi).
    """
    template: str | None = None
    for candidate in (series_ticker, event_ticker, ticker):
        template = SERIES_PREFIX_MAP.get(_prefix(candidate))
        if template:
            break

    if template is None:
        haystack = f"{title} {subtitle}"
        for pattern, tid in TITLE_KEYWORDS:
            if pattern.search(haystack):
                template = tid
                break

    if template is None:
        return None

    strike_text = yes_sub_title or subtitle or title
    strike_value, direction = parse_strike(strike_text)
    return MarketClassification(
        template_id=template,
        strike_value=strike_value,
        strike_direction=direction,
    )


__all__ = [
    "DEFAULT_MACRO_SERIES_TICKERS",
    "MarketClassification",
    "classify",
    "parse_strike",
]
