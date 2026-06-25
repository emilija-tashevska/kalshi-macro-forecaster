"""Registry of economic releases that become ``event_calendar`` rows.

Phase 1.6 builds the economic-release calendar *from data we already
ingested* rather than scraping a third-party calendar. For every series
listed here, each period's **first print** (the value as first reported,
i.e. the row with the earliest ``vintage_date`` for that
``observation_date``) becomes one calendar event:

    actual_value     = the first-reported number ("the print")
    release_date     = when that first print became public
    observation_date = the period it describes

This makes the Phase 1.6 checkpoint — "every release in the database has
a corresponding calendar entry" — true *by construction*.

Consensus + surprise
--------------------
A "surprise" (actual - consensus) is only meaningful when the consensus
forecast is in the **same units and frequency** as the actual. The one
clean internal match we have today is **real GDP**: the SPF nowcast
(``SPF_RGDP_MEDIAN_NOWCAST``) forecasts the *level* of real GDP for the
survey quarter, which is exactly what the BEA advance estimate
(``GDPC1``) reports for that same quarter. For monthly prints (CPI, NFP,
unemployment) we have no same-frequency internal consensus yet, so we
store ``actual`` only and leave consensus/surprise NULL until a true
release-consensus feed (DBnomics / Trading Economics) is wired in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class ReleaseEvent:
    """One series whose releases we turn into calendar events.

    ``consensus_series_id`` points at another series already in
    ``series_observations`` (typically an SPF derived series) to use as
    the pre-release consensus. It is only set when actual and consensus
    share units and frequency, so that ``surprise = actual - consensus``
    is well-defined. ``None`` means "actual only, no surprise".
    """

    series_id: str
    event_name: str
    template_id: str | None = None
    consensus_series_id: str | None = None
    notes: str = ""


CALENDAR_RELEASES: Final[tuple[ReleaseEvent, ...]] = (
    # ── Inflation ──────────────────────────────────────────────────────
    ReleaseEvent("CPIAUCSL", "CPI (headline)", template_id="cpi_yoy"),
    ReleaseEvent("CPILFESL", "Core CPI", template_id="cpi_yoy"),
    ReleaseEvent("PCEPI", "PCE Price Index"),
    ReleaseEvent("PCEPILFE", "Core PCE"),
    ReleaseEvent("PPIACO", "Producer Price Index"),
    # ── Labor market ───────────────────────────────────────────────────
    ReleaseEvent("PAYEMS", "Nonfarm Payrolls", template_id="nfp"),
    ReleaseEvent("UNRATE", "Unemployment Rate", template_id="unemployment"),
    ReleaseEvent("JTSJOL", "JOLTS Job Openings"),
    ReleaseEvent("ICSA", "Initial Jobless Claims"),
    # ── Growth & activity ──────────────────────────────────────────────
    ReleaseEvent(
        "GDPC1",
        "Real GDP",
        template_id="gdp",
        consensus_series_id="SPF_RGDP_MEDIAN_NOWCAST",
        notes="Consensus = SPF same-quarter real GDP level nowcast.",
    ),
    ReleaseEvent("INDPRO", "Industrial Production"),
    ReleaseEvent("RSXFS", "Retail Sales"),
)


# Template + metadata for FOMC decision events (no underlying release
# series — the "actual" is the post-meeting target rate).
FOMC_EVENT_NAME: Final[str] = "FOMC Rate Decision"
FOMC_TEMPLATE_ID: Final[str] = "fed_decision"
FOMC_RATE_SERIES: Final[str] = "DFEDTARU"


def all_release_series_ids() -> list[str]:
    """Every series_id that produces calendar events. Handy for tests."""
    return [r.series_id for r in CALENDAR_RELEASES]


def find_release(series_id: str) -> ReleaseEvent | None:
    for r in CALENDAR_RELEASES:
        if r.series_id == series_id:
            return r
    return None
