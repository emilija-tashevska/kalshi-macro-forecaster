"""Render a Fed-cut feature row into a forecasting prompt, and parse the
probability back out of the model's reply.

The prompt deliberately gives the model the *same* point-in-time features
the XGBoost baseline sees (no peeking at the outcome or future data), so
the two are directly comparable. We ask for a single calibrated
probability and a terminal ``PROBABILITY: <0..1>`` line we can parse
robustly.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

from kalshi_train.features.fed_cut import CONTEXT_FEATURE_NAMES, FED_CUT_FEATURE_SPECS

SYSTEM_PROMPT = (
    "You are a calibrated macroeconomic forecaster specializing in U.S. "
    "monetary policy. You estimate the probability that the Federal Open "
    "Market Committee (FOMC) will CUT the federal funds target range at its "
    "next scheduled meeting, using only the economic data provided (which "
    "reflects what was knowable as of the stated date). Be well-calibrated: "
    "historically rate cuts are relatively infrequent, so avoid extreme "
    "probabilities unless the evidence is strong. Think briefly, then end "
    "your reply with a single line in exactly this format:\n"
    "PROBABILITY: <number between 0 and 1>"
)

_CONTEXT_DESCRIPTIONS = {
    "meetings_since_last_cut": "Consecutive prior meetings without a cut",
    "prior_meeting_was_cut": "Prior meeting was a cut (1=yes, 0=no)",
}


def _fmt(value: Any) -> str | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(f):
        return None
    return f"{f:.4g}"


def render_fed_cut_prompt(row: Mapping[str, Any]) -> tuple[str, str]:
    """Return ``(system, user)`` prompts for one feature row.

    ``row`` is a feature-matrix row (dict or pandas Series) containing the
    feature columns plus ``meeting_date`` / ``as_of_date``.
    """
    meeting = row.get("meeting_date")
    as_of = row.get("as_of_date")

    lines = [
        "Forecast the next FOMC rate decision.",
        "",
        f"FOMC meeting date: {meeting}",
        f"Data known as of: {as_of}",
        "",
        "Economic indicators (point-in-time):",
    ]
    for spec in FED_CUT_FEATURE_SPECS:
        rendered = _fmt(row.get(spec.name))
        if rendered is not None:
            lines.append(f"- {spec.description}: {rendered}")
    for name in CONTEXT_FEATURE_NAMES:
        rendered = _fmt(row.get(name))
        if rendered is not None:
            lines.append(f"- {_CONTEXT_DESCRIPTIONS.get(name, name)}: {rendered}")

    lines += [
        "",
        "What is the probability the FOMC cuts the target range at this "
        "meeting? Reason briefly, then end with:",
        "PROBABILITY: <number between 0 and 1>",
    ]
    return SYSTEM_PROMPT, "\n".join(lines)


_LABELLED_RE = re.compile(r"PROBABILITY\s*[:=]\s*(-?\d+(?:\.\d+)?)\s*(%?)", re.I)
_PERCENT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*%")
_DECIMAL_RE = re.compile(r"\b(0?\.\d+|1(?:\.0+)?|0|1)\b")


def parse_probability(text: str) -> float | None:
    """Extract a probability in [0, 1] from a model reply.

    Resolution order: the explicit ``PROBABILITY:`` line, then any
    percentage, then a bare decimal. Percent values are divided by 100.
    Returns ``None`` if nothing parseable is found.
    """
    if not text:
        return None

    m = _LABELLED_RE.search(text)
    if m:
        val = float(m.group(1))
        if m.group(2) == "%":
            val /= 100.0
        return _clamp(val)

    # Prefer the last percentage / decimal mentioned (usually the conclusion).
    pcts = _PERCENT_RE.findall(text)
    if pcts:
        return _clamp(float(pcts[-1]) / 100.0)

    decs = _DECIMAL_RE.findall(text)
    if decs:
        return _clamp(float(decs[-1]))
    return None


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


__all__ = ["SYSTEM_PROMPT", "parse_probability", "render_fed_cut_prompt"]
