"""Assemble SFT chat examples from snapshots and write the JSONL splits.

Each example is a (system, user, assistant) chat triple. The assistant
"completion" pairs a short, leakage-free reasoning chain with a
**horizon-aware calibrated probability** rather than a hard 0/1 — training
on raw outcomes teaches over-confidence, which is exactly what our proper
scoring rules punish.

Splitting is **group-aware and temporal**: all horizons of one meeting go
to the same split, and earlier meetings train / later meetings test, so no
snapshot of a test meeting can leak into training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from kalshi_train.config import PROJECT_ROOT
from kalshi_train.llm.prompt import parse_probability, render_fed_cut_prompt
from kalshi_train.sft.snapshots import DEFAULT_HORIZONS, build_fed_cut_snapshots

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "sft"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "phase4_sft_dataset.md"

CONFIDENCE_MAX = 0.85
CONFIDENCE_SPAN_DAYS = 90.0
PROB_FLOOR = 0.02
PROB_CEIL = 0.98


@dataclass(slots=True)
class SFTReport:
    output_dir: Path
    report_path: Path
    base_rate: float
    n_examples: int
    n_meetings: int
    split_counts: dict[str, int]
    horizon_counts: dict[int, int]
    sanity_passed: bool
    sanity_notes: list[str] = field(default_factory=list)


# ── Calibrated target ──────────────────────────────────────────────────


def calibrated_probability(
    outcome: int,
    horizon_days: int,
    base_rate: float,
    *,
    conf_max: float = CONFIDENCE_MAX,
    span_days: float = CONFIDENCE_SPAN_DAYS,
) -> float:
    """Blend the historical base rate (far out) toward the outcome (close in).

    At the longest horizon the target sits at the base rate (we couldn't
    have known); as the meeting approaches, confidence ramps toward the
    realized outcome but never reaches 0/1. This yields soft, calibrated
    targets instead of over-confident hard labels.
    """
    weight = conf_max * max(0.0, 1.0 - horizon_days / span_days)
    prob = base_rate + (float(outcome) - base_rate) * weight
    return round(min(PROB_CEIL, max(PROB_FLOOR, prob)), 4)


def _fmt(value: Any) -> str | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if pd.isna(f):
        return None
    return f"{f:.4g}"


def build_reasoning(row: dict[str, Any], target_prob: float, base_rate: float) -> str:
    """A short, leakage-free rationale ending in a parseable PROBABILITY line.

    The rationale cites observed indicators and the historical base rate but
    never references the realized outcome.
    """
    bits: list[str] = []
    for name, label in (
        ("ff_target_upper", "policy rate (upper)"),
        ("dgs2", "2Y yield"),
        ("t10y3m", "10Y-3M spread"),
        ("unrate", "unemployment"),
        ("core_cpi_yoy", "core CPI YoY"),
    ):
        rendered = _fmt(row.get(name))
        if rendered is not None:
            bits.append(f"{label} {rendered}")
    signals = "; ".join(bits) if bits else "limited indicators available"
    horizon = int(row["horizon_days"])
    return (
        f"As of {row['as_of_date']}, {horizon} days before the meeting, the key "
        f"point-in-time signals are: {signals}. Historically the FOMC cuts at "
        f"about {base_rate:.0%} of meetings, and uncertainty {horizon} days out "
        f"is elevated relative to the days just before the decision. Weighing the "
        f"level and trend of these indicators against that base rate:\n"
        f"PROBABILITY: {target_prob:.2f}"
    )


def snapshot_to_example(
    row: dict[str, Any], *, base_rate: float, split: str
) -> dict[str, Any]:
    """Turn one snapshot row into a chat example with metadata."""
    target = calibrated_probability(int(row["label"]), int(row["horizon_days"]), base_rate)
    system, user = render_fed_cut_prompt(row)
    assistant = build_reasoning(row, target, base_rate)
    meeting = row["meeting_date"]
    as_of = row["as_of_date"]
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": {
            "template_id": "fed_decision",
            "resolution_id": row["resolution_id"],
            "meeting_date": meeting.isoformat() if isinstance(meeting, date) else str(meeting),
            "as_of_date": as_of.isoformat() if isinstance(as_of, date) else str(as_of),
            "horizon_days": int(row["horizon_days"]),
            "label": int(row["label"]),
            "target_prob": target,
            "split": split,
        },
    }


# ── Group-aware temporal split ─────────────────────────────────────────


def group_temporal_split(
    meeting_dates: list[date], *, train_frac: float = 0.70, val_frac: float = 0.15
) -> dict[date, str]:
    """Assign whole meetings to train/val/test in chronological order."""
    ordered = sorted(set(meeting_dates))
    n = len(ordered)
    train_end = max(1, int(n * train_frac))
    val_end = max(train_end + 1, int(n * (train_frac + val_frac)))
    val_end = min(val_end, n)
    mapping: dict[date, str] = {}
    for i, m in enumerate(ordered):
        if i < train_end:
            mapping[m] = "train"
        elif i < val_end:
            mapping[m] = "val"
        else:
            mapping[m] = "test"
    return mapping


# ── Orchestrator ───────────────────────────────────────────────────────


def build_sft_dataset(
    *,
    start: str = "2000-01-01",
    end: str | None = None,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    db_path: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    report_path: Path = DEFAULT_REPORT_PATH,
    write_report: bool = True,
) -> SFTReport:
    """Build the SFT dataset end-to-end and write the JSONL splits."""
    snaps = build_fed_cut_snapshots(start=start, end=end, horizons=horizons, db_path=db_path)
    if snaps.empty:
        raise RuntimeError("No snapshots produced — ingest FRED data first.")

    meetings = [pd.Timestamp(m).date() for m in snaps["meeting_date"]]
    split_map = group_temporal_split(meetings)

    # Base rate from TRAIN meetings only (one label per meeting), so the soft
    # target's anchor never peeks at val/test outcomes.
    per_meeting = snaps.drop_duplicates("resolution_id")
    train_labels = [
        int(r.label)
        for r in per_meeting.itertuples()
        if split_map[pd.Timestamp(r.meeting_date).date()] == "train"
    ]
    base_rate = sum(train_labels) / len(train_labels) if train_labels else 0.5

    examples_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for row in snaps.to_dict(orient="records"):
        split = split_map[pd.Timestamp(row["meeting_date"]).date()]
        examples_by_split[split].append(
            snapshot_to_example(row, base_rate=base_rate, split=split)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, examples in examples_by_split.items():
        path = output_dir / f"{split}.jsonl"
        with path.open("w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

    sanity_passed, notes = _run_sanity_checks(examples_by_split)

    split_counts = {k: len(v) for k, v in examples_by_split.items()}
    horizon_counts = {int(h): int((snaps["horizon_days"] == h).sum()) for h in horizons}
    report = SFTReport(
        output_dir=output_dir,
        report_path=report_path,
        base_rate=base_rate,
        n_examples=len(snaps),
        n_meetings=int(per_meeting.shape[0]),
        split_counts=split_counts,
        horizon_counts=horizon_counts,
        sanity_passed=sanity_passed,
        sanity_notes=notes,
    )

    if write_report:
        _write_report(report, snaps, split_map)
    return report


def _run_sanity_checks(
    examples_by_split: dict[str, list[dict[str, Any]]],
) -> tuple[bool, list[str]]:
    notes: list[str] = []
    ok = True

    # 1. No meeting appears in more than one split (group integrity).
    meetings_per_split = {
        s: {e["metadata"]["resolution_id"] for e in exs}
        for s, exs in examples_by_split.items()
    }
    train_v_test = meetings_per_split["train"] & meetings_per_split["test"]
    train_v_val = meetings_per_split["train"] & meetings_per_split["val"]
    val_v_test = meetings_per_split["val"] & meetings_per_split["test"]
    if train_v_test or train_v_val or val_v_test:
        ok = False
        notes.append("FAIL: a meeting leaks across splits.")
    else:
        notes.append("OK: splits are group-disjoint (no meeting in two splits).")

    # 2. Leakage: as_of strictly before meeting for every example.
    leaks = [
        e
        for exs in examples_by_split.values()
        for e in exs
        if e["metadata"]["as_of_date"] >= e["metadata"]["meeting_date"]
    ]
    if leaks:
        ok = False
        notes.append(f"FAIL: {len(leaks)} examples have as_of >= meeting_date.")
    else:
        notes.append("OK: every snapshot's as_of precedes its meeting date.")

    # 3. The assistant reasoning's parsed probability matches the target.
    mismatches = 0
    for exs in examples_by_split.values():
        for e in exs:
            assistant = e["messages"][-1]["content"]
            parsed = parse_probability(assistant)
            if parsed is None or abs(parsed - e["metadata"]["target_prob"]) > 0.01:
                mismatches += 1
    if mismatches:
        ok = False
        notes.append(f"FAIL: {mismatches} completions don't parse to their target prob.")
    else:
        notes.append("OK: every completion parses back to its calibrated target.")

    return ok, notes


def _write_report(
    report: SFTReport, snaps: pd.DataFrame, split_map: dict[date, str]
) -> None:
    report.report_path.parent.mkdir(parents=True, exist_ok=True)
    meeting_labels = snaps.drop_duplicates("resolution_id")["label"]
    targets = [
        calibrated_probability(int(r.label), int(r.horizon_days), report.base_rate)
        for r in snaps.itertuples()
    ]
    tser = pd.Series(targets)

    lines = [
        "# Phase 4 — SFT dataset (Fed-cut)",
        "",
        "## Summary",
        "",
        "- Target: **fed_decision** (the wired target; others are future work)",
        f"- Meetings (resolutions): **{report.n_meetings}**",
        f"- Snapshots / examples: **{report.n_examples}** "
        f"(meetings x {len(report.horizon_counts)} horizons)",
        f"- Train / val / test examples: "
        f"**{report.split_counts['train']} / {report.split_counts['val']} / "
        f"{report.split_counts['test']}**",
        f"- Train base rate (cut frequency): **{report.base_rate:.1%}**",
        f"- Snapshot-level positive rate: **{meeting_labels.mean():.1%}** (meeting-level)",
        "",
        "## Calibrated targets",
        "",
        "Soft targets blend the train base rate (far horizons) toward the "
        "realized outcome (near horizons); never hard 0/1.",
        "",
        f"- target prob min / mean / max: **{tser.min():.3f} / {tser.mean():.3f} / "
        f"{tser.max():.3f}**",
        "",
        "## Examples per horizon (days before meeting)",
        "",
        pd.Series(report.horizon_counts).to_frame("examples").to_markdown(),
        "",
        "## Sanity checks",
        "",
        *[f"- {note}" for note in report.sanity_notes],
        "",
        f"**All checks passed: {report.sanity_passed}**",
        "",
        "## Format",
        "",
        "Chat JSONL (`data/sft/{train,val,test}.jsonl`): each line has "
        "`messages` (system/user/assistant) + `metadata`. Phase 5 trains on "
        "the `messages` with completion-only loss masking.",
        "",
        "## Known limitations (path to ~50k examples)",
        "",
        "- Only the **fed_decision** target is wired. Adding the other six "
        "templates (CPI, NFP, unemployment, GDP, 10Y yield, recession) with "
        "their strike variations is what scales this toward the ~50k target.",
        "- Reasoning chains are **templated** (deterministic, leakage-free). "
        "An optional strong-LLM teacher can replace them later.",
        "",
    ]
    report.report_path.write_text("\n".join(lines))
    _ = split_map


__all__ = [
    "SFTReport",
    "build_reasoning",
    "build_sft_dataset",
    "calibrated_probability",
    "group_temporal_split",
    "snapshot_to_example",
]
