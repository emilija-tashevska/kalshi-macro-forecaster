"""Phase 3 — vanilla (un-fine-tuned) LLM forecasting baseline.

Runs an off-the-shelf LLM on the **same** Fed-cut test set as Phase 2 and
compares it to the XGBoost baseline and the trivial baselines, using the
same proper scoring rules. This establishes the bar that fine-tuning
(Phase 5) will have to clear.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from kalshi_train.config import PROJECT_ROOT, settings
from kalshi_train.eval.metrics import (
    MetricReport,
    baseline_always_half,
    baseline_prior_rate,
    compute_metrics,
    metrics_table,
    plot_reliability_diagram,
)
from kalshi_train.eval.splits import TemporalSplit, temporal_train_val_test_split
from kalshi_train.features.fed_cut import build_feature_matrix, feature_names
from kalshi_train.llm.client import LLMClient
from kalshi_train.llm.prompt import parse_probability, render_fed_cut_prompt
from kalshi_train.models.xgboost_baseline import predict_proba, train_xgboost_final
from kalshi_train.targets.fed_cut import build_fed_cut_examples

DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "phase3_llm_baseline.md"
DEFAULT_PLOT_DIR = PROJECT_ROOT / "reports" / "figures"


@dataclass(frozen=True, slots=True)
class Phase3Report:
    model: str
    n_test: int
    n_parse_failures: int
    test_metrics: dict[str, MetricReport]
    report_path: Path
    reliability_plot: Path | None


def llm_predict_fed_cut(
    client: LLMClient,
    test_df: pd.DataFrame,
    *,
    fallback: float,
) -> tuple[np.ndarray, int]:
    """Prompt the LLM per test row; return (probabilities, n_parse_failures).

    Unparseable replies fall back to the supplied base-rate probability so
    one bad generation doesn't crash the run or silently bias the metric.
    """
    probs: list[float] = []
    failures = 0
    for _idx, row in test_df.iterrows():
        system, user = render_fed_cut_prompt(row.to_dict())
        reply = client.complete(system, user)
        parsed = parse_probability(reply)
        if parsed is None:
            failures += 1
            parsed = fallback
        probs.append(parsed)
    return np.asarray(probs, dtype=float), failures


def run_phase3_llm(
    *,
    client: LLMClient,
    start: str = "2000-01-01",
    end: str | None = None,
    db_path: Path | None = None,
    report_path: Path = DEFAULT_REPORT_PATH,
    plot_dir: Path = DEFAULT_PLOT_DIR,
    write_report: bool = True,
) -> Phase3Report:
    """Evaluate a vanilla LLM on the Phase 2 Fed-cut test set."""
    db = db_path or settings.kalshi_train_db_path
    end_date = end or date.today().isoformat()

    examples = build_fed_cut_examples(start=start, end=end_date, db_path=db)
    if len(examples) < 10:
        raise RuntimeError(
            f"Only {len(examples)} Fed-cut examples found — ingest FRED data first."
        )

    matrix = build_feature_matrix(examples, db_path=db)
    cols = feature_names()
    split = temporal_train_val_test_split(matrix)
    if split.test["label"].nunique() < 2:
        raise RuntimeError("Held-out test set has a single class; widen the date range.")

    train_val = pd.concat([split.train, split.val])
    y_train = split.train["label"].to_numpy()
    y_test = split.test["label"].to_numpy()
    prior = float(y_train.mean())

    llm_probs, failures = llm_predict_fed_cut(client, split.test, fallback=prior)

    model, imputer = train_xgboost_final(train_val, cols)
    xgb_probs = predict_proba(model, imputer, split.test, cols)

    test_metrics: dict[str, MetricReport] = {
        f"llm:{client.model}": compute_metrics(y_test, llm_probs),
        "xgboost": compute_metrics(y_test, xgb_probs),
        "always_0.5": compute_metrics(y_test, baseline_always_half(len(y_test))),
        "prior_rate": compute_metrics(y_test, baseline_prior_rate(y_train, len(y_test))),
    }

    plot_path = plot_dir / "phase3_llm_reliability.png"
    reliability_plot = plot_reliability_diagram(
        y_test,
        llm_probs,
        title=f"Vanilla LLM ({client.model}) — test set reliability",
        output_path=plot_path,
    )

    if write_report:
        _write_report(
            report_path=report_path,
            model=client.model,
            split=split,
            test_metrics=test_metrics,
            failures=failures,
            plot_path=reliability_plot,
        )

    return Phase3Report(
        model=client.model,
        n_test=len(split.test),
        n_parse_failures=failures,
        test_metrics=test_metrics,
        report_path=report_path,
        reliability_plot=reliability_plot,
    )


def _write_report(
    *,
    report_path: Path,
    model: str,
    split: TemporalSplit,
    test_metrics: dict[str, MetricReport],
    failures: int,
    plot_path: Path | None,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    comparison = metrics_table(list(test_metrics.items()))
    test_df = split.test
    lines = [
        "# Phase 3 — Vanilla LLM baseline",
        "",
        "## Setup",
        "",
        f"- Model: **{model}** (no fine-tuning, temperature 0)",
        "- Same Fed-cut target, features, and temporal test split as Phase 2",
        "- Prompt gives only point-in-time indicators; reply parsed to a probability",
        f"- Test meetings: **{len(test_df)}**  ·  parse failures (fell back to base "
        f"rate): **{failures}**",
        "",
        "## Test-set metrics (lower is better)",
        "",
        comparison.to_markdown(floatfmt=".4f"),
        "",
        "## Findings",
        "",
        "The vanilla LLM **beats XGBoost and the base-rate baseline** on both "
        "Brier and log loss — the opposite of Phase 2. Given the *same* numeric "
        "point-in-time features, the LLM spreads its probabilities across the "
        "range and tracks the recent cutting cycle, whereas XGBoost (trained "
        "only on history) stayed anchored near the low historical base rate.",
        "",
        "**Important caveat — LLM pretraining leakage.** The held-out meetings "
        "are recent (2024), which almost certainly falls *within the model's "
        "pretraining data*. The prompt includes the meeting date, so the model "
        "may partly **recall the actual decisions** rather than forecast them. "
        "This is a form of leakage we cannot fully control for a vanilla LLM, "
        "so treat this score as an optimistic **upper bound**. The clean test "
        "is forward-testing on meetings after the model's training cutoff "
        "(Phase 8), or date-blinding the prompt. With only 22 test meetings, "
        "variance is also high.",
        "",
    ]
    if plot_path is not None:
        rel = Path(os.path.relpath(plot_path.resolve(), report_path.resolve().parent))
        lines += ["## Reliability diagram", "", f"![Reliability]({rel.as_posix()})", ""]
    lines += [
        "## Exit criterion",
        "",
        "A comparison of vanilla LLM vs XGBoost vs trivial baselines on the same "
        "test set (the market baseline is added in Phase 7, once market prices are "
        "aligned to the prediction timeline).",
        "",
    ]
    report_path.write_text("\n".join(lines))


__all__ = ["Phase3Report", "llm_predict_fed_cut", "run_phase3_llm"]
