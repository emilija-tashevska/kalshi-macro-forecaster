"""Phase 2 end-to-end pipeline: Fed-cut XGBoost baseline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

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
from kalshi_train.models.xgboost_baseline import (
    predict_proba,
    train_xgboost_final,
    train_xgboost_temporal_cv,
)
from kalshi_train.targets.fed_cut import FedCutExample, build_fed_cut_examples

DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "phase2_xgboost.md"
DEFAULT_PLOT_DIR = PROJECT_ROOT / "reports" / "figures"


@dataclass(frozen=True, slots=True)
class Phase2Report:
    n_examples: int
    n_features: int
    train_size: int
    val_size: int
    test_size: int
    test_metrics: dict[str, MetricReport]
    cv_mean_metrics: MetricReport
    feature_importance: pd.Series
    report_path: Path
    reliability_plot: Path | None


def run_phase2_fed_cut(
    *,
    start: str = "2000-01-01",
    end: str | None = None,
    db_path: Path | None = None,
    report_path: Path = DEFAULT_REPORT_PATH,
    plot_dir: Path = DEFAULT_PLOT_DIR,
    write_report: bool = True,
) -> Phase2Report:
    """Build dataset, train XGBoost, compare to baselines, write report."""
    db = db_path or settings.kalshi_train_db_path
    end_date = end or date.today().isoformat()

    examples = build_fed_cut_examples(start=start, end=end_date, db_path=db)
    if len(examples) < 10:
        raise RuntimeError(
            f"Only {len(examples)} Fed-cut examples found — ingest FRED data first "
            f"(kalshi-train ingest fred --skip-optional) or widen the date range."
        )

    matrix = build_feature_matrix(examples, db_path=db)
    cols = feature_names()
    split = temporal_train_val_test_split(matrix)
    if split.test["label"].nunique() < 2:
        raise RuntimeError(
            "Held-out test set contains a single class. Widen the date range or "
            "wait for more FOMC meetings in the sample."
        )

    train_val = pd.concat([split.train, split.val])
    y_train = split.train["label"].to_numpy()
    y_test = split.test["label"].to_numpy()

    cv_result = train_xgboost_temporal_cv(train_val, cols, n_splits=5)
    model, imputer = train_xgboost_final(train_val, cols)
    test_probs = predict_proba(model, imputer, split.test, cols)

    test_metrics: dict[str, MetricReport] = {
        "xgboost": compute_metrics(y_test, test_probs),
        "always_0.5": compute_metrics(y_test, baseline_always_half(len(y_test))),
        "prior_rate": compute_metrics(
            y_test, baseline_prior_rate(y_train, len(y_test))
        ),
    }

    plot_path = plot_dir / "phase2_fed_cut_reliability.png"
    reliability_plot = plot_reliability_diagram(
        y_test,
        test_probs,
        title="Fed-cut XGBoost — test set reliability",
        output_path=plot_path,
    )

    if write_report:
        _write_markdown_report(
            report_path=report_path,
            examples=examples,
            matrix=matrix,
            split=split,
            test_metrics=test_metrics,
            cv_mean=cv_result.mean_metrics,
            importance=cv_result.feature_importance,
            plot_path=reliability_plot,
        )

    return Phase2Report(
        n_examples=len(examples),
        n_features=len(cols),
        train_size=len(split.train),
        val_size=len(split.val),
        test_size=len(split.test),
        test_metrics=test_metrics,
        cv_mean_metrics=cv_result.mean_metrics,
        feature_importance=cv_result.feature_importance,
        report_path=report_path,
        reliability_plot=reliability_plot,
    )


def _write_markdown_report(
    *,
    report_path: Path,
    examples: list[FedCutExample],
    matrix: pd.DataFrame,
    split: TemporalSplit,
    test_metrics: dict[str, MetricReport],
    cv_mean: MetricReport,
    importance: pd.Series,
    plot_path: Path | None,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    pos_rate = matrix["label"].mean()
    comparison = metrics_table(list(test_metrics.items()))

    lines = [
        "# Phase 2 — Fed-cut XGBoost baseline",
        "",
        "## Target",
        "",
        'Binary: **Will the Fed cut rates at the next FOMC meeting?**',
        "",
        "- Label = 1 when `DFEDTARU` drops vs the prior meeting, else 0",
        "- Features computed via the point-in-time interface only",
        "- Prediction `as_of_date` = business day before the announcement",
        "",
        "## Dataset",
        "",
        f"- Examples: **{len(examples)}** meetings",
        f"- Positive (cut) rate: **{pos_rate:.1%}**",
        f"- Train / val / test: **{len(split.train)} / {len(split.val)} / {len(split.test)}**",
        f"- Features: **{len(feature_names())}**",
        "",
        "## Test-set metrics (lower is better)",
        "",
        comparison.to_markdown(floatfmt=".4f"),
        "",
        "## Temporal CV (train+val, out-of-fold)",
        "",
        f"- Mean OOF Brier: **{cv_mean.brier:.4f}**",
        f"- Mean OOF log loss: **{cv_mean.log_loss:.4f}**",
        "",
        "## Top feature importances (CV average)",
        "",
        importance.head(15).to_frame("importance").to_markdown(floatfmt=".4f"),
        "",
    ]
    if plot_path is not None:
        # Robust to relative vs absolute path mixes (CLI passes a relative
        # report path while the plot dir is absolute).
        rel = Path(os.path.relpath(plot_path.resolve(), report_path.resolve().parent))
        lines.extend(
            [
                "## Reliability diagram",
                "",
                f"![Reliability]({rel.as_posix()})",
                "",
            ]
        )
    lines.extend(
        [
            "## Exit criterion",
            "",
            "XGBoost should beat `always_0.5` and `prior_rate` on held-out Brier and log loss.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines))
