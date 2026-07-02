"""Evaluate a fine-tuned LoRA adapter on the held-out test meetings.

Generates a probability per test example (using the *same* prompt format
and retrieved-text the model was trained on — we reuse the Phase 4
test.jsonl prompts directly), parses it, and scores it against the trivial
baselines (and XGBoost when the DB is available) on the same meetings the
Phase 2/3 baselines used. Heavy imports are lazy.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from kalshi_train.config import PROJECT_ROOT
from kalshi_train.eval.metrics import (
    MetricReport,
    baseline_always_half,
    baseline_prior_rate,
    compute_metrics,
    metrics_table,
    plot_reliability_diagram,
)
from kalshi_train.finetune.config import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR
from kalshi_train.finetune.data import format_prompt, read_split
from kalshi_train.llm.prompt import parse_probability

logger = logging.getLogger(__name__)

DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "phase5_finetune.md"
DEFAULT_PLOT_DIR = PROJECT_ROOT / "reports" / "figures"


@dataclass(slots=True)
class FinetuneEvalReport:
    adapter_dir: Path
    horizon: int | None
    n_examples: int
    n_parse_failures: int
    test_metrics: dict[str, MetricReport]
    report_path: Path
    reliability_plot: Path | None
    extra: dict[str, Any] = field(default_factory=dict)


def _select_examples(data_dir: Path, horizon: int | None) -> list[dict[str, Any]]:
    rows = read_split(data_dir, "test")
    if horizon is not None:
        rows = [r for r in rows if r["metadata"].get("horizon_days") == horizon]
    return rows


def score_predictions(
    labels: np.ndarray, probs: np.ndarray, *, train_labels: np.ndarray
) -> dict[str, MetricReport]:
    """Fine-tuned model vs trivial baselines on the same labels."""
    n = len(labels)
    return {
        "finetuned_llm": compute_metrics(labels, probs),
        "always_0.5": compute_metrics(labels, baseline_always_half(n)),
        "prior_rate": compute_metrics(labels, baseline_prior_rate(train_labels, n)),
    }


def _load_model(adapter_dir: Path, base_model: str | None) -> tuple[Any, Any]:
    import torch  # noqa: PLC0415
    from peft import PeftModel  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    if base_model is None:
        cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
        base_model = cfg.get("base_model_name_or_path")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(base_model)
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device), tokenizer


def _prompt_max_len(tokenizer: Any, max_new_tokens: int, cap: int = 4096) -> int:
    """Longest prompt we can feed, respecting the model's context window."""
    model_max = getattr(tokenizer, "model_max_length", cap)
    if not isinstance(model_max, int) or model_max <= 0 or model_max > 10**7:
        model_max = cap
    return max(16, min(cap, model_max) - max_new_tokens)


def generate_probability(
    model: Any, tokenizer: Any, prompt: str, *, max_new_tokens: int = 96
) -> float | None:
    """Greedy-generate a completion and parse a probability from it."""
    import torch  # noqa: PLC0415

    max_len = _prompt_max_len(tokenizer, max_new_tokens)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return parse_probability(text)


def run_finetune_eval(
    *,
    adapter_dir: Path = DEFAULT_OUTPUT_DIR,
    base_model: str | None = None,
    data_dir: Path = DEFAULT_DATA_DIR,
    horizon: int | None = 1,
    report_path: Path = DEFAULT_REPORT_PATH,
    plot_dir: Path = DEFAULT_PLOT_DIR,
    max_new_tokens: int = 96,
    write_report: bool = True,
) -> FinetuneEvalReport:
    """Generate + score the fine-tuned adapter on the test split."""
    examples = _select_examples(data_dir, horizon)
    if not examples:
        raise RuntimeError(f"No test examples (horizon={horizon}). Run `dataset build`.")
    train_labels = np.array(
        [r["metadata"]["label"] for r in read_split(data_dir, "train")], dtype=float
    )
    labels = np.array([r["metadata"]["label"] for r in examples], dtype=float)
    prior = float(train_labels.mean()) if len(train_labels) else 0.5

    model, tokenizer = _load_model(adapter_dir, base_model)
    probs: list[float] = []
    failures = 0
    for ex in examples:
        p = generate_probability(model, tokenizer, format_prompt(ex), max_new_tokens=max_new_tokens)
        if p is None:
            failures += 1
            p = prior
        probs.append(p)
    probs_arr = np.asarray(probs, dtype=float)

    test_metrics = score_predictions(labels, probs_arr, train_labels=train_labels)

    plot_path = plot_dir / "phase5_finetune_reliability.png"
    reliability_plot = plot_reliability_diagram(
        labels, probs_arr,
        title="Fine-tuned LLM — test set reliability",
        output_path=plot_path,
    )

    report = FinetuneEvalReport(
        adapter_dir=adapter_dir,
        horizon=horizon,
        n_examples=len(examples),
        n_parse_failures=failures,
        test_metrics=test_metrics,
        report_path=report_path,
        reliability_plot=reliability_plot,
    )
    if write_report:
        _write_report(report, metrics_table(list(test_metrics.items())))
    return report


def _write_report(report: FinetuneEvalReport, table: Any) -> None:
    report.report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 5 — Fine-tuned LoRA evaluation",
        "",
        f"- Adapter: `{report.adapter_dir}`",
        f"- Horizon: {report.horizon} day(s) before meeting  ·  "
        f"test examples: **{report.n_examples}**  ·  parse failures: "
        f"**{report.n_parse_failures}**",
        "",
        "## Test-set metrics (lower is better)",
        "",
        table.to_markdown(floatfmt=".4f"),
        "",
        "> Compare against `reports/phase2_xgboost.md` and "
        "`reports/phase3_llm_baseline.md` (same held-out meetings). Exit "
        "criterion: beat the vanilla LLM and XGBoost on Brier and log loss.",
        "",
    ]
    if report.reliability_plot is not None:
        rel = Path(
            os.path.relpath(
                report.reliability_plot.resolve(), report.report_path.resolve().parent
            )
        )
        lines += ["## Reliability diagram", "", f"![Reliability]({rel.as_posix()})", ""]
    report.report_path.write_text("\n".join(lines))


__all__ = [
    "FinetuneEvalReport",
    "generate_probability",
    "run_finetune_eval",
    "score_predictions",
]
