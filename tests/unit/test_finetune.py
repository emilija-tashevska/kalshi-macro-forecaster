"""Unit tests for the Phase 5 fine-tuning data/format/eval plumbing.

These cover the torch-free logic: JSONL loading, prompt/target formatting
with the completion-only response template, and the scoring function. The
actual LoRA training + generation are covered by the CPU smoke test
(marked slow) and the real GPU run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kalshi_train.finetune.config import FinetuneConfig, smoke_config
from kalshi_train.finetune.data import (
    RESPONSE_TEMPLATE,
    build_text_records,
    format_for_training,
    format_prompt,
    read_split,
)
from kalshi_train.finetune.evaluate import score_predictions


def _write_split(data_dir: Path, split: str, rows: list[dict]) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / f"{split}.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _example(prob: float = 0.3, label: int = 0, horizon: int = 1) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "You are a forecaster."},
            {"role": "user", "content": "Indicators: unemployment 4.0"},
            {"role": "assistant", "content": f"Reasoning.\nPROBABILITY: {prob}"},
        ],
        "metadata": {"label": label, "horizon_days": horizon, "target_prob": prob},
    }


# ── formatting ─────────────────────────────────────────────────────────


def test_format_prompt_ends_with_response_template() -> None:
    p = format_prompt(_example())
    assert p.endswith(RESPONSE_TEMPLATE)
    assert "You are a forecaster." in p
    assert "unemployment 4.0" in p
    assert "PROBABILITY" not in p  # answer excluded from the prompt side


def test_format_for_training_is_prompt_plus_completion() -> None:
    ex = _example(prob=0.42)
    full = format_for_training(ex)
    prompt = format_prompt(ex)
    assert full.startswith(prompt)
    assert full[len(prompt) :] == "Reasoning.\nPROBABILITY: 0.42"


def test_read_split_and_build_text_records(tmp_path: Path) -> None:
    _write_split(tmp_path, "train", [_example(0.1), _example(0.9)])
    rows = read_split(tmp_path, "train")
    assert len(rows) == 2
    recs = build_text_records(tmp_path, "train")
    assert all(set(r) == {"text"} for r in recs)
    assert RESPONSE_TEMPLATE in recs[0]["text"]


# ── config ─────────────────────────────────────────────────────────────


def test_smoke_config_is_cpu_and_tiny() -> None:
    cfg = smoke_config(Path("/tmp/x"))
    assert cfg.device == "cpu"
    assert cfg.load_in_4bit is False
    assert "tiny" in cfg.base_model


def test_default_config_targets_8b_gpu() -> None:
    cfg = FinetuneConfig()
    assert cfg.load_in_4bit is True
    assert "q_proj" in cfg.target_modules


# ── scoring ────────────────────────────────────────────────────────────


def test_score_predictions_includes_model_and_baselines() -> None:
    labels = np.array([0, 0, 1, 0])
    probs = np.array([0.1, 0.2, 0.8, 0.3])
    train_labels = np.array([0, 0, 0, 1, 0])
    m = score_predictions(labels, probs, train_labels=train_labels)
    assert set(m) == {"finetuned_llm", "always_0.5", "prior_rate"}
    # A good model should beat coin-flip on Brier here.
    assert m["finetuned_llm"].brier < m["always_0.5"].brier
