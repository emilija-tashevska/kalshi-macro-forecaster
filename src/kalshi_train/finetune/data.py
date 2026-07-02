"""Load the Phase 4 SFT JSONL and format it for LoRA training / eval.

We render each chat example into a single ``text`` string with an explicit
response template. Training uses completion-only loss keyed on that
template (everything before it is masked), which is model-agnostic — it
doesn't depend on a particular tokenizer's chat template, so the same
recipe works for the tiny smoke model and the real 8B model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# The collator masks all tokens up to and including this marker, so loss is
# computed only on the forecast (assistant) text that follows it.
RESPONSE_TEMPLATE = "\n\n### Forecast:\n"


def read_split(data_dir: Path, split: str) -> list[dict[str, Any]]:
    """Read ``{data_dir}/{split}.jsonl`` into a list of example dicts."""
    path = data_dir / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"SFT split not found: {path} (run `dataset build`).")
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _roles(messages: list[dict[str, str]]) -> dict[str, str]:
    out = {"system": "", "user": "", "assistant": ""}
    for m in messages:
        if m["role"] in out:
            out[m["role"]] = m["content"]
    return out


def format_prompt(example: dict[str, Any]) -> str:
    """The input side (system + user) ending in the response template."""
    r = _roles(example["messages"])
    head = f"{r['system']}\n\n{r['user']}" if r["system"] else r["user"]
    return f"{head}{RESPONSE_TEMPLATE}"


def format_for_training(example: dict[str, Any]) -> str:
    """Full training text: prompt + assistant completion."""
    r = _roles(example["messages"])
    return f"{format_prompt(example)}{r['assistant']}"


def build_text_records(data_dir: Path, split: str) -> list[dict[str, str]]:
    """Return ``[{'text': ...}]`` records ready for a HF dataset."""
    return [{"text": format_for_training(ex)} for ex in read_split(data_dir, split)]


def load_hf_dataset(data_dir: Path, split: str) -> Any:
    """Build a ``datasets.Dataset`` of ``text`` records (lazy import)."""
    from datasets import Dataset  # noqa: PLC0415 - optional heavy dep, lazy

    return Dataset.from_list(build_text_records(data_dir, split))


__all__ = [
    "RESPONSE_TEMPLATE",
    "build_text_records",
    "format_for_training",
    "format_prompt",
    "load_hf_dataset",
    "read_split",
]
