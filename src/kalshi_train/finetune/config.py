"""Fine-tuning run configuration.

Defaults target a real GPU run (Llama-3.1-8B-Instruct, 4-bit + LoRA). For
the local CPU smoke test we override ``base_model`` with a tiny model and
drop 4-bit. Everything here is plain data so it's importable without torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from kalshi_train.config import PROJECT_ROOT

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "sft"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "fed_cut_lora"


@dataclass(slots=True)
class FinetuneConfig:
    # Model / IO
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    data_dir: Path = DEFAULT_DATA_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    )

    # Optimization
    learning_rate: float = 2e-4
    num_epochs: float = 3.0
    per_device_batch_size: int = 1
    grad_accum: int = 8
    max_seq_len: int = 4096
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    seed: int = 42

    # Runtime
    load_in_4bit: bool = True
    bf16: bool = True
    device: str | None = None  # None -> auto (cuda/mps/cpu)
    use_wandb: bool = False
    logging_steps: int = 10

    # Metadata carried into the report
    notes: str = ""
    extra: dict[str, str] = field(default_factory=dict)


def smoke_config(output_dir: Path, data_dir: Path | None = None) -> FinetuneConfig:
    """A tiny, CPU-friendly config to prove the pipeline runs end-to-end."""
    return FinetuneConfig(
        base_model="sshleifer/tiny-gpt2",
        data_dir=data_dir or DEFAULT_DATA_DIR,
        output_dir=output_dir,
        target_modules=("c_attn",),  # gpt2 attention proj
        num_epochs=1.0,
        per_device_batch_size=2,
        grad_accum=1,
        max_seq_len=512,
        load_in_4bit=False,
        bf16=False,
        device="cpu",
        notes="CPU smoke test (tiny-gpt2).",
    )


__all__ = ["DEFAULT_DATA_DIR", "DEFAULT_OUTPUT_DIR", "FinetuneConfig", "smoke_config"]
