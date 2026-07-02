"""LoRA supervised fine-tuning (transformers + peft).

We tokenize each example and mask everything up to the response template so
loss is computed only on the forecast (completion-only training). Using the
plain ``transformers.Trainer`` + a manual label mask keeps us robust to the
churn in higher-level SFT wrappers, and works identically for the tiny CPU
smoke model and the real 8B GPU model.

All heavy imports are inside functions so this module imports without torch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kalshi_train.finetune.config import FinetuneConfig
from kalshi_train.finetune.data import (
    RESPONSE_TEMPLATE,
    format_for_training,
    format_prompt,
    read_split,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FinetuneResult:
    adapter_dir: Path
    base_model: str
    n_train: int
    n_eval: int
    train_loss: float | None
    steps: int


def _pick_device(requested: str | None) -> str:
    import torch  # noqa: PLC0415

    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _tokenize_example(
    tokenizer: Any, example: dict[str, Any], max_len: int
) -> dict[str, list[int]]:
    """Tokenize one example, masking prompt tokens with -100 (completion-only)."""
    prompt = format_prompt(example)
    full = format_for_training(example)
    eos = tokenizer.eos_token or ""
    full_ids = tokenizer(full + eos, truncation=True, max_length=max_len)["input_ids"]
    prompt_ids = tokenizer(prompt, truncation=True, max_length=max_len)["input_ids"]
    n_prompt = min(len(prompt_ids), len(full_ids))
    labels = [-100] * n_prompt + full_ids[n_prompt:]
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def _build_dataset(tokenizer: Any, examples: list[dict[str, Any]], max_len: int) -> Any:
    from datasets import Dataset  # noqa: PLC0415

    rows = [_tokenize_example(tokenizer, ex, max_len) for ex in examples]
    return Dataset.from_list(rows)


def run_finetune(config: FinetuneConfig) -> FinetuneResult:
    """Fit a LoRA adapter on the SFT train split and save it to ``output_dir``."""
    import torch  # noqa: PLC0415
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: PLC0415
    from transformers import (  # noqa: PLC0415
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    device = _pick_device(config.device)
    logger.info("Fine-tuning %s on %s", config.base_model, device)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {}
    if config.load_in_4bit and device == "cuda":
        from transformers import BitsAndBytesConfig  # noqa: PLC0415

        model_kwargs["quantization_config"] = BitsAndBytesConfig(  # type: ignore[no-untyped-call]
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"

    model: Any = AutoModelForCausalLM.from_pretrained(config.base_model, **model_kwargs)
    if config.load_in_4bit and device == "cuda":
        model = prepare_model_for_kbit_training(model)  # type: ignore[no-untyped-call]
    elif device != "cuda":
        model = model.to(device)

    lora = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_ex = read_split(config.data_dir, "train")
    eval_ex = read_split(config.data_dir, "val")
    train_ds = _build_dataset(tokenizer, train_ex, config.max_seq_len)
    eval_ds = _build_dataset(tokenizer, eval_ex, config.max_seq_len)

    use_bf16 = config.bf16 and device == "cuda"
    args = TrainingArguments(
        output_dir=str(config.output_dir / "checkpoints"),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.grad_accum,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        eval_strategy="epoch",
        save_strategy="no",
        bf16=use_bf16,
        seed=config.seed,
        report_to=["wandb"] if config.use_wandb else [],
        use_cpu=(device == "cpu"),
    )
    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )
    train_out = trainer.train()

    config.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(config.output_dir))
    tokenizer.save_pretrained(str(config.output_dir))

    return FinetuneResult(
        adapter_dir=config.output_dir,
        base_model=config.base_model,
        n_train=len(train_ex),
        n_eval=len(eval_ex),
        train_loss=float(train_out.training_loss) if train_out else None,
        steps=int(train_out.global_step) if train_out else 0,
    )


__all__ = ["FinetuneResult", "run_finetune"]


# Re-exported for callers/tests that build masks directly.
_RESPONSE_TEMPLATE = RESPONSE_TEMPLATE
