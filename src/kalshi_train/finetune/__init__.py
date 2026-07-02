"""Phase 5 — LoRA supervised fine-tuning.

``config`` holds the run hyperparameters; ``data`` reads the Phase 4 SFT
JSONL into a single-``text`` format with a completion-only response
template; ``train`` runs LoRA SFT (transformers + peft + trl, lazily
imported); ``evaluate`` loads the adapter, generates probabilities on the
held-out test meetings, and compares to the XGBoost / vanilla-LLM /
trivial baselines on the same split.

Heavy ML deps are imported lazily so the data/format logic stays testable
without torch, and so non-training commands don't pay the import cost.
"""
