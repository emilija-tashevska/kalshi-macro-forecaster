"""Phase 3 — vanilla LLM forecasting baseline.

``client`` wraps OpenAI / Anthropic behind a tiny ``LLMClient`` protocol
(plus an on-disk response cache); ``prompt`` renders a feature row into a
forecasting prompt and parses the probability back out. The orchestrator
lives in ``kalshi_train.training.phase3_llm`` and reuses the exact Phase 2
dataset + temporal split so the comparison is apples-to-apples.
"""
