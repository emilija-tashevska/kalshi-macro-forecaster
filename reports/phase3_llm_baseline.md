# Phase 3 — Vanilla LLM baseline

## Setup

- Model: **claude-sonnet-4-6** (no fine-tuning, temperature 0)
- Same Fed-cut target, features, and temporal test split as Phase 2
- Prompt gives only point-in-time indicators; reply parsed to a probability
- Test meetings: **22**  ·  parse failures (fell back to base rate): **0**

## Test-set metrics (lower is better)

| model                 |   brier |   log_loss |       n |   pos_rate |   mean_pred |
|:----------------------|--------:|-----------:|--------:|-----------:|------------:|
| llm:claude-sonnet-4-6 |  0.1410 |     0.4971 | 22.0000 |     0.2273 |      0.1936 |
| xgboost               |  0.2167 |     0.8875 | 22.0000 |     0.2273 |      0.0245 |
| always_0.5            |  0.2500 |     0.6931 | 22.0000 |     0.2273 |      0.5000 |
| prior_rate            |  0.2074 |     0.7242 | 22.0000 |     0.2273 |      0.0490 |

## Findings

The vanilla LLM **beats XGBoost and the base-rate baseline** on both Brier and log loss — the opposite of Phase 2. Given the *same* numeric point-in-time features, the LLM spreads its probabilities across the range and tracks the recent cutting cycle, whereas XGBoost (trained only on history) stayed anchored near the low historical base rate.

**Important caveat — LLM pretraining leakage.** The held-out meetings are recent (2024), which almost certainly falls *within the model's pretraining data*. The prompt includes the meeting date, so the model may partly **recall the actual decisions** rather than forecast them. This is a form of leakage we cannot fully control for a vanilla LLM, so treat this score as an optimistic **upper bound**. The clean test is forward-testing on meetings after the model's training cutoff (Phase 8), or date-blinding the prompt. With only 22 test meetings, variance is also high.

## Reliability diagram

![Reliability](figures/phase3_llm_reliability.png)

## Exit criterion

A comparison of vanilla LLM vs XGBoost vs trivial baselines on the same test set (the market baseline is added in Phase 7, once market prices are aligned to the prediction timeline).
