# Phase 2 — Fed-cut XGBoost baseline

## Target

Binary: **Will the Fed cut rates at the next FOMC meeting?**

- Label = 1 when `DFEDTARU` drops vs the prior meeting, else 0
- Features computed via the point-in-time interface only
- Prediction `as_of_date` = business day before the announcement

## Dataset

- Examples: **146** meetings
- Positive (cut) rate: **6.8%**
- Train / val / test: **102 / 22 / 22**
- Features: **30**

## Test-set metrics (lower is better)

| model      |   brier |   log_loss |       n |   pos_rate |   mean_pred |
|:-----------|--------:|-----------:|--------:|-----------:|------------:|
| xgboost    |  0.2167 |     0.8875 | 22.0000 |     0.2273 |      0.0245 |
| always_0.5 |  0.2500 |     0.6931 | 22.0000 |     0.2273 |      0.5000 |
| prior_rate |  0.2074 |     0.7242 | 22.0000 |     0.2273 |      0.0490 |

## Temporal CV (train+val, out-of-fold)

- Mean OOF Brier: **0.0738**
- Mean OOF log loss: **0.7669**

## Top feature importances (CV average)

|                         |   importance |
|:------------------------|-------------:|
| dgs10                   |       0.0336 |
| t10y3m                  |       0.0319 |
| icsa                    |       0.0207 |
| unrate                  |       0.0190 |
| t10y2y                  |       0.0187 |
| core_cpi_yoy            |       0.0171 |
| t10yie                  |       0.0159 |
| meetings_since_last_cut |       0.0136 |
| dgs2                    |       0.0090 |
| walcl_chg_63bd          |       0.0087 |
| baa10ym                 |       0.0031 |
| dgs10_chg_63bd          |       0.0030 |
| payems_chg_21bd         |       0.0028 |
| t5yie                   |       0.0025 |
| vix                     |       0.0006 |

## Reliability diagram

![Reliability](figures/phase2_fed_cut_reliability.png)

## Exit criterion

XGBoost should beat `always_0.5` and `prior_rate` on held-out Brier and log loss.
