# Phase 4 — SFT dataset (Fed-cut)

## Summary

- Target: **fed_decision** (the wired target; others are future work)
- Meetings (resolutions): **146**
- Snapshots / examples: **1022** (meetings x 7 horizons)
- Train / val / test examples: **714 / 154 / 154**
- Train base rate (cut frequency): **4.9%**
- Snapshot-level positive rate: **6.8%** (meeting-level)

## Calibrated targets

Soft targets blend the train base rate (far horizons) toward the realized outcome (near horizons); never hard 0/1.

- target prob min / mean / max: **0.020 / 0.065 / 0.848**

## Examples per horizon (days before meeting)

|    |   examples |
|---:|-----------:|
| 90 |        146 |
| 60 |        146 |
| 30 |        146 |
| 14 |        146 |
|  7 |        146 |
|  3 |        146 |
|  1 |        146 |

## Sanity checks

- OK: splits are group-disjoint (no meeting in two splits).
- OK: every snapshot's as_of precedes its meeting date.
- OK: every completion parses back to its calibrated target.

**All checks passed: True**

## Format

Chat JSONL (`data/sft/{train,val,test}.jsonl`): each line has `messages` (system/user/assistant) + `metadata`. Phase 5 trains on the `messages` with completion-only loss masking.

## Known limitations (path to ~50k examples)

- Only the **fed_decision** target is wired. Adding the other six templates (CPI, NFP, unemployment, GDP, 10Y yield, recession) with their strike variations is what scales this toward the ~50k target.
- Reasoning chains are **templated** (deterministic, leakage-free). An optional strong-LLM teacher can replace them later.
