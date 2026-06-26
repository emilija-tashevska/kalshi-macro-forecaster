"""Phase 4 — supervised fine-tuning (SFT) dataset construction.

``snapshots`` turns each resolved event into multiple point-in-time
lookback snapshots; ``dataset`` renders each snapshot into a
(system, user, assistant) chat example with a horizon-aware *calibrated*
target probability, splits group-aware + temporally, and writes
``data/sft/{train,val,test}.jsonl`` plus a stats/sanity report.

Currently wired for the Fed-cut target (the one with a target + feature
builder). The other six question templates plug in by adding their own
target/feature modules; the snapshot → example → split machinery is
target-agnostic.
"""
