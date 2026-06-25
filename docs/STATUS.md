# Project Status

_Last updated: 2026-06-25_

A living snapshot of what is built, what is tested, and what data is
actually in the database. For the full plan and per-phase detail, see the
[README](../README.md). For the data inventory, see
[data_spec.md](./data_spec.md).

---

## TL;DR

- **Phases 0 → 1.6 are code-complete and tested** (numeric, text, markets,
  calendar). Phase 2 (XGBoost baseline) was prototyped early.
- **94 unit tests pass; ruff + mypy clean.**
- **Only the Fed text corpus is currently persisted** in
  `data/kalshi_train.db` (504 documents). Everything else is code-complete
  and live-verified but has **not been run into the main DB** yet (FRED
  needs an API key; Kalshi/Polymarket/calendar were only smoke-tested
  against throwaway DBs).
- **None of the Phase 1.4–1.6 work is committed yet** (working tree has
  uncommitted changes on top of the `Phase 2` commit).

---

## Phase completion

| Phase | Title | Code | Tests | Data in main DB |
|---|---|---|---|---|
| 0 | Scaffolding, schema, DB tooling | ✅ | ✅ | n/a |
| 1.1 | Point-in-time foundation + leakage guard | ✅ | ✅ | n/a |
| 1.2 | FRED / ALFRED numeric ingestion | ✅ | ✅ | ❌ needs API key |
| 1.3 | SPF (Survey of Professional Forecasters) | ✅ | ✅ | ❌ not run |
| 1.4 | Text corpus (Fed statements/minutes/Beige Book) | ✅ | ✅ | ✅ 504 docs |
| 1.5 | Kalshi + Polymarket markets | ✅ | ✅ | ❌ not run (temp-DB only) |
| 1.6 | Event calendar (FOMC + releases) | ✅ | ✅ | ❌ not run |
| 1.7 | Data-quality dashboard | ⬜ | ⬜ | — |
| 2 | XGBoost Fed-cut baseline | ✅ | ✅ | needs FRED data to train |
| 3+ | LLM baseline, fine-tuning, ensemble, trading | ⬜ | ⬜ | — |

Legend: ✅ done · ⬜ not started · ❌ not yet populated

---

## What is actually in `data/kalshi_train.db` right now

| Table | Rows | Notes |
|---|---:|---|
| `text_documents` | 504 | FOMC statements (222, 2000–2025), minutes (207, 2000–2025), Beige Book (75, 2017+). FTS5-searchable. |
| `question_templates` | 7 | Seeded at schema init. |
| `series_definitions` | 0 | Awaiting FRED ingest. |
| `series_observations` | 0 | Awaiting FRED ingest. |
| `kalshi_markets` | 0 | Code-complete; not run into main DB. |
| `kalshi_price_history` | 0 | Code-complete; not run into main DB. |
| `polymarket_markets` | 0 | Code-complete; not run into main DB. |
| `event_calendar` | 0 | Code-complete; not run into main DB. |
| `resolutions` | 0 | Populated in a later phase. |

> The DB file is gitignored (it does not travel with the repo).

---

## How to populate each source

```bash
# Numeric (Phase 1.2 / 1.3) — FRED needs a free key in .env (FRED_API_KEY=...)
kalshi-train ingest fred --skip-optional
kalshi-train ingest spf

# Text corpus (Phase 1.4) — keyless; ~15 min for full 2000→today backfill
kalshi-train ingest text --start 2000-01-01

# Markets (Phase 1.5) — keyless
kalshi-train ingest kalshi              # defaults to KX macro series
kalshi-train ingest polymarket --max 1000

# Event calendar (Phase 1.6) — derives from already-ingested data
kalshi-train ingest calendar

# Inspect
kalshi-train db-info
```

A full populate order: `fred → spf → text → kalshi → polymarket → calendar`
(calendar last, since it derives release events from the numeric data).

---

## Quality

- **Tests:** 94 unit tests passing; 4 integration tests auto-skip without
  network/keys. Run: `uv run pytest -m "not integration"`.
- **Lint/types:** `ruff` and `mypy --strict` both clean across 41 source
  files.
- **Toolchain note:** the local venv uses Python 3.14 via `uv`. `xgboost`
  requires the `libomp` system library (`brew install libomp`) for the
  Phase 2 test to run.

---

## Known gaps / deferred work

- **FRED data not loaded** — needs a free `FRED_API_KEY` in `.env`.
- **Markets/calendar not in the main DB** — code is done and live-verified
  (172 Kalshi macro markets, 733 GDP price rows, 30 Polymarket markets in
  smoke tests); just needs a real run into `data/kalshi_train.db`.
- **Beige Book pre-2017** — only 2017+ captured; older issues use legacy
  exact-date URLs not yet generated (~130 more documents available).
- **Text corpus breadth** — SEP projections, Fed speeches, and
  ECB/BoE/BLS/BEA narratives are not implemented (the README's ~5,000-doc
  target assumed these). The source-dispatch design makes each one a
  single added URL builder + parser.
- **Phase 1.7 dashboard** — not started.

---

## Git state

All Phase 1.4–1.6 work is **uncommitted** on top of `9fd4135 Phase 2`.

- Modified: `README.md`, `cli.py`, `data/fomc_calendar.py`,
  `data/sources/kalshi_models.py`, `db/ingest.py`
- New: `data/calendar_registry.py`, `data/ingest_calendar.py`,
  `data/ingest_kalshi.py`, `data/ingest_polymarket.py`,
  `data/ingest_text.py`, `data/kalshi_macro.py`,
  `data/sources/fed_text.py`, `data/sources/polymarket.py`, and their
  tests (`tests/unit/test_ingest_calendar.py`,
  `test_ingest_markets.py`, `test_ingest_text.py`).

Suggested next step: commit 1.4/1.5/1.6 (e.g. one commit per sub-phase),
then either add the FRED key or build Phase 1.7.
