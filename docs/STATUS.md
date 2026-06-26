# Project Status

_Last updated: 2026-06-26_

A living snapshot of what is built, what is tested, and what data is
actually in the database. For the full plan and per-phase detail, see the
[README](../README.md). For the data inventory, see
[data_spec.md](./data_spec.md).

---

## TL;DR

- **Phase 1 is COMPLETE** (sub-phases 1.1 → 1.7), code-complete, tested,
  AND populated in `data/kalshi_train.db`: numeric (FRED + SPF), text
  corpus, markets (Kalshi + Polymarket), event calendar, and a
  data-quality dashboard. Phase 2 (XGBoost baseline) was prototyped early
  and can now be trained on real data.
- **101 unit tests pass; ruff + mypy clean.**
- The DB now holds **~369k numeric vintage rows, 550 text documents, 172
  Kalshi markets + 20k price rows, ~1.6k Polymarket markets (1.5k
  resolved), and 4.9k calendar events** (see table below).
- **Next:** train Phase 2 for real (first baseline metrics), then Phase 3
  (LLM baseline).

---

## Phase completion

| Phase | Title | Code | Tests | Data in main DB |
|---|---|---|---|---|
| 0 | Scaffolding, schema, DB tooling | ✅ | ✅ | n/a |
| 1.1 | Point-in-time foundation + leakage guard | ✅ | ✅ | n/a |
| 1.2 | FRED / ALFRED numeric ingestion | ✅ | ✅ | ✅ 99 series, ~369k rows |
| 1.3 | SPF (Survey of Professional Forecasters) | ✅ | ✅ | ✅ 38 series, 5.6k rows |
| 1.4 | Text corpus (Fed statements/minutes/Beige Book) | ✅ | ✅ | ✅ 504 docs |
| 1.5 | Kalshi + Polymarket markets | ✅ | ✅ | ✅ 172 + 62 markets |
| 1.6 | Event calendar (FOMC + releases) | ✅ | ✅ | ✅ 4.9k events |
| 1.7 | Data-quality dashboard (Streamlit) | ✅ | ✅ | n/a (read-only view) |
| 2 | XGBoost Fed-cut baseline | ✅ | ✅ | needs FRED data to train |
| 3+ | LLM baseline, fine-tuning, ensemble, trading | ⬜ | ⬜ | — |

Legend: ✅ done · ⬜ not started · ❌ not yet populated

---

## What is actually in `data/kalshi_train.db` right now

_(populated 2026-06-26)_

| Table | Rows | Notes |
|---|---:|---|
| `series_definitions` | 99 | FRED (61 ok) + SPF (38) derived series. |
| `series_observations` | 368,565 | FRED/ALFRED vintages + SPF. |
| `text_documents` | 550 | FOMC statements (222, 2000–2025), minutes (207), Beige Book (121, 2011+). FTS5-searchable. |
| `kalshi_markets` | 172 | Macro markets across 5 templates. |
| `kalshi_price_history` | 20,037 | Daily candlesticks. |
| `polymarket_markets` | 1,610 | Macro markets across all 7 templates; 1,548 resolved (back to Oct 2023), 62 still open. Tag-based pull (Economy/GDP/CPI/jobs/recession). |
| `polymarket_price_history` | 0 | Deferred — see Phase 4 gaps. |
| `event_calendar` | 4,880 | Release events + 238 FOMC decisions; 105 GDP events carry SPF consensus + surprise. |
| `question_templates` | 7 | Seeded at schema init. |
| `resolutions` | 0 | Populated in a later phase. |

> The DB file is gitignored (it does not travel with the repo). 3 optional
> FRED series were dropped (FRED renamed/discontinued them):
> `PCETRIM12M656SFRBDAL`, `GOLDAMGBD228NLBM`, `EXHOSLUSM495S`.

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
kalshi-train db-info     # quick row counts
make dashboard           # Phase 1.7 data-quality dashboard (Streamlit, :8501)
make db-browser          # Datasette web UI (:8001)
```

A full populate order: `fred → spf → text → kalshi → polymarket → calendar`
(calendar last, since it derives release events from the numeric data).

---

## Working across machines (data sync)

The SQLite DB (`data/kalshi_train.db`, ~26 MB) is **gitignored** and does
not travel with the repo. Two ways to get it onto another machine:

1. **Regenerate** — run the ingest commands above. Free, no infra. Text +
   FRED data are deterministic. ⚠️ **Kalshi/Polymarket market data drifts**
   over time (markets settle, candlesticks get truncated), so regeneration
   does *not* reproduce a past market snapshot.

2. **Cloud snapshot (recommended once data must be identical across
   machines, e.g. Phase 3+ evals)** — push/pull a compressed snapshot to
   any rclone remote (Cloudflare R2 / Backblaze B2 / S3 / Google Drive;
   ~free at this size):

   ```bash
   # one-time on each machine
   rclone config                      # create a remote, e.g. "r2"
   export DATA_REMOTE=r2:kalshi-train/kalshi_train.db.gz

   make data-push                     # after ingests, upload snapshot
   make data-pull                     # on another machine, download it
   ```

   Record the snapshot date below when you push, so results stay traceable.

   - **Latest snapshot:** _none pushed yet._

Secrets (`.env`, incl. `FRED_API_KEY`) are gitignored — copy them to each
machine manually; never commit them.

---

## Quality

- **Tests:** 101 unit tests passing; 4 integration tests auto-skip without
  network/keys. Run: `uv run pytest -m "not integration"`.
- **Lint/types:** `ruff` and `mypy --strict` both clean across 44 source
  files.
- **Toolchain note:** the local venv uses Python 3.14 via `uv`. `xgboost`
  requires the `libomp` system library (`brew install libomp`) for the
  Phase 2 test to run.

---

## To fix in Phase 4 (dataset construction)

These don't block Phase 2/3 but must be resolved before the SFT dataset is
built, because they bear on **leakage** and **market baselines**:

- **Polymarket price-history gap.** `polymarket_price_history` is empty.
  The Gamma `/markets` endpoint returns only metadata + final resolution,
  not the price *time series*. Pulling the implied-probability path needs
  Polymarket's CLOB/timeseries (or subgraph) API, keyed by each market's
  `clobTokenIds`. Until then Polymarket gives us labels but no
  point-in-time market-implied probability.
- **Event-calendar consensus gap.** `event_calendar` only carries
  `consensus`/`surprise` for **GDP** (the one series with a clean
  same-frequency SPF nowcast). Monthly prints (CPI, NFP, unemployment,
  etc.) have `consensus = NULL`, so no surprise. Filling these needs a
  real release-consensus feed (DBnomics or Trading Economics) aligned to
  each release's frequency.
- **Market ↔ Fed-timeline alignment (leakage standardization).** Market
  data is **not yet standardized to the prediction `as_of` timeline** and
  is **not yet consumed by any model**. When we build training examples,
  every market feature must be the price *as known on* the example's
  `as_of_date` (`kalshi_price_history.period_end_date <= as_of`), and
  resolutions used only as labels — mirroring the numeric PIT guard.
  Coverage caveat: Kalshi candlesticks start ~2025 and Polymarket ~Oct
  2023, so markets can only baseline *recent* events, not the full 2000+
  FRED history.
- **Non-US market noise.** A few Polymarket rows are non-US (e.g. "Canada
  recession", "Eurozone inflation") tagged by keyword; tighten the
  classifier to US-only when building the dataset.

## Known gaps / deferred work

- **3 discontinued FRED series** — `PCETRIM12M656SFRBDAL`,
  `GOLDAMGBD228NLBM`, `EXHOSLUSM495S` 400 on FRED (renamed/removed). All
  optional; need updated IDs in `data/registry.py`.
- **Beige Book pre-2011** — 2011+ captured; older issues are JS-rendered
  single-page apps (a server GET returns only a table of contents), so
  they'd need a headless browser (~80 more documents).
- **Text corpus breadth** — SEP projections, Fed speeches, and
  ECB/BoE/BLS/BEA narratives are not implemented (the README's ~5,000-doc
  target assumed these). The source-dispatch design makes each one a
  single added URL builder + parser.

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
