# Kalshi Model Train

A learning project: train a fine-tuned LLM to make calibrated probabilistic forecasts on Kalshi macroeconomic markets (Fed decisions, CPI, employment, GDP, yields, recessions) — and rigorously measure whether it beats vanilla LLMs, classical ML baselines, and the market itself.

> **Status:** Phase 1 data layer code-complete through 1.6 — numeric (FRED/SPF), text corpus (Fed statements/minutes/Beige Book), markets (Kalshi + Polymarket), and the event calendar are all built and tested. Phase 2 XGBoost baseline prototyped. Remaining: add the FRED API key to populate numeric data, then Phase 1.7 (data-quality dashboard). 94 unit tests passing.

---

## Table of contents

1. [Project goal](#project-goal)
2. [The seven prediction targets](#the-seven-prediction-targets)
3. [System architecture](#system-architecture)
4. [Phased build plan](#phased-build-plan)
5. [Reusing the Black Swan project](#reusing-the-black-swan-project)
6. [Project conventions](#project-conventions)
7. [How to inspect the database](#how-to-inspect-the-database)
8. [Glossary](#glossary)

---

## Project goal

Build a system that, given a Kalshi macro market and the world's information up to that point, outputs a **calibrated probability** of the YES outcome. Measure that probability against three baselines:

1. **Vanilla LLM** (GPT-4o or Claude, with the same context but no fine-tuning).
2. **Classical ML** (XGBoost on hand-engineered structured features).
3. **The market itself** (Kalshi price or, where Kalshi history is too short, Fed funds futures and other implied-probability sources).

Primary evaluation metrics: **Brier score**, **log loss**, **reliability diagrams**, and a paper-trading **simulated PnL with fractional-Kelly sizing**.

This is a **learning project**. Each phase has an explicit pedagogical checkpoint where we'll stop, explain what was built, and verify the database is legible before moving on.

---

## The seven prediction targets

We target a portfolio of US macroeconomic binary questions. Each is its own **question template** — the same shape of question asked over and over through history, producing thousands of training examples.

| # | Question template | Frequency | Why |
|---|---|---|---|
| 1 | Fed rate decision (cut / hold / hike, ±25bps strikes) | ~8/year | High signal-to-noise, rich text context, clean resolution |
| 2 | CPI YoY release vs strike | Monthly | Most-watched inflation print; many strike points multiply examples |
| 3 | Non-Farm Payrolls vs strike | Monthly | Market-moving, noisy, lots of leading indicators |
| 4 | Unemployment rate direction / level | Monthly | Slow-moving, autocorrelated, connects to recession signal |
| 5 | GDP growth vs strike | Quarterly | Headline number; smaller N but strong leading indicators |
| 6 | 10-year Treasury yield direction (weekly) | Weekly | High data quantity, helps transfer learning to other targets |
| 7 | NBER recession within next 12 months | Monthly snapshot | Rare-event prediction; text-rich signal in Fed language |

Together these give us roughly **50,000+ training examples** after expanding to per-date snapshots and strike-point variations.

---

## System architecture

```text
┌────────────────────────────────────────────────────────────────────┐
│                       KALSHI MODEL TRAIN                            │
└────────────────────────────────────────────────────────────────────┘

  [DATA LAYER]               [MODEL LAYER]            [EVAL / EXEC LAYER]

  Kalshi client ──┐          ┌─────────────┐         ┌───────────────┐
  Polymarket ────┤          │  Feature    │         │ Backtester    │
  Fed funds fut ─┤──data──▶ │  builder    │         │ (point-in-time)│
                  │          ├─────────────┤         └───────────────┘
  FRED / ALFRED ──┤          │ Baseline:   │         ┌───────────────┐
  BLS / BEA ──────┤──macro──▶│ XGBoost     │──prob──▶│ Evaluator     │
  Eurostat / ECB ─┤          ├─────────────┤         │ Brier / LL    │
                  │          │ Baseline:   │         │ Calibration   │
  Fed / Beige ────┤          │ vanilla LLM │         └───────────────┘
  FOMC scraper ──┤──text───▶ ├─────────────┤         ┌───────────────┐
  BLS/BEA narr ──┤          │ Fine-tuned  │         │ Paper trader  │
  GDELT (later) ─┘          │ LLM (LoRA)  │         │ Fractional    │
                             ├─────────────┤         │ Kelly sizing  │
                             │ Ensemble    │         └───────────────┘
                             └─────────────┘
                                    ▲
                                    │
                             ┌──────┴──────┐
                             │ Continued   │
                             │ pretraining │
                             │ + LoRA SFT  │
                             └─────────────┘
```

---

## Phased build plan

Each phase has a **goal**, **work items**, **learning checkpoint** (what concepts the user will understand after), and an **exit criterion** (how we know the phase is done).

We stop at every phase boundary and walk through what was built before proceeding.

### Phase 0 — Foundations

**Goal:** Clean, tested, modern Python project skeleton.

**Work items:**

- Install/verify Python 3.11+, install `uv` for package management.
- Project structure (see [Project conventions](#project-conventions)).
- `pyproject.toml` with all dependencies pinned.
- Pre-commit hooks: `ruff` (lint + format), `mypy` (types).
- `pytest` skeleton with one trivial passing test.
- Initial git repo + sensible `.gitignore`.
- Inspect Black Swan repo and copy the reusable Kalshi client.
- Database initialization: SQLite at `data/kalshi_train.db` with sane defaults (WAL mode, foreign keys on).

**Learning checkpoint:** What makes a reproducible ML project. Why `uv`. Why we lint+type-check from day one. The folder structure and where things go.

**Exit criterion:** `uv run pytest` passes. `uv run python -m kalshi_train --version` prints the version. The database file exists and contains one or more empty tables defined by the schema.

---

### Phase 1 — Data Layer

**Goal:** Every byte of data needed (numeric + text + markets + calendar) ingested **with point-in-time vintage discipline** into the local SQLite database.

This phase is the longest and most important. We will execute it in sub-phases so we can verify the DB after each.

#### Phase 1.1 — Schema & vintage discipline foundation ✓ COMPLETE

- ✓ Full SQLite schema (see [Database schema](#database-schema-conceptual)).
- ✓ `pit_value`, `pit_frame`, `pit_history` — the *only* legal way features touch labels.
- ✓ Property-based tests (Hypothesis) that attempt to construct leakage and assert it's rejected.
- ✓ `docs/data_spec.md` locking the ~71 priority numeric series with explicit vintage policy per series.
- ✓ CLI: `kalshi-train pit SERIES --as-of YYYY-MM-DD` for spot-checking.

**Checkpoint:** see the conversation log for the worked CPI example explaining observation_date / vintage_date / release_date and how the leakage guard works.

#### Phase 1.2 — FRED / ALFRED numeric ingestion ✓ CODE COMPLETE (awaiting API key)

- ✓ Vintage-aware async FRED client (`data/sources/fred.py`) with rate limiting, retries, ALFRED vintage queries.
- ✓ Series registry (`data/registry.py`) codifying the ~60 priority series from `docs/data_spec.md`.
- ✓ Orchestrator (`data/ingest_fred.py`) that walks the registry and writes through `db/ingest.py`.
- ✓ CLI: `kalshi-train ingest fred [--series ID]... [--skip-optional] [--observation-start DATE] [--limit N]`.
- ✓ 16 unit tests (mocked) + 2 live integration tests (auto-skipped without API key).
- ⏳ To actually run: drop a free FRED key in `.env`, then `kalshi-train ingest fred --skip-optional`.

**Checkpoint:** after the key is added, `make db-summary` shows ~30 series with thousands of observations, and `kalshi-train pit CPIAUCSL --as-of 2024-11-07` returns the actual contemporaneous CPI value.

#### Phase 1.3 — Other numeric sources

- BLS, BEA, Eurostat, ECB, BoE numeric series.
- CME Fed funds futures historical settlements.

**Checkpoint:** verify all ~71 series have data with full date ranges.

#### Phase 1.4 — Text corpus ingestion ✓ CODE COMPLETE

Federal Reserve communications, fetched via deterministic URLs (no key):

- ✓ FOMC statement scraper — URLs derived from the FOMC calendar (reuses Phase 1.6).
- ✓ FOMC minutes scraper — same calendar, with an approximate +21-day publication date.
- ✓ Beige Book scraper — probes all release months.
- ✓ Async `FedTextClient` (404 → `None` so candidate URLs can be safely probed) + pure, testable HTML parsers (`data/sources/fed_text.py`).
- ✓ Clean body extraction (strips nav/script/footer) + dedup via `body_hash` + automatic FTS5 indexing.
- ✓ CLI: `kalshi-train ingest text [--type fomc_statement|fomc_minutes|beige_book] [--start --end --limit]`.
- ✓ Per-document **ordered URL fallbacks** to handle the Fed's era-specific schemes and 2-day meetings (statement/minutes filename uses the meeting's *last* day): modern `pressreleases` + legacy `boarddocs` paths, each tried for the listed date and date+1.
- ✓ 9 unit tests (HTML fixtures, parser, probe-and-skip, idempotency, FTS search). **Verified live — full 2000→2025 backfill**: 504 documents stored (222 FOMC statements, 207 minutes, 75 Beige Books), all FTS-searchable.

> Known gap: Beige Books before 2017 use legacy exact-date URLs we don't yet generate (only 2017+ captured). SEP projections, Fed speeches, and ECB/BoE/BLS/BEA narratives are deferred; the source-dispatch design makes each a single added URL builder + parser.

**Checkpoint:** `kalshi-train ingest text` stores clean statement/minutes/Beige Book text, searchable via `text_documents_fts`.

#### Phase 1.5 — Kalshi & Polymarket ingestion ✓ CODE COMPLETE

- ✓ Wired the Phase-0 Kalshi client into a real ingestor (`data/ingest_kalshi.py`): market metadata → `kalshi_markets`, daily candlesticks → `kalshi_price_history` (the market-implied probability series).
- ✓ Macro **allowlist + classifier** (`data/kalshi_macro.py`): series-prefix and title-keyword matching maps markets to our 7 templates and parses strikes (above/below/between). Uses the live `KX`-prefixed series (`KXFED`, `KXCPI`, `KXCPIYOY`, `KXU3`, `KXPAYROLLS`, `KXGDP`).
- ✓ Candlesticks use the working series-scoped endpoint (the bare `/historical` path 404s).
- ✓ Polymarket Gamma client (`data/sources/polymarket.py`, keyless) + ingestor → `polymarket_markets`, with best-effort resolution parsing.
- ✓ CLI: `kalshi-train ingest kalshi [--series ... --status --no-prices --max]`, `kalshi-train ingest polymarket [--max]`.
- ✓ 10 unit tests (classifier, strike parsing, flattening, candlestick mapping, resolution, idempotency). **Verified live**: 172 Kalshi macro markets across 5 templates; 733 candlestick rows for GDP markets; 30 Polymarket macro markets from 966 scanned.

**Checkpoint:** `kalshi-train db-info` shows macro markets per template in `kalshi_markets`, with price history in `kalshi_price_history`.

#### Phase 1.6 — Calendar / event metadata ✓ CODE COMPLETE

Built *from data we already hold* — no new API key needed:

- ✓ Economic release calendar derived from `series_observations`: each period's **first print** (earliest vintage) becomes an `event_calendar` row with `actual_value`, `release_date`, `observation_date` (`data/calendar_registry.py` + `data/ingest_calendar.py`).
- ✓ `surprise = actual - consensus` where a same-frequency forecast exists (real GDP `GDPC1` vs SPF nowcast `SPF_RGDP_MEDIAN_NOWCAST`). Monthly prints leave consensus NULL until a release-consensus feed (DBnomics / Trading Economics) is wired.
- ✓ FOMC meeting schedule → one `fed_decision` row per meeting, decision classified cut / hold / hike from `DFEDTARU` via the PIT interface. This closes the Phase 2 loop: `fomc_calendar.fomc_meeting_dates` now prefers `event_calendar` over the static fallback.
- ✓ CLI: `kalshi-train ingest calendar [--start DATE] [--end DATE] [--skip-releases] [--skip-fomc]`.
- ✓ 7 unit tests on a synthetic DB (first-print vs revision, surprise math, decision classification, idempotency, DB-preferred FOMC dates).

**Checkpoint:** `kalshi-train ingest calendar` populates the FOMC schedule immediately; release events fill in once `ingest fred` / `ingest spf` have run, so every ingested release maps to a calendar entry by construction.

> **Note on sequencing:** Phase 2 (XGBoost) was built early as an end-to-end vertical slice, ahead of sub-phases 1.4–1.6. Those are now complete; the data layer is back in order.

#### Phase 1.7 — Data quality dashboard ✓ COMPLETE

- ✓ Read-only Streamlit dashboard (`src/kalshi_train/dashboard/`): table sizes, per-series numeric coverage + staleness, empty-series report, text-corpus coverage by year, Kalshi/Polymarket coverage, calendar consensus/surprise availability, ingest audit log, and an interactive point-in-time spot-check.
- ✓ Reporting logic lives in pure, unit-tested query functions (`dashboard/queries.py`); the Streamlit script is a thin render layer.
- ✓ Launch with `make dashboard` (installs the `dashboard` extra) → http://localhost:8501.
- ✓ 6 unit tests on the query functions (incl. empty-DB safety).

**Learning checkpoint for Phase 1:** Why 80% of ML is data engineering. Point-in-time discipline. The difference between observation date, release date, and vintage date. Why we never trust "current" data for historical predictions.

**Exit criterion:** A snapshot report shows all priority series and text sources ingested with full coverage. The point-in-time query passes its property tests. The user can run `make db-shell` and browse the database manually.

---

### Phase 2 — Baseline ML (XGBoost)

**Goal:** A classical XGBoost model on hand-engineered features that beats trivial baselines, evaluated with proper temporal cross-validation.

**Work items:**

- Define the first target: "Will the Fed cut rates at the next FOMC meeting?" (binary).
- Hand-engineer ~20-30 features per target using only the point-in-time interface.
- Temporal train/val/test split (no random shuffling).
- `TimeSeriesSplit` cross-validation.
- Brier score, log loss, calibration plot, reliability diagram.
- Feature importance analysis.

**Learning checkpoint:** Why random k-fold is forbidden on time series. What proper temporal CV looks like. Reliability diagrams. Brier score as a *proper scoring rule*. Hyperparameter tuning. Overfitting in small data.

**Exit criterion:** XGBoost beats "always predict prior" and "always predict 0.5" baselines on a held-out test set. Reports stored in `reports/phase2_xgboost.md`.

---

### Phase 3 — LLM Baseline (no fine-tuning) ✓ CODE COMPLETE (awaiting API key)

**Goal:** Establish the un-fine-tuned LLM baseline that fine-tuning will have to beat.

**Built:**

- ✓ `LLMClient` protocol with OpenAI + Anthropic wrappers (`llm/client.py`), SDKs lazily imported, temperature 0.
- ✓ On-disk response cache (`CachedLLM` → `data/cache/llm`) so prompts are never paid for twice.
- ✓ Prompt rendering of a point-in-time feature row + robust probability parsing (`llm/prompt.py`).
- ✓ Orchestrator (`training/phase3_llm.py`) running the LLM on the **same** Fed-cut test split as Phase 2, compared to XGBoost + trivial baselines, with a reliability plot.
- ✓ CLI: `kalshi-train baseline llm [--provider openai|anthropic] [--model ...]`.
- ✓ 12 unit tests (prompt, parsing, cache, end-to-end with a fake LLM). **To run live:** add `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` to `.env`.

**Learning checkpoint:** Prompt engineering for forecasting. Why vanilla LLMs are systematically miscalibrated. The "consult the experts" prompt pattern. Token-level vs text-level probability extraction.

**Exit criterion:** Report comparing XGBoost vs vanilla LLM vs market vs trivial baselines on the same test set, stored at `reports/phase3_llm_baseline.md`.

---

### Phase 4 — Fine-tuning Dataset Construction

**Goal:** A high-quality SFT (supervised fine-tuning) dataset of ~40,000-50,000 (prompt, completion) pairs.

**Work items:**

- Snapshot generator: for each historical event, generate examples at multiple lookback horizons (90, 60, 30, 14, 7, 3, 1 days before resolution).
- Prompt templating (text rendering of features).
- Ideal-completion generation:
  - Calibrated probability anchored to historical base rate, not raw 0/1.
  - Synthetic reasoning chains generated by a strong LLM, with the answer replaced by the calibrated probability.
- Group-aware train/val/test split (whole resolutions to one side; never split snapshots of the same event).
- Dataset stats + sanity-check report.

**Learning checkpoint:** Why dataset quality dominates model architecture. Why training on hard 0/1 labels for probabilistic tasks teaches over-confidence. The format of (prompt, completion) SFT data.

**Exit criterion:** `data/sft/train.jsonl`, `data/sft/val.jsonl`, `data/sft/test.jsonl` exist with the expected counts and pass sanity checks (no leakage, balanced across targets, reasoning chains parse).

---

### Phase 4.5 — Continued pretraining (optional but recommended)

**Goal:** A base LLM that already "speaks Fed" before we do SFT.

**Work items:**

- Tokenize the full text corpus (~50-100M tokens).
- Set up a RunPod / Lambda A100 instance.
- Run LoRA continued pretraining on Llama-3.1-8B-Instruct for ~2 epochs.
- Save adapter to `models/pretrain_lora/`.

**Learning checkpoint:** Continued pretraining vs instruction tuning. Why domain adaptation helps when SFT data is limited.

**Exit criterion:** Adapter file saved. Perplexity on a held-out chunk of macro text dropped vs base model.

---

### Phase 5 — LoRA Supervised Fine-Tuning

**Goal:** A fine-tuned LLM that beats vanilla LLM and XGBoost on Brier on the held-out test set.

**Work items:**

- Training script using `transformers` + `peft` + `trl`.
- LoRA on top of the (optionally pretrained) base model.
- Completion-only loss masking.
- Hyperparameter sweep (LR, rank, epochs).
- Weights & Biases logging.
- Evaluation on the same test set as Phases 2-3.

**Learning checkpoint:** What a training loop actually looks like. LoRA math intuition. Early stopping. Mixed precision. The full modern HF training stack.

**Exit criterion:** A fine-tuned adapter beats the vanilla LLM on Brier and log loss on the test set. Report at `reports/phase5_finetune.md`.

---

### Phase 6 — Ensembling & Calibration

**Goal:** A final stacked ensemble that beats every individual model, properly calibrated.

**Work items:**

- Logistic-regression stacker over XGBoost + fine-tuned LLM (+ market price as a feature).
- Post-hoc calibration via isotonic regression.
- Cluster-robust standard errors on improvement metrics.

**Learning checkpoint:** Ensembling theory. Isotonic vs Platt scaling. Cluster-robust inference (since per-resolution snapshots are not independent).

**Exit criterion:** Best ensemble beats every component on Brier, log loss, and reliability. Final report at `reports/phase6_ensemble.md`.

---

### Phase 7 — Paper Trading & PnL Simulator

**Goal:** A backtest that takes model probabilities and historical Kalshi prices and reports simulated PnL.

**Work items:**

- Backtest harness that replays Kalshi prices.
- Fractional-Kelly position sizing.
- PnL, hit rate, Sharpe, max drawdown.
- Sensitivity analysis (slippage, threshold).

**Learning checkpoint:** Kelly criterion. Why full Kelly will ruin you. The gap between EV-positive and risk-adjusted-positive.

**Exit criterion:** Backtest report at `reports/phase7_paper.md` showing realistic PnL with confidence intervals.

---

### Phase 8 — Live Forward-Testing & Monitoring

**Goal:** A daily cron that pulls new Kalshi markets, predicts, logs, and tracks calibration over time.

**Work items:**

- Scheduled job (launchd on macOS).
- Streamlit dashboard for predictions, calibration drift, simulated PnL.
- Drift alerts.

**Learning checkpoint:** Backtest-vs-live gap. Slippage. Model staleness. When to retrain.

**Exit criterion:** Dashboard running, daily predictions logged for 2+ weeks.

---

## Reusing the Black Swan project

After reading [emilija-tashevska/black-swan-event-intelligence](https://github.com/emilija-tashevska/black-swan-event-intelligence), here is what carries over:

| File | Reusability | Plan |
|---|---|---|
| `backend/kalshi.py` | **High** — clean async client with pagination, retries, rate limiting | Lift wholesale into `kalshi_train/data/sources/kalshi.py` with our additions (no event-summary path needed) |
| `backend/models.py` (Candlestick, etc.) | **Medium** — Pydantic models for Kalshi responses are useful | Lift the API-response models; drop the black-swan-specific ones |
| `backend/database.py` | **Low** — schema is purpose-built for black swans, not vintaged macro data | Don't lift; design our own schema from scratch |
| `backend/collector.py` orchestration | **Low** — black-swan-specific filter logic, AI summary logic | Don't lift; we have different orchestration needs |
| Sports/category prefix lists | **Medium** — we need the inverse: filter *for* macro markets, not against sports | Borrow the pattern, build a macro-prefix allowlist |
| FastAPI server | **Not relevant** — we're not exposing a web API | Skip |
| Next.js frontend | **Not relevant** — we'll use Streamlit for our internal dashboards | Skip |

**Net:** ~300 lines of Python carry over, mostly the Kalshi HTTP client and Pydantic models.

---

## Project conventions

### Folder structure (target)

```text
Kalshi-Model-Train/
├── README.md                    ← this file
├── pyproject.toml               ← package manifest
├── .pre-commit-config.yaml      ← ruff, mypy hooks
├── Makefile                     ← common commands
├── .env.example                 ← API keys template (no secrets committed)
├── data/
│   ├── kalshi_train.db          ← main SQLite DB (gitignored)
│   ├── raw/                     ← raw text corpus (HTML, PDFs) (gitignored)
│   └── sft/                     ← SFT JSONL files (gitignored)
├── models/                      ← trained adapters (gitignored)
├── reports/                     ← markdown reports per phase
├── notebooks/                   ← Jupyter for exploration
├── src/kalshi_train/
│   ├── __init__.py
│   ├── config.py                ← settings via pydantic-settings
│   ├── db/
│   │   ├── schema.sql           ← canonical schema
│   │   ├── connection.py
│   │   └── point_in_time.py     ← THE leakage-safe query interface
│   ├── data/
│   │   ├── sources/             ← one module per data source
│   │   │   ├── fred.py
│   │   │   ├── bls.py
│   │   │   ├── bea.py
│   │   │   ├── kalshi.py        ← (adapted from black-swan repo)
│   │   │   ├── polymarket.py
│   │   │   ├── fomc_scraper.py
│   │   │   ├── beige_book.py
│   │   │   └── ecb.py
│   │   └── ingest.py            ← orchestration
│   ├── features/                ← Phase 2+
│   ├── models/                  ← Phase 2+
│   ├── training/                ← Phase 5+
│   ├── eval/                    ← Phase 6+
│   └── trading/                 ← Phase 7+
├── tests/
│   ├── test_point_in_time.py    ← critical
│   └── ...
└── scripts/
    ├── inspect_db.py            ← user-facing DB inspection
    └── ...
```

### Python tooling

- **Package manager:** `uv` (modern, ~10x faster than `pip`).
- **Lint + format:** `ruff` (does both).
- **Type-check:** `mypy --strict` on the `src/` tree.
- **Tests:** `pytest` + `hypothesis` for property tests.
- **Notebooks:** Jupyter, kept in `notebooks/` and not run as part of CI.

### Versioning / git

- `main` branch.
- Conventional commits encouraged but not enforced.
- `data/`, `models/`, `*.db`, `*.jsonl`, `.env` all gitignored.

### Secrets

API keys live in `.env` (gitignored), loaded via `python-dotenv` and validated by `pydantic-settings` in `src/kalshi_train/config.py`. `.env.example` is committed with placeholders.

You'll need free API keys for: **FRED**, **BLS**, **BEA**, **Kalshi**. We may also use **OpenAI** and **Anthropic** in Phase 3 (paid, but cheap).

---

## How to inspect the database

You asked to be able to always look at the DB in legible format. We support **three** ways:

### Option A — `make db-shell` (interactive)

Opens a SQLite shell with column-mode display and headers enabled. Great for ad-hoc SQL.

```bash
make db-shell
sqlite> .tables
sqlite> SELECT * FROM series_observations WHERE series_id = 'CPIAUCSL' LIMIT 10;
```

### Option B — `make db-summary` (one-shot overview)

Runs `scripts/inspect_db.py`, which prints a compact report:

- Number of rows in each table.
- Date ranges of numeric series.
- Document count per text source.
- Recent Kalshi markets ingested.

### Option C — Datasette (web UI, no code) [recommended]

For full interactive browsing in the browser, with sortable tables, JSON export, and SQL queries:

```bash
make db-browser   # launches datasette on http://localhost:8001
```

Datasette is a single dependency, runs locally, and gives you a beautiful read-only web UI over the SQLite file. We'll wire this up in Phase 0.

---

## Database schema (conceptual)

Designed for **vintage-honest time-series ML**. Here are the core tables (full DDL lives in `src/kalshi_train/db/schema.sql` once Phase 0 is done):

- **`series_definitions`** — metadata for each numeric series (FRED ID, name, frequency, units, revises Y/N).
- **`series_observations`** — the actual numeric data, with `observation_date`, `release_date`, `vintage_date`, and `value`. Same `series_id` × `observation_date` can have multiple rows (one per vintage).
- **`text_documents`** — full-text storage with `source`, `document_type`, `title`, `published_date`, `body`, `url`. FTS5 index for fast search.
- **`kalshi_markets`** — market metadata (ticker, question, open, close, outcome).
- **`kalshi_prices`** — price snapshots over time.
- **`event_calendar`** — economic release calendar with consensus + actual + surprise.
- **`question_templates`** — our 7 target families.
- **`resolutions`** — labeled outcomes for each historical instance of each question template (this is what we train against).

The **`point_in_time_query(as_of_date, ...)`** function — defined in code, not as a SQL view — is the only legal way to read this data for ML purposes. It guarantees no future information leaks into features.

---

## Glossary

**Brier score**: $BS = \frac{1}{N}\sum (p_i - o_i)^2$ where $p_i$ is predicted probability, $o_i$ is the 0/1 outcome. Lower is better.

**Calibration**: when you say 70%, does it happen 70% of the time? Orthogonal to accuracy.

**Continued pretraining**: unsupervised next-token training on a domain corpus, to specialize a general LLM before SFT.

**FRED / ALFRED**: Federal Reserve Economic Data and its Archival vintage cousin. Free, comprehensive, our primary numeric data source.

**LoRA**: Low-Rank Adaptation. Fine-tune a tiny adapter matrix instead of all model weights. Saves ~100x memory.

**Point-in-time**: building features using only data that was actually known at the time of the historical prediction, not as-of-today data.

**SFT**: Supervised Fine-Tuning. Training on (prompt, ideal_completion) pairs.

**Vintage**: the version of a data point as known at a particular historical date. CPI for September 2024 may have been "2.4%" at release and revised to "2.5%" later — those are two vintages of the same observation.

---

## Next steps

We are about to start **Phase 0**. After Phase 0 completes, we will stop, walk through the project structure together, and only then proceed to Phase 1.1.
