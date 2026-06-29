-- ════════════════════════════════════════════════════════════════════
-- Kalshi Model Train — canonical SQLite schema
-- ════════════════════════════════════════════════════════════════════
--
-- Design goals:
--
--   1. Vintage-honest time-series storage. Every observation knows
--      WHEN it was reported, so we can reconstruct "what the world
--      looked like" on any historical date — no future leakage.
--
--   2. One source of truth. This file is the schema. Both sync and
--      async code execute it via `db.connection.init_schema()`. No
--      ORMs, no migrations framework (for now): the file IS the spec.
--
--   3. Everything is `CREATE TABLE IF NOT EXISTS`, so this file is
--      idempotent and safe to run repeatedly.
--
--   4. We tag each table with its phase so users know what's empty
--      at any given point.
--
-- Numeric series store the value AS A STRING in addition to the float,
-- because some sources (CPI, GDP) come with extra precision that the
-- float would silently truncate. We standardize on `value_text` as the
-- canonical record and `value` as a convenience copy.
-- ════════════════════════════════════════════════════════════════════


-- ────────────────────────────────────────────────────────────────────
-- TABLE: series_definitions
-- Phase: 1.2
-- Purpose: metadata for every numeric time series we ingest.
-- ────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS series_definitions (
    series_id           TEXT PRIMARY KEY,           -- e.g. "CPIAUCSL"
    source              TEXT NOT NULL,              -- "FRED", "BLS", "BEA", ...
    title               TEXT NOT NULL,              -- human-readable
    units               TEXT NOT NULL DEFAULT '',   -- "Percent", "Index", "Thousands of Persons"
    frequency           TEXT NOT NULL,              -- "daily", "weekly", "monthly", "quarterly"
    seasonal_adjustment TEXT NOT NULL DEFAULT '',   -- "SA", "NSA"
    revises             INTEGER NOT NULL DEFAULT 0, -- 0=never revised, 1=may revise (use ALFRED)
    category            TEXT NOT NULL DEFAULT '',   -- "inflation", "labor", "growth", "rates", ...
    notes               TEXT NOT NULL DEFAULT '',
    first_seen          TEXT,                       -- earliest observation we have
    last_seen           TEXT,                       -- most recent observation we have
    last_ingested_at    TEXT,                       -- when we last refreshed this series
    UNIQUE (source, series_id)
);

CREATE INDEX IF NOT EXISTS idx_series_def_source   ON series_definitions(source);
CREATE INDEX IF NOT EXISTS idx_series_def_category ON series_definitions(category);


-- ────────────────────────────────────────────────────────────────────
-- TABLE: series_observations
-- Phase: 1.2
-- Purpose: the actual numeric data, with full vintage history.
--
-- Composite key:
--   (series_id, observation_date, vintage_date)
--
-- Why? CPI for September 2024 may have been reported as 2.4% on
-- October 10, then revised to 2.5% three months later. Both rows
-- coexist:
--   ('CPIAUCSL', '2024-09-01', '2024-10-10', 2.4)
--   ('CPIAUCSL', '2024-09-01', '2025-01-15', 2.5)
--
-- Our point-in-time query selects the row with the LATEST vintage_date
-- that is <= the as_of_date.
-- ────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS series_observations (
    series_id        TEXT NOT NULL,
    observation_date TEXT NOT NULL,                -- ISO date the value describes ("2024-09-01")
    vintage_date     TEXT NOT NULL,                -- ISO date this specific value was reported
    release_date     TEXT NOT NULL,                -- ISO date+time the release became public
    value            REAL,                         -- nullable: some releases are "."/missing
    value_text       TEXT,                         -- canonical string form, preserves precision
    ingested_at      TEXT NOT NULL,                -- when this row was inserted (audit)
    PRIMARY KEY (series_id, observation_date, vintage_date),
    FOREIGN KEY (series_id) REFERENCES series_definitions(series_id) ON DELETE CASCADE
);

-- Critical query pattern: "give me all observations of series X with
-- vintage_date <= as_of_date". We index by series + vintage for that.
CREATE INDEX IF NOT EXISTS idx_obs_series_vintage
    ON series_observations(series_id, vintage_date);

CREATE INDEX IF NOT EXISTS idx_obs_series_obsdate
    ON series_observations(series_id, observation_date);

CREATE INDEX IF NOT EXISTS idx_obs_release_date
    ON series_observations(release_date);


-- ────────────────────────────────────────────────────────────────────
-- TABLE: text_documents
-- Phase: 1.4
-- Purpose: full-text storage of every document we scrape — FOMC
-- statements, minutes, speeches, Beige Book, release narratives, etc.
-- ────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS text_documents (
    doc_id            TEXT PRIMARY KEY,             -- stable hash of (source, url) or natural key
    source            TEXT NOT NULL,                -- "fed", "ecb", "bls", "bea", ...
    document_type     TEXT NOT NULL,                -- "fomc_statement", "fomc_minutes", "beige_book", ...
    title             TEXT NOT NULL,
    author            TEXT NOT NULL DEFAULT '',
    published_date    TEXT NOT NULL,                -- ISO date of public release
    effective_date    TEXT,                         -- e.g. for FOMC statement, the meeting date
    body              TEXT NOT NULL,                -- full plain-text content
    body_hash         TEXT NOT NULL,                -- sha256 of body for dedup / change detection
    url               TEXT NOT NULL DEFAULT '',
    metadata_json     TEXT NOT NULL DEFAULT '{}',   -- extra source-specific fields
    ingested_at       TEXT NOT NULL,
    UNIQUE (source, document_type, published_date, title)
);

CREATE INDEX IF NOT EXISTS idx_text_published   ON text_documents(published_date);
CREATE INDEX IF NOT EXISTS idx_text_source_type ON text_documents(source, document_type);

-- Full-text search index. SQLite FTS5 is fast and ships with the lib.
-- We use it for retrieval in Phase 3 (RAG baseline) and Phase 4 (snapshot
-- generation lookups).
CREATE VIRTUAL TABLE IF NOT EXISTS text_documents_fts USING fts5(
    doc_id UNINDEXED,
    title,
    body,
    content='text_documents',
    content_rowid='rowid',
    tokenize='unicode61'
);

-- Triggers to keep the FTS index in sync.
CREATE TRIGGER IF NOT EXISTS text_documents_ai AFTER INSERT ON text_documents BEGIN
    INSERT INTO text_documents_fts(rowid, doc_id, title, body)
    VALUES (new.rowid, new.doc_id, new.title, new.body);
END;
CREATE TRIGGER IF NOT EXISTS text_documents_ad AFTER DELETE ON text_documents BEGIN
    INSERT INTO text_documents_fts(text_documents_fts, rowid, doc_id, title, body)
    VALUES('delete', old.rowid, old.doc_id, old.title, old.body);
END;
CREATE TRIGGER IF NOT EXISTS text_documents_au AFTER UPDATE ON text_documents BEGIN
    INSERT INTO text_documents_fts(text_documents_fts, rowid, doc_id, title, body)
    VALUES('delete', old.rowid, old.doc_id, old.title, old.body);
    INSERT INTO text_documents_fts(rowid, doc_id, title, body)
    VALUES (new.rowid, new.doc_id, new.title, new.body);
END;


-- ────────────────────────────────────────────────────────────────────
-- TABLE: text_chunks
-- Phase: 4 (RAG)
-- Purpose: passage-level chunks of text_documents with embeddings, for
-- point-in-time retrieval. published_date is denormalized so retrieval
-- can cheaply filter to "knowable as of date X". Embeddings are stored as
-- raw little-endian float32 bytes.
-- ────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS text_chunks (
    chunk_id        TEXT PRIMARY KEY,             -- f"{doc_id}:{chunk_index}"
    doc_id          TEXT NOT NULL,
    source          TEXT NOT NULL,
    document_type   TEXT NOT NULL,
    published_date  TEXT NOT NULL,                -- denormalized from text_documents
    chunk_index     INTEGER NOT NULL,
    text            TEXT NOT NULL,
    embedding       BLOB,                         -- float32 vector, or NULL if not embedded
    embed_model     TEXT NOT NULL DEFAULT '',
    dim             INTEGER NOT NULL DEFAULT 0,
    ingested_at     TEXT NOT NULL,
    FOREIGN KEY (doc_id) REFERENCES text_documents(doc_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chunks_pubdate ON text_chunks(published_date);
CREATE INDEX IF NOT EXISTS idx_chunks_doc     ON text_chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_type    ON text_chunks(document_type);


-- ────────────────────────────────────────────────────────────────────
-- TABLE: question_templates
-- Phase: 1.5
-- Purpose: the 7 prediction targets the project is built around.
-- Pre-populated with our targets at DB init time.
-- ────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS question_templates (
    template_id      TEXT PRIMARY KEY,             -- "fed_decision", "cpi_yoy", etc.
    title            TEXT NOT NULL,
    description      TEXT NOT NULL,
    frequency        TEXT NOT NULL,                -- expected resolution cadence
    outcome_type     TEXT NOT NULL,                -- "binary", "categorical", "binary_strike"
    notes            TEXT NOT NULL DEFAULT ''
);


-- ────────────────────────────────────────────────────────────────────
-- TABLE: kalshi_markets
-- Phase: 1.5
-- Purpose: Kalshi market metadata + outcome.
-- Schema modeled after Kalshi's API; lifted from the Black Swan repo
-- and trimmed to what we need for forecasting.
-- ────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS kalshi_markets (
    ticker                   TEXT PRIMARY KEY,
    event_ticker             TEXT NOT NULL,
    series_ticker            TEXT NOT NULL DEFAULT '',
    market_type              TEXT NOT NULL DEFAULT '',
    title                    TEXT NOT NULL DEFAULT '',
    subtitle                 TEXT NOT NULL DEFAULT '',
    yes_sub_title            TEXT NOT NULL DEFAULT '',
    no_sub_title             TEXT NOT NULL DEFAULT '',
    rules_primary            TEXT NOT NULL DEFAULT '',
    rules_secondary          TEXT NOT NULL DEFAULT '',
    open_time                TEXT,                  -- ISO timestamps
    close_time               TEXT,
    created_time             TEXT,
    settlement_time          TEXT,
    status                   TEXT NOT NULL DEFAULT '',
    result                   TEXT NOT NULL DEFAULT '', -- "yes", "no", "" (unresolved)
    settlement_value_dollars TEXT,

    -- Mapping back to our question templates. Populated by a
    -- post-ingest classifier in Phase 1.5.
    template_id              TEXT,
    strike_value             REAL,                  -- e.g. 3.0 for "CPI > 3.0%"
    strike_direction         TEXT NOT NULL DEFAULT '', -- "above", "below", "equals", "between"

    last_price_dollars       TEXT NOT NULL DEFAULT '0.0000',
    volume_fp                TEXT NOT NULL DEFAULT '0.00',
    open_interest_fp         TEXT NOT NULL DEFAULT '0.00',

    ingested_at              TEXT NOT NULL,
    last_refreshed_at        TEXT NOT NULL,

    FOREIGN KEY (template_id) REFERENCES question_templates(template_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_kalshi_event       ON kalshi_markets(event_ticker);
CREATE INDEX IF NOT EXISTS idx_kalshi_series      ON kalshi_markets(series_ticker);
CREATE INDEX IF NOT EXISTS idx_kalshi_template    ON kalshi_markets(template_id);
CREATE INDEX IF NOT EXISTS idx_kalshi_result      ON kalshi_markets(result);
CREATE INDEX IF NOT EXISTS idx_kalshi_close       ON kalshi_markets(close_time);


-- ────────────────────────────────────────────────────────────────────
-- TABLE: kalshi_price_history
-- Phase: 1.5
-- Purpose: daily price/volume snapshots per market — the historical
-- candlestick data we use as the "market-implied probability" series.
-- ────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS kalshi_price_history (
    ticker          TEXT NOT NULL,
    period_end_ts   INTEGER NOT NULL,             -- unix seconds, end of candlestick
    period_end_date TEXT NOT NULL,                -- ISO date, denormalized for easy joins
    open_dollars    TEXT,
    high_dollars    TEXT,
    low_dollars     TEXT,
    close_dollars   TEXT,                          -- canonical "implied prob" if YES
    mean_dollars    TEXT,
    yes_bid_close   TEXT,
    yes_ask_close   TEXT,
    volume_fp       TEXT,
    open_interest_fp TEXT,
    PRIMARY KEY (ticker, period_end_ts),
    FOREIGN KEY (ticker) REFERENCES kalshi_markets(ticker) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_kalshi_price_date ON kalshi_price_history(period_end_date);


-- ────────────────────────────────────────────────────────────────────
-- TABLE: polymarket_markets
-- Phase: 1.5 (optional, for cross-reference and pre-Kalshi history)
-- ────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS polymarket_markets (
    condition_id     TEXT PRIMARY KEY,
    slug             TEXT NOT NULL DEFAULT '',
    question         TEXT NOT NULL DEFAULT '',
    description      TEXT NOT NULL DEFAULT '',
    template_id      TEXT,
    strike_value     REAL,
    strike_direction TEXT NOT NULL DEFAULT '',
    start_date       TEXT,
    end_date         TEXT,
    resolved         INTEGER NOT NULL DEFAULT 0,
    outcome          TEXT NOT NULL DEFAULT '',
    metadata_json    TEXT NOT NULL DEFAULT '{}',
    ingested_at      TEXT NOT NULL,
    FOREIGN KEY (template_id) REFERENCES question_templates(template_id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS polymarket_price_history (
    condition_id  TEXT NOT NULL,
    period_end_ts INTEGER NOT NULL,
    period_end_date TEXT NOT NULL,
    yes_price     REAL,
    volume        REAL,
    PRIMARY KEY (condition_id, period_end_ts),
    FOREIGN KEY (condition_id) REFERENCES polymarket_markets(condition_id) ON DELETE CASCADE
);


-- ────────────────────────────────────────────────────────────────────
-- TABLE: event_calendar
-- Phase: 1.6
-- Purpose: every scheduled economic event, with consensus + actual
-- + surprise. The "surprise" is the most predictive single feature.
-- ────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS event_calendar (
    event_id          TEXT PRIMARY KEY,             -- stable natural ID
    event_name        TEXT NOT NULL,                -- "CPI YoY", "FOMC Decision", etc.
    series_id         TEXT,                         -- link to series_definitions when applicable
    template_id       TEXT,                         -- link to question_templates when applicable
    release_date      TEXT NOT NULL,                -- when the value was/will be released
    observation_date  TEXT,                         -- which period it describes
    consensus_value   REAL,                         -- median professional forecast pre-release
    actual_value      REAL,                         -- actual reported value
    surprise          REAL,                         -- actual - consensus, when both known
    country           TEXT NOT NULL DEFAULT 'US',
    notes             TEXT NOT NULL DEFAULT '',
    ingested_at       TEXT NOT NULL,
    FOREIGN KEY (series_id) REFERENCES series_definitions(series_id) ON DELETE SET NULL,
    FOREIGN KEY (template_id) REFERENCES question_templates(template_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_event_release    ON event_calendar(release_date);
CREATE INDEX IF NOT EXISTS idx_event_template   ON event_calendar(template_id);
CREATE INDEX IF NOT EXISTS idx_event_series     ON event_calendar(series_id);


-- ────────────────────────────────────────────────────────────────────
-- TABLE: resolutions
-- Phase: 1.5 / Phase 4
-- Purpose: the labels we train against. One row per historical
-- instance of a question_template, with the resolved outcome.
-- ────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS resolutions (
    resolution_id    TEXT PRIMARY KEY,             -- stable natural ID
    template_id      TEXT NOT NULL,                -- e.g. "cpi_yoy"
    resolution_date  TEXT NOT NULL,                -- the day the answer became known
    period_label     TEXT NOT NULL,                -- e.g. "2024-09" (CPI September 2024)
    strike_value     REAL,                         -- numeric threshold the question asked about
    strike_direction TEXT NOT NULL DEFAULT '',     -- "above", "below", etc.
    outcome          INTEGER NOT NULL,             -- 0 or 1 (binary)
    outcome_value    REAL,                         -- the actual measured number, for reference
    market_price_at_close REAL,                    -- market-implied probability at resolution
    metadata_json    TEXT NOT NULL DEFAULT '{}',
    ingested_at      TEXT NOT NULL,
    FOREIGN KEY (template_id) REFERENCES question_templates(template_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_res_template_date ON resolutions(template_id, resolution_date);
CREATE INDEX IF NOT EXISTS idx_res_date          ON resolutions(resolution_date);


-- ────────────────────────────────────────────────────────────────────
-- TABLE: ingest_runs
-- Phase: 1.x
-- Purpose: lightweight audit trail. Every ingest writes one row so we
-- know "what did we last update CPI? when?".
-- ────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ingest_runs (
    run_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    source        TEXT NOT NULL,                   -- "fred", "fomc_scraper", "kalshi", ...
    target        TEXT NOT NULL,                   -- specific series_id or doc batch name
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    status        TEXT NOT NULL DEFAULT 'running', -- "running", "ok", "error"
    rows_added    INTEGER NOT NULL DEFAULT 0,
    rows_updated  INTEGER NOT NULL DEFAULT 0,
    error_message TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_run_source ON ingest_runs(source);
CREATE INDEX IF NOT EXISTS idx_run_target ON ingest_runs(target);


-- ────────────────────────────────────────────────────────────────────
-- TABLE: metadata (general-purpose key-value store)
-- Phase: 0
-- ────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);


-- ════════════════════════════════════════════════════════════════════
-- Seed data: question templates.
-- INSERT OR IGNORE so re-running schema.sql is safe.
-- ════════════════════════════════════════════════════════════════════
INSERT OR IGNORE INTO question_templates (template_id, title, description, frequency, outcome_type, notes) VALUES
    ('fed_decision',     'Fed rate decision',          'Will the FOMC cut/hold/hike at the next meeting?', 'per-meeting (~8/yr)', 'binary_strike', 'Multiple strikes per meeting'),
    ('cpi_yoy',          'CPI YoY release vs strike',  'Will headline CPI YoY exceed/fall-below X at next release?', 'monthly', 'binary_strike', 'Multiple strikes per release'),
    ('nfp',              'Non-Farm Payrolls vs strike','Will NFP change exceed/fall-below X at next release?', 'monthly', 'binary_strike', ''),
    ('unemployment',     'Unemployment rate',          'Will U-3 rise/fall or exceed strike at next release?', 'monthly', 'binary_strike', ''),
    ('gdp',              'GDP growth vs strike',       'Will real GDP growth exceed strike at next release?', 'quarterly', 'binary_strike', ''),
    ('yield_10y',        '10Y Treasury yield direction','Will the 10Y yield close above/below strike on Friday?', 'weekly', 'binary_strike', ''),
    ('recession_12m',    'Recession within 12 months', 'NBER recession start within 12 months of as-of date?', 'monthly', 'binary', '');


INSERT OR IGNORE INTO metadata (key, value) VALUES
    ('schema_version', '1'),
    ('schema_phase',   '0');
