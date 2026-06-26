# Kalshi Model Train — convenience commands.
# All commands run via `uv` so they use the project-managed Python.

.PHONY: help install dev-install sync lock test test-fast lint format typecheck check \
        db-init db-shell db-summary db-browser dashboard data-push data-pull clean clean-cache \
        pre-commit-install pre-commit-run

help:
	@echo "Kalshi Model Train — available targets:"
	@echo ""
	@echo "  Setup:"
	@echo "    install            Install runtime dependencies"
	@echo "    dev-install        Install with dev + data-sources extras"
	@echo "    sync               Sync the lockfile (after editing pyproject.toml)"
	@echo "    pre-commit-install Install git pre-commit hooks"
	@echo ""
	@echo "  Quality:"
	@echo "    lint               Run ruff lint"
	@echo "    format             Run ruff format (writes changes)"
	@echo "    typecheck          Run mypy --strict on src/"
	@echo "    test               Run all tests"
	@echo "    test-fast          Run tests excluding slow + integration"
	@echo "    check              Run lint + typecheck + test-fast"
	@echo ""
	@echo "  Database (read-only inspection):"
	@echo "    db-init            Initialize an empty database with the schema"
	@echo "    db-shell           Open an interactive SQLite shell"
	@echo "    db-summary         Print a one-shot DB summary report"
	@echo "    db-browser         Launch Datasette read-only web UI on :8001"
	@echo "    dashboard          Launch the Streamlit data-quality dashboard"
	@echo ""
	@echo "  Data sync (set DATA_REMOTE, e.g. r2:bucket/kalshi_train.db.gz):"
	@echo "    data-push          Compress + upload the DB snapshot via rclone"
	@echo "    data-pull          Download + decompress the DB snapshot via rclone"
	@echo ""
	@echo "  Cleanup:"
	@echo "    clean              Remove build / cache artifacts"
	@echo "    clean-cache        Remove ruff/mypy/pytest caches only"

# ── Setup ──────────────────────────────────────────────────────────────

install:
	uv sync

dev-install:
	uv sync --extra dev --extra data-sources

sync:
	uv lock

pre-commit-install:
	uv run pre-commit install

pre-commit-run:
	uv run pre-commit run --all-files

# ── Quality ────────────────────────────────────────────────────────────

lint:
	uv run ruff check src tests scripts

format:
	uv run ruff format src tests scripts
	uv run ruff check --fix src tests scripts

typecheck:
	uv run mypy

test:
	uv run pytest

test-fast:
	uv run pytest -m "not slow and not integration"

check: lint typecheck test-fast

# ── Database ───────────────────────────────────────────────────────────

DB_PATH ?= data/kalshi_train.db

db-init:
	uv run python -m kalshi_train.scripts.init_db

db-shell:
	@echo "Opening SQLite shell on $(DB_PATH). Type .help for commands, .quit to exit."
	@sqlite3 -column -header $(DB_PATH)

db-summary:
	uv run python scripts/inspect_db.py

db-browser:
	@echo "Launching Datasette on http://localhost:8001 (read-only)."
	uv run datasette serve $(DB_PATH) --port 8001 --immutable $(DB_PATH)

dashboard:
	@echo "Launching data-quality dashboard on http://localhost:8501"
	uv run --extra dashboard streamlit run src/kalshi_train/dashboard/app.py

# ── Data sync (cloud snapshot via rclone) ──────────────────────────────
# The SQLite DB is gitignored. To move it between machines, snapshot it to
# any rclone remote (Cloudflare R2 / Backblaze B2 / S3 / Google Drive).
#
#   1. Install rclone:  https://rclone.org/install/
#   2. Configure a remote:  rclone config   (e.g. name it "r2")
#   3. Point DATA_REMOTE at an object path, then push/pull:
#        export DATA_REMOTE=r2:kalshi-train/kalshi_train.db.gz
#        make data-push      # after running ingests, upload the snapshot
#        make data-pull      # on another machine, download it
#
# Note: text + FRED data are reproducible via the ingest commands, but
# Kalshi/Polymarket market data drifts over time — snapshot it here when
# you need byte-identical data across machines (e.g. for Phase 3+ evals).
DATA_REMOTE ?=

data-push:
	@test -n "$(DATA_REMOTE)" || { echo "Set DATA_REMOTE, e.g. r2:bucket/kalshi_train.db.gz"; exit 1; }
	@command -v rclone >/dev/null || { echo "rclone not installed: https://rclone.org/install/"; exit 1; }
	@test -f "$(DB_PATH)" || { echo "No DB at $(DB_PATH); run the ingest commands first."; exit 1; }
	@echo "Compressing $(DB_PATH) and uploading to $(DATA_REMOTE)..."
	@gzip -c "$(DB_PATH)" > "$(DB_PATH).gz"
	@rclone copyto "$(DB_PATH).gz" "$(DATA_REMOTE)" --progress
	@rm -f "$(DB_PATH).gz"
	@echo "Pushed. Remember to update the snapshot date in docs/STATUS.md."

data-pull:
	@test -n "$(DATA_REMOTE)" || { echo "Set DATA_REMOTE, e.g. r2:bucket/kalshi_train.db.gz"; exit 1; }
	@command -v rclone >/dev/null || { echo "rclone not installed: https://rclone.org/install/"; exit 1; }
	@mkdir -p "$(dir $(DB_PATH))"
	@echo "Downloading $(DATA_REMOTE) and decompressing to $(DB_PATH)..."
	@rclone copyto "$(DATA_REMOTE)" "$(DB_PATH).gz" --progress
	@gunzip -c "$(DB_PATH).gz" > "$(DB_PATH)"
	@rm -f "$(DB_PATH).gz"
	@echo "Pulled to $(DB_PATH)."

# ── Cleanup ────────────────────────────────────────────────────────────

clean: clean-cache
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +

clean-cache:
	rm -rf .ruff_cache .mypy_cache .pytest_cache .hypothesis .coverage htmlcov
