"""Centralized settings, loaded from environment / `.env`.

We use pydantic-settings so every config value is type-checked and
documented in one place. The same `Settings` object is imported wherever
config is needed, eliminating module-level os.getenv calls scattered
through the codebase.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root is two levels up from this file: src/kalshi_train/config.py
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """All runtime configuration for the project.

    Values are loaded from environment variables, then from `.env` if
    present. Secrets are wrapped in `SecretStr` so they never appear in
    logs by accident.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_prefix="",  # we use full names so .env reads naturally
        extra="ignore",
        case_sensitive=False,
    )

    # ── Database ────────────────────────────────────────────────────
    kalshi_train_db_path: Path = Field(
        default=PROJECT_ROOT / "data" / "kalshi_train.db",
        description="Path to the SQLite database file.",
    )
    kalshi_train_log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # ── Data source API keys (added as we hit each phase) ──────────
    fred_api_key: SecretStr | None = None
    bls_api_key: SecretStr | None = None
    bea_api_key: SecretStr | None = None
    kalshi_api_key_id: str | None = None
    kalshi_private_key_path: Path | None = None

    # ── LLM baselines (Phase 3) ─────────────────────────────────────
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o"
    anthropic_model: str = "claude-sonnet-4-6"
    openai_embed_model: str = "text-embedding-3-small"

    # ── Experiment tracking (Phase 5+) ──────────────────────────────
    wandb_api_key: SecretStr | None = None

    # ── Helpers ─────────────────────────────────────────────────────
    @property
    def db_url(self) -> str:
        """SQLite URI suitable for sqlite3 / aiosqlite."""
        return f"sqlite:///{self.kalshi_train_db_path}"

    @property
    def schema_path(self) -> Path:
        return PROJECT_ROOT / "src" / "kalshi_train" / "db" / "schema.sql"


# Singleton for the whole codebase to import.
settings = Settings()
