"""LLM client wrappers (OpenAI + Anthropic) behind a small protocol.

Design notes:

- The provider SDKs are **lazily imported** inside each client so the unit
  tests (which use a fake client) don't need ``openai`` / ``anthropic``
  installed, and Phase 0-2 stay light.
- ``CachedLLM`` wraps any client with an on-disk JSON cache keyed by
  ``(model, system, user)``. LLM calls cost money and historical prompts
  are deterministic, so we never pay twice for the same prompt. The cache
  lives under ``data/cache/llm`` (gitignored).
- Calls are synchronous: a baseline run is ~dozens of prompts, so the
  simplicity is worth more than concurrency here.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

from kalshi_train.config import PROJECT_ROOT, settings

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "llm"
MAX_TOKENS = 1024


class LLMError(RuntimeError):
    """Raised on missing API keys or provider failures."""


@runtime_checkable
class LLMClient(Protocol):
    """Minimal text-completion interface our baseline depends on."""

    @property
    def model(self) -> str: ...

    def complete(self, system: str, user: str) -> str: ...


def _cache_key(model: str, system: str, user: str) -> str:
    return hashlib.sha256(f"{model}\x00{system}\x00{user}".encode()).hexdigest()


class CachedLLM:
    """Wrap an ``LLMClient`` with a persistent on-disk response cache."""

    def __init__(self, inner: LLMClient, cache_dir: Path | None = None) -> None:
        self._inner = inner
        self._dir = cache_dir or DEFAULT_CACHE_DIR

    @property
    def model(self) -> str:
        return self._inner.model

    def complete(self, system: str, user: str) -> str:
        key = _cache_key(self.model, system, user)
        path = self._dir / f"{key}.json"
        if path.exists():
            cached: str = json.loads(path.read_text())["response"]
            return cached
        response = self._inner.complete(system, user)
        self._dir.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"model": self.model, "system": system, "user": user, "response": response}
            )
        )
        return response


class OpenAIClient:
    """OpenAI chat-completions wrapper (temperature 0 for reproducibility)."""

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self.model = model or settings.openai_model
        secret = settings.openai_api_key
        self._api_key = api_key or (secret.get_secret_value() if secret else None)
        if not self._api_key:
            raise LLMError("OPENAI_API_KEY is not set — add it to .env.")
        self._client: object | None = None

    def _ensure(self) -> object:
        if self._client is None:
            try:
                from openai import OpenAI  # noqa: PLC0415 - optional dep, lazy
            except ImportError as e:  # pragma: no cover - depends on extra
                raise LLMError("openai not installed: `uv sync --extra llm`.") from e
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def complete(self, system: str, user: str) -> str:
        client = self._ensure()
        resp = client.chat.completions.create(  # type: ignore[attr-defined]
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
        )
        content: str | None = resp.choices[0].message.content
        return content or ""


class AnthropicClient:
    """Anthropic messages wrapper (temperature 0)."""

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self.model = model or settings.anthropic_model
        secret = settings.anthropic_api_key
        self._api_key = api_key or (secret.get_secret_value() if secret else None)
        if not self._api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set — add it to .env.")
        self._client: object | None = None

    def _ensure(self) -> object:
        if self._client is None:
            try:
                from anthropic import Anthropic  # noqa: PLC0415 - optional dep, lazy
            except ImportError as e:  # pragma: no cover - depends on extra
                raise LLMError("anthropic not installed: `uv sync --extra llm`.") from e
            self._client = Anthropic(api_key=self._api_key)
        return self._client

    def complete(self, system: str, user: str) -> str:
        client = self._ensure()
        resp = client.messages.create(  # type: ignore[attr-defined]
            model=self.model,
            max_tokens=MAX_TOKENS,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        parts = [blk.text for blk in resp.content if getattr(blk, "type", None) == "text"]
        return "".join(parts)


def make_client(provider: str, model: str | None = None) -> LLMClient:
    """Build a cached client for ``openai`` or ``anthropic``."""
    provider = provider.lower()
    inner: LLMClient
    if provider == "openai":
        inner = OpenAIClient(model=model)
    elif provider == "anthropic":
        inner = AnthropicClient(model=model)
    else:
        raise LLMError(f"Unknown provider {provider!r}; use 'openai' or 'anthropic'.")
    return CachedLLM(inner)


__all__ = [
    "AnthropicClient",
    "CachedLLM",
    "LLMClient",
    "LLMError",
    "OpenAIClient",
    "make_client",
]
