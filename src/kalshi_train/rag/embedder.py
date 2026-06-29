"""Text embedding behind a tiny protocol (OpenAI by default).

The OpenAI SDK is lazily imported so unit tests can use a fake embedder
without the dependency or an API key. Embedding is batched; callers pass a
list of texts and get back a list of equal-length float vectors.
"""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable

from kalshi_train.config import settings

DEFAULT_BATCH = 128


class EmbedderError(RuntimeError):
    """Missing API key or provider failure."""


@runtime_checkable
class Embedder(Protocol):
    model: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    """OpenAI embeddings (default ``text-embedding-3-small``, dim 1536)."""

    _DIMS: ClassVar[dict[str, int]] = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        *,
        batch_size: int = DEFAULT_BATCH,
    ) -> None:
        self.model = model or settings.openai_embed_model
        self.dim = self._DIMS.get(self.model, 1536)
        self._batch_size = batch_size
        secret = settings.openai_api_key
        self._api_key = api_key or (secret.get_secret_value() if secret else None)
        if not self._api_key:
            raise EmbedderError("OPENAI_API_KEY is not set — add it to .env.")
        self._client: object | None = None

    def _ensure(self) -> object:
        if self._client is None:
            try:
                from openai import OpenAI  # noqa: PLC0415 - optional dep, lazy
            except ImportError as e:  # pragma: no cover - depends on extra
                raise EmbedderError("openai not installed: `uv sync --extra llm`.") from e
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._ensure()
        out: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = [t.replace("\n", " ") for t in texts[start : start + self._batch_size]]
            resp = client.embeddings.create(model=self.model, input=batch)  # type: ignore[attr-defined]
            out.extend(item.embedding for item in resp.data)
        return out


__all__ = ["DEFAULT_BATCH", "Embedder", "EmbedderError", "OpenAIEmbedder"]
