"""Polymarket Gamma API client (markets metadata).

We use the public, keyless Gamma REST API
(``https://gamma-api.polymarket.com``) rather than the on-chain subgraph:
it's simpler and sufficient for our purpose — cross-referencing macro
markets and capturing resolutions, including some that predate Kalshi's
macro coverage. (Tick-level price history via the subgraph is a deeper
follow-up; this pass stores market metadata + outcomes.)

The client is a thin async wrapper; ``PolymarketMarketModel`` parses the
subset of fields we persist, tolerating the API's JSON-encoded list
fields (``outcomes``/``outcomePrices`` arrive as strings like
``'["Yes", "No"]'``).
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from types import TracebackType
from typing import Any, Self

import httpx
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

BASE_URL = "https://gamma-api.polymarket.com"
DEFAULT_TIMEOUT_SECONDS = 30.0
RATE_LIMIT_DELAY = 0.3
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 1.5
DEFAULT_PAGE = 100


def _as_list(value: Any) -> list[Any]:
    """Gamma encodes list fields as JSON strings; normalize to a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        return parsed if isinstance(parsed, list) else [parsed]
    return [value]


class PolymarketMarketModel(BaseModel):
    """Subset of a Gamma ``/markets`` object."""

    model_config = ConfigDict(extra="ignore")

    id: str | int | None = None
    conditionId: str | None = None
    slug: str = ""
    question: str = ""
    description: str = ""
    startDate: str | None = None
    endDate: str | None = None
    closed: bool = False
    outcomes: str | list[str] | None = None
    outcomePrices: str | list[str] | None = None

    def resolved_condition_id(self) -> str:
        return self.conditionId or self.slug or str(self.id or "")

    def resolved_outcome(self) -> str:
        """Best-effort winning outcome for a closed market.

        Pairs ``outcomes`` with ``outcomePrices`` and returns the label of
        the entry priced closest to 1.0. Empty string when undetermined.
        """
        if not self.closed:
            return ""
        labels = [str(x) for x in _as_list(self.outcomes)]
        prices = []
        for p in _as_list(self.outcomePrices):
            try:
                prices.append(float(p))
            except (TypeError, ValueError):
                prices.append(0.0)
        if not labels or len(labels) != len(prices):
            return ""
        winner = max(range(len(prices)), key=lambda i: prices[i])
        return labels[winner].lower()


class PolymarketClient:
    """Async client over the Polymarket Gamma REST API."""

    def __init__(
        self, base_url: str = BASE_URL, *, rate_limit_delay: float = RATE_LIMIT_DELAY
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._rate_limit_delay = rate_limit_delay
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> Self:
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=DEFAULT_TIMEOUT_SECONDS,
            headers={"Accept": "application/json"},
        )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _get(self, path: str, params: dict[str, Any]) -> Any:
        if self._client is None:
            raise RuntimeError("PolymarketClient must be used as an async context manager.")
        for attempt in range(MAX_RETRIES):
            await asyncio.sleep(self._rate_limit_delay)
            try:
                resp = await self._client.get(path, params=params)
            except httpx.RequestError:
                if attempt == MAX_RETRIES - 1:
                    raise
                await asyncio.sleep(RETRY_BACKOFF_BASE**attempt)
                continue
            if resp.status_code == 429:
                await asyncio.sleep(RETRY_BACKOFF_BASE ** (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json()
        return []

    async def paginate_markets(
        self,
        params: dict[str, Any] | None = None,
        *,
        page_size: int = DEFAULT_PAGE,
    ) -> AsyncIterator[list[PolymarketMarketModel]]:
        """Yield successive pages of parsed markets via limit/offset."""
        offset = 0
        base = dict(params or {})
        while True:
            try:
                page = await self._get(
                    "/markets", {**base, "limit": page_size, "offset": offset}
                )
            except httpx.HTTPStatusError as e:
                # Gamma caps how deep you can page and returns 4xx past the
                # limit; treat that as the end of the result set.
                if e.response.status_code in (400, 422):
                    logger.info(
                        "Polymarket pagination ended at offset %d (HTTP %d)",
                        offset,
                        e.response.status_code,
                    )
                    break
                raise
            items = page if isinstance(page, list) else page.get("data", [])
            if not items:
                break
            models: list[PolymarketMarketModel] = []
            for raw in items:
                try:
                    models.append(PolymarketMarketModel.model_validate(raw))
                except Exception as exc:
                    logger.debug("Skipping unparseable Polymarket row: %s", exc)
            if models:
                yield models
            if len(items) < page_size:
                break
            offset += page_size


__all__ = ["BASE_URL", "PolymarketClient", "PolymarketMarketModel"]
