"""Federal Reserve text source: HTTP client + pure HTML parsers.

Phase 1.4 ingests the Fed's policy-communication corpus. We split the
concern in two so the brittle part (HTML structure) is unit-testable
without a network:

- ``FedTextClient`` — a thin async httpx wrapper that fetches a URL and
  returns its HTML, or ``None`` on a 404. Returning ``None`` (rather
  than raising) lets the orchestrator *probe* deterministic URLs built
  from the FOMC calendar without knowing in advance which exist yet
  (e.g. minutes for the most recent meeting aren't published for ~3
  weeks).

- ``html_to_text`` / ``build_document`` — pure functions that turn raw
  HTML into a clean ``TextDocument``. These are what the tests exercise
  against saved fixtures.

The Fed serves stable, predictable URLs for statements, minutes, and the
Beige Book, so the orchestrator generates candidate URLs rather than
crawling index pages (which change layout often).
"""

from __future__ import annotations

import asyncio
import logging
import re
from types import TracebackType
from typing import Self

import httpx
from bs4 import BeautifulSoup

from kalshi_train.db.ingest import DateLike, TextDocument

logger = logging.getLogger(__name__)

BASE_URL = "https://www.federalreserve.gov"
DEFAULT_TIMEOUT_SECONDS = 30.0
RATE_LIMIT_DELAY = 0.5
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 1.5

# The Fed returns 403 for clients without a browser-like User-Agent.
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) kalshi-train-research/0.1"
)

# Content blocks the Fed wraps article bodies in, most-specific first.
_CONTENT_SELECTORS = (
    "#article",
    "#content",
    "div.col-xs-12.col-sm-8.col-md-8",
    "div.col-xs-12.col-sm-8.col-md-9",
    "main",
)

_DROP_TAGS = ("script", "style", "noscript", "nav", "header", "footer", "form")


class FedTextClient:
    """Async fetcher for federalreserve.gov HTML pages."""

    def __init__(
        self, base_url: str = BASE_URL, *, rate_limit_delay: float = RATE_LIMIT_DELAY
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._rate_limit_delay = rate_limit_delay
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> Self:
        self._client = httpx.AsyncClient(
            timeout=DEFAULT_TIMEOUT_SECONDS,
            headers={"User-Agent": _USER_AGENT, "Accept": "text/html"},
            follow_redirects=True,
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

    async def fetch(self, url: str) -> str | None:
        """Return page HTML, or ``None`` if the page does not exist (404).

        Raises on other persistent HTTP / network errors so genuine
        problems aren't silently swallowed.
        """
        if self._client is None:
            raise RuntimeError("FedTextClient must be used as an async context manager.")
        full = url if url.startswith("http") else f"{self._base_url}{url}"

        for attempt in range(MAX_RETRIES):
            await asyncio.sleep(self._rate_limit_delay)
            try:
                resp = await self._client.get(full)
            except httpx.RequestError as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                logger.warning("Fed request error on %s: %s (retry)", full, e)
                await asyncio.sleep(RETRY_BACKOFF_BASE**attempt)
                continue
            if resp.status_code == 404:
                return None
            if resp.status_code == 429:
                await asyncio.sleep(RETRY_BACKOFF_BASE ** (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.text
        return None


# ── Pure parsing helpers ───────────────────────────────────────────────


def _collapse_blank_lines(text: str) -> str:
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def html_to_text(html: str) -> str:
    """Extract clean, readable body text from a Fed HTML page.

    Strips chrome (scripts, nav, header/footer), prefers the article
    content block, and falls back to the whole ``<body>`` when no known
    wrapper is present.
    """
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(list(_DROP_TAGS)):
        tag.decompose()

    container = None
    for selector in _CONTENT_SELECTORS:
        container = soup.select_one(selector)
        if container is not None and container.get_text(strip=True):
            break
    if container is None:
        container = soup.body or soup

    return _collapse_blank_lines(container.get_text("\n", strip=True))


def extract_title(html: str, default: str = "") -> str:
    """Best-effort document title from common Fed markup."""
    soup = BeautifulSoup(html, "lxml")
    for selector in ("h3.title", "h2.title", "h1", "title"):
        el = soup.select_one(selector)
        if el is not None:
            text = el.get_text(" ", strip=True)
            if text:
                return text
    return default


def build_document(
    html: str,
    *,
    source: str,
    document_type: str,
    url: str,
    published_date: DateLike,
    effective_date: DateLike | None = None,
    title: str | None = None,
    min_body_chars: int = 200,
) -> TextDocument | None:
    """Turn raw HTML into a ``TextDocument`` (or ``None`` if it's too thin).

    A too-short body usually means we fetched a placeholder / error page
    rather than a real document, so we reject it instead of polluting the
    corpus.
    """
    body = html_to_text(html)
    if len(body) < min_body_chars:
        logger.debug("Skipping %s: body only %d chars", url, len(body))
        return None
    resolved_title = title or extract_title(html, default=document_type)
    return TextDocument(
        source=source,
        document_type=document_type,
        title=resolved_title[:300],
        published_date=published_date,
        body=body,
        url=url,
        effective_date=effective_date,
    )


__all__ = [
    "BASE_URL",
    "FedTextClient",
    "build_document",
    "extract_title",
    "html_to_text",
]
