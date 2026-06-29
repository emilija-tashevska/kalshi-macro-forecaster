"""Split documents into overlapping passages for embedding/retrieval.

We chunk on paragraph boundaries where possible, packing paragraphs up to
a character budget with a small overlap so a passage retains context. The
budget is in characters (a cheap proxy for tokens) to avoid a tokenizer
dependency here.
"""

from __future__ import annotations

import re

DEFAULT_CHUNK_CHARS = 1500
DEFAULT_OVERLAP_CHARS = 200
MIN_CHUNK_CHARS = 80

_PARA_SPLIT = re.compile(r"\n\s*\n+")


def chunk_text(
    text: str,
    *,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    min_chunk_chars: int = MIN_CHUNK_CHARS,
) -> list[str]:
    """Return overlapping passages of roughly ``chunk_chars`` each.

    Paragraphs are packed greedily; an over-long paragraph is hard-split.
    Consecutive chunks share ``overlap_chars`` of tail text for continuity.
    Chunks shorter than ``min_chunk_chars`` are dropped (boilerplate).
    """
    text = (text or "").strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in _PARA_SPLIT.split(text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        remainder = para
        # Hard-split paragraphs that exceed the budget on their own.
        while len(remainder) > chunk_chars:
            head, remainder = remainder[:chunk_chars], remainder[chunk_chars:]
            if current:
                chunks.append(current)
                current = ""
            chunks.append(head)
        if not remainder:
            continue
        if not current:
            current = remainder
        elif len(current) + 2 + len(remainder) <= chunk_chars:
            current = f"{current}\n\n{remainder}"
        else:
            chunks.append(current)
            current = remainder
    if current:
        chunks.append(current)

    # Add tail overlap between adjacent chunks for context continuity.
    if overlap_chars > 0 and len(chunks) > 1:
        overlapped: list[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-overlap_chars:]
            overlapped.append(f"{tail}\n\n{chunks[i]}")
        chunks = overlapped

    return [c for c in chunks if len(c) >= min_chunk_chars]


__all__ = ["chunk_text"]
