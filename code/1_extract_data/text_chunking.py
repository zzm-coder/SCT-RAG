"""Deterministic Markdown-aware text chunking for relation extraction."""

from __future__ import annotations

import re


def split_text_into_chunks(text: str, max_chars: int = 1500, overlap: int = 100) -> list[str]:
    """Split text without dropping content, preferring paragraph/sentence boundaries.

    ``overlap`` is taken from the end of the previous chunk.  It is bounded so
    that every iteration still advances, which makes the function safe for very
    small test chunk sizes as well as production settings.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must satisfy 0 <= overlap < max_chars")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    length = len(normalized)
    min_boundary = max(1, max_chars // 2)
    while start < length:
        hard_end = min(start + max_chars, length)
        end = hard_end
        if hard_end < length:
            window = normalized[start:hard_end]
            candidates = [m.end() for m in re.finditer(r"\n\s*\n|\n|[\u3002！？；]\s*", window)]
            usable = [pos for pos in candidates if pos >= min_boundary]
            if usable:
                end = start + usable[-1]
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        next_start = max(start + 1, end - overlap)
        start = next_start
    return chunks
