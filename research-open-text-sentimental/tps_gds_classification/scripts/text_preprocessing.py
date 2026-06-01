"""
Shared text preprocessing for TPS vs GDS classification.

Used by the fetch script and can be reused by training / inference code
so cleaning stays consistent.
"""

from __future__ import annotations

import re
from typing import Any

URL_PATTERN = re.compile(
    r"https?://[^\s]+|www\.[^\s]+",
    re.IGNORECASE,
)
NON_ALNUM_SPACE = re.compile(r"[^a-z0-9\s]+")
WS = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, remove URLs, strip non-alphanumeric except spaces, collapse whitespace."""
    s = text.lower()
    s = URL_PATTERN.sub(" ", s)
    s = NON_ALNUM_SPACE.sub(" ", s)
    s = WS.sub(" ", s).strip()
    return s


def top_comments(comments: list[dict[str, Any]], limit: int) -> list[str]:
    if not comments:
        return []
    sorted_comments = sorted(
        comments,
        key=lambda c: (c.get("score") is None, -(c.get("score") or 0)),
    )
    bodies: list[str] = []
    for c in sorted_comments[:limit]:
        b = (c.get("body") or "").strip()
        if b:
            bodies.append(b)
    return bodies


def combine_text(title: str, body: str, comment_bodies: list[str]) -> str:
    parts = [title.strip(), body.strip(), *comment_bodies]
    return "\n".join(p for p in parts if p)
