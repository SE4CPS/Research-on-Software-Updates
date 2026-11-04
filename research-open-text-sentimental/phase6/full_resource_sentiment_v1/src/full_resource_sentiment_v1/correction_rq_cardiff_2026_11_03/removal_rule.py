"""Whole-field removal/unassessable detector (extends established preprocess rule).

Does NOT flag ordinary prose that merely mentions removal/deletion.
"""

from __future__ import annotations

import re

from full_resource_sentiment_v1.relevance_classify.preprocess import (
    deterministic_unassessable as _base_deterministic_unassessable,
    field_is_empty,
    field_is_whole_removal_notice as _base_field_is_whole_removal_notice,
)

# Spaced Reddit whole-field notices not covered by the original REMOVAL_FIELD_RE
SPACED_REDDIT_TITLE_RE = re.compile(r"^\[\s*Removed\s+by\s+Reddit\s*\]$", re.IGNORECASE)
SPACED_REDDIT_POLICY_BODY_RE = re.compile(
    r"^\[\s*Removed\s+by\s+Reddit\s+on\s+account\s+of\s+violating[\s\S]*\]$",
    re.IGNORECASE,
)
SPACED_BRACKET_REMOVED_RE = re.compile(r"^\[\s*(removed|deleted)\s*\]$", re.IGNORECASE)


def field_is_whole_removal_notice(text: str) -> bool:
    if _base_field_is_whole_removal_notice(text):
        return True
    stripped = (text or "").strip()
    if not stripped:
        return False
    if SPACED_REDDIT_TITLE_RE.fullmatch(stripped):
        return True
    if SPACED_REDDIT_POLICY_BODY_RE.fullmatch(stripped):
        return True
    if SPACED_BRACKET_REMOVED_RE.fullmatch(stripped):
        return True
    return False


def deterministic_unassessable(title: str, body: str) -> str | None:
    """No recoverable original title/body remains (whole-field empty/removal)."""
    base = _base_deterministic_unassessable(title, body)
    if base:
        return base
    title_empty = field_is_empty(title)
    body_empty = field_is_empty(body)
    title_removed = field_is_whole_removal_notice(title)
    body_removed = field_is_whole_removal_notice(body)
    title_gone = title_empty or title_removed
    body_gone = body_empty or body_removed
    if title_gone and body_gone:
        if title_removed or body_removed:
            return "deterministic_removal_notice_extended"
        return "deterministic_empty"
    return None
