"""Deterministic preprocessing for synthetic (and later real) original posts.

Redaction is one-way. Original hashes are retained; placeholders are not a
reversible map. Comments are never submitted to the model.
"""

from __future__ import annotations

import re
from typing import Any

from full_resource_sentiment_v1.adapter import extract_post_body
from full_resource_sentiment_v1.hashing import sha256_text
from full_resource_sentiment_v1.relevance_audit import EMAIL_RE, HANDLE_RE, PHONE_RE, URL_RE
from full_resource_sentiment_v1.relevance_classify.spec import (
    DEFAULT_MAX_INPUT_TOKENS,
    estimate_tokens_from_chars,
    user_message,
)

PROTECTED_OUTCOME_KEYS = frozenset(
    {
        "human_relevant",
        "gold_label",
        "true_label",
        "audit_label",
        "reviewer_label",
        "protected_outcome",
        "protected_label",
        "reserved_label",
        "reserved_outcome",
        "fresh_label",
        "fresh_outcome",
        "evaluation_outcome",
        "evaluation_label",
        "s2_target",
        "s2_in_scope",
        "joint_label",
        "frozen_label",
    }
)

FORBIDDEN_PATH_PARTS = (
    "reserved_test",
    "fresh_test_20261007",
    "relevance_audit_final",
    "broad_release_train",
    "accepted_workstream_i",
)
SOURCE_PATH_KEYS = frozenset(
    {
        "source_path",
        "source",
        "path",
        "capture_path",
        "file",
        "filename",
        "input_path",
        "data_path",
        "origin_path",
        "archive_path",
    }
)

# Whole-field only. A phrase inside otherwise original text is not a removal notice.
REMOVAL_FIELD_RE = re.compile(
    r"""
    ^
    (
        \[(removed|deleted)\]
        | \[removed\ by\ moderator\]
        | removed
        | deleted
        | removed\ by\ reddit\.?
        | (sorry,?\s+)?this\ post\ (was|has\ been)\ (deleted|removed)
          (\ by\ reddit('s\ [^.]+)?)?\.?
        | this\ content\ (was|has\ been)\ (deleted|removed)\.?
    )
    $
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Heuristic flag only. Does not skip scoring and is not a human judgment.
REMOVAL_HEURISTIC_RE = re.compile(
    r"\[\s*(removed|deleted)(?:\s+by\s+[^\]]+)?\s*\]"
    r"|removed by reddit"
    r"|this post (was|has been) (deleted|removed)"
    r"|this content (was|has been) (deleted|removed)",
    re.I,
)
SLASH_U_HANDLE_RE = re.compile(r"(?<!\w)/u/[A-Za-z0-9_-]+")
REDDIT_USER_PATH_RE = re.compile(r"(?i)\b(?:www\.)?reddit\.com/(?:user|u)/[A-Za-z0-9_-]+")
RESIDUAL_EMAIL_RE = EMAIL_RE
RESIDUAL_PHONE_RE = PHONE_RE


class ProtectedOutcomeSuppliedError(ValueError):
    """Raised when a protected evaluation outcome field is present on an input."""


def record_id(record: dict[str, Any]) -> str:
    for key in ("synthetic_id", "id", "post_id", "reddit_id", "redditId"):
        value = record.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    raise ValueError("record has no synthetic_id/reddit_id")


def reject_protected_outcomes(record: dict[str, Any]) -> None:
    hits = _forbidden_keys(record)
    path_hits = _forbidden_paths(record)
    if hits or path_hits:
        raise ProtectedOutcomeSuppliedError(
            "protected outcome fields/paths are forbidden in classifier inputs: "
            + ", ".join(sorted(hits | path_hits))
        )


def _forbidden_keys(obj: Any, *, found: set[str] | None = None) -> set[str]:
    found = set() if found is None else found
    if isinstance(obj, dict):
        for key, value in obj.items():
            lowered = str(key).lower()
            if lowered in PROTECTED_OUTCOME_KEYS:
                found.add(str(key))
            _forbidden_keys(value, found=found)
    elif isinstance(obj, list):
        for item in obj:
            _forbidden_keys(item, found=found)
    return found


def _forbidden_paths(record: dict[str, Any]) -> set[str]:
    """Scan source-metadata path fields only, not title/body prose."""
    found: set[str] = set()
    for key, value in record.items():
        if str(key).lower() not in SOURCE_PATH_KEYS:
            continue
        if not isinstance(value, str):
            continue
        lowered = value.replace("\\", "/").lower()
        for part in FORBIDDEN_PATH_PARTS:
            if part in lowered:
                found.add(part)
    return found


def extract_title_body(record: dict[str, Any]) -> tuple[str, str, str, bool]:
    title = "" if record.get("title") is None else str(record.get("title") or "")
    adapter_fields = ("author_description", "post_body", "selftext", "description")
    if any(field in record for field in adapter_fields):
        body, field, present = extract_post_body(record)
        return title, body, field, present
    if "body" in record:
        body = "" if record.get("body") is None else str(record.get("body") or "")
        return title, body, "body", True
    return title, "", "absent", False


def field_is_whole_removal_notice(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    return REMOVAL_FIELD_RE.fullmatch(stripped) is not None


def field_is_empty(text: str) -> bool:
    return not (text or "").strip()


def removal_notice_heuristic(title: str, body: str) -> bool:
    """Flag likely removal language. Never by itself assigns unassessable."""
    return bool(REMOVAL_HEURISTIC_RE.search(title or "") or REMOVAL_HEURISTIC_RE.search(body or ""))


def deterministic_unassessable(title: str, body: str) -> str | None:
    """Return a basis when no recoverable original title/body remains.

    An intact title with a removed/empty body stays assessable.
    """
    title_empty = field_is_empty(title)
    body_empty = field_is_empty(body)
    title_removed = field_is_whole_removal_notice(title)
    body_removed = field_is_whole_removal_notice(body)
    title_gone = title_empty or title_removed
    body_gone = body_empty or body_removed
    if title_gone and body_gone:
        if title_removed or body_removed:
            return "deterministic_removal_notice"
        return "deterministic_empty"
    return None


def redact_one_way(text: str) -> tuple[str, dict[str, int]]:
    """Replace PII with placeholders. This is not reversible; no replacement map is kept."""
    counts = {
        "email": 0,
        "phone": 0,
        "url": 0,
        "handle": 0,
        "slash_u_handle": 0,
        "reddit_user_path": 0,
    }

    def sub_email(match: re.Match[str]) -> str:
        counts["email"] += 1
        return "[REDACTED_EMAIL]"

    def sub_phone(match: re.Match[str]) -> str:
        counts["phone"] += 1
        return "[REDACTED_PHONE]"

    def sub_url(match: re.Match[str]) -> str:
        counts["url"] += 1
        return "[REDACTED_URL]"

    def sub_handle(match: re.Match[str]) -> str:
        counts["handle"] += 1
        return "[REDACTED_HANDLE]"

    def sub_slash_u(match: re.Match[str]) -> str:
        counts["slash_u_handle"] += 1
        return "[REDACTED_HANDLE]"

    def sub_user_path(match: re.Match[str]) -> str:
        counts["reddit_user_path"] += 1
        return "[REDACTED_URL]"

    redacted = EMAIL_RE.sub(sub_email, text or "")
    redacted = URL_RE.sub(sub_url, redacted)
    redacted = REDDIT_USER_PATH_RE.sub(sub_user_path, redacted)
    redacted = PHONE_RE.sub(sub_phone, redacted)
    redacted = HANDLE_RE.sub(sub_handle, redacted)
    redacted = SLASH_U_HANDLE_RE.sub(sub_slash_u, redacted)
    return redacted, counts


def residual_pii_types(text: str) -> list[str]:
    found: list[str] = []
    if RESIDUAL_EMAIL_RE.search(text or ""):
        found.append("email")
    if RESIDUAL_PHONE_RE.search(text or ""):
        found.append("phone")
    if HANDLE_RE.search(text or "") or SLASH_U_HANDLE_RE.search(text or ""):
        found.append("handle")
    if URL_RE.search(text or "") or REDDIT_USER_PATH_RE.search(text or ""):
        found.append("url")
    return found


def redaction_types(counts: dict[str, int]) -> list[str]:
    names = []
    if counts.get("email"):
        names.append("email")
    if counts.get("phone"):
        names.append("phone")
    if counts.get("url") or counts.get("reddit_user_path"):
        names.append("url")
    if counts.get("handle") or counts.get("slash_u_handle"):
        names.append("handle")
    return names


def evidence_is_substring(evidence: str | None, submitted: str) -> bool:
    if evidence is None:
        return False
    if evidence == "":
        return False
    return evidence in submitted


def evidence_in_title_or_body(evidence: str | None, title: str, body: str) -> bool:
    """Positive evidence must quote redacted title or body, not TITLE:/BODY: wrappers."""
    if not isinstance(evidence, str) or not evidence.strip():
        return False
    if evidence in (title or "") or evidence in (body or ""):
        return True
    return False


def preprocess_record(
    record: dict[str, Any],
    *,
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
) -> dict[str, Any]:
    reject_protected_outcomes(record)
    post_id = record_id(record)
    title, body, body_field, body_field_present = extract_title_body(record)
    comments = record.get("comments")
    comments_present = isinstance(comments, list) and len(comments) > 0
    unassessable_basis = deterministic_unassessable(title, body)

    title_redacted, title_counts = redact_one_way(title)
    body_redacted, body_counts = redact_one_way(body)
    combined_counts = {
        key: int(title_counts.get(key, 0)) + int(body_counts.get(key, 0)) for key in title_counts
    }
    submitted = user_message(title_redacted, body_redacted)
    submitted_tokens = estimate_tokens_from_chars(len(submitted))
    truncated = submitted_tokens > max_input_tokens
    residual = residual_pii_types(title_redacted) + residual_pii_types(body_redacted)

    return {
        "synthetic_id": post_id,
        "title_original": title,
        "body_original": body,
        "title_submitted": title_redacted,
        "body_submitted": body_redacted,
        "submitted_text": submitted,
        "body_field": body_field,
        "body_field_present": body_field_present,
        "comments_ignored": True,
        "comments_present": comments_present,
        "unassessable_basis": unassessable_basis,
        "removal_notice_heuristic": removal_notice_heuristic(title, body),
        "title_sha256": sha256_text(title),
        "body_sha256": sha256_text(body),
        "model_input_sha256": sha256_text(submitted),
        "redacted": any(combined_counts.values()),
        "redaction_counts": combined_counts,
        "redaction_types": redaction_types(combined_counts),
        "residual_pii_suspected": bool(residual),
        "residual_pii_types": sorted(set(residual)),
        "title_chars": len(title),
        "body_chars": len(body),
        "submitted_chars": len(submitted),
        "input_tokens_est": submitted_tokens,
        "truncated": truncated,
        "max_input_tokens": max_input_tokens,
        "protected_membership": bool(record.get("protected_membership")),
        "case_code": None if record.get("case_code") is None else str(record.get("case_code")),
        "reddit_id": None if record.get("reddit_id") is None else str(record.get("reddit_id")),
        "stratum": None if record.get("stratum") is None else str(record.get("stratum")),
        "created_month": None if record.get("created_month") is None else str(record.get("created_month")),
        "count_gap": bool(record.get("count_gap")),
        "capture_id": None if record.get("capture_id") is None else str(record.get("capture_id")),
        "classifier_run_id": None if record.get("classifier_run_id") is None else str(record.get("classifier_run_id")),
    }
