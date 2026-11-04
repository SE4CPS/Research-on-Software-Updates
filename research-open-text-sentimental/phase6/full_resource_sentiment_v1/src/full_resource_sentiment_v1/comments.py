"""Normalized comments: original order, chronological order, roles, missingness."""

from __future__ import annotations

from typing import Any

from .hashing import sha256_text
from .roles import infer_author_role
from .timestamps import parse_created_utc

COMMENTS_ABSENT = "absent"
COMMENTS_PRESENT_EMPTY = "present_empty"
COMMENTS_PRESENT_NONEMPTY = "present_nonempty"
COMMENTS_MALFORMED = "malformed"


def comments_status(raw_post: dict[str, Any]) -> str:
    if "comments" not in raw_post:
        return COMMENTS_ABSENT
    value = raw_post.get("comments")
    if value is None:
        return COMMENTS_ABSENT
    if not isinstance(value, list):
        return COMMENTS_MALFORMED
    if len(value) == 0:
        return COMMENTS_PRESENT_EMPTY
    return COMMENTS_PRESENT_NONEMPTY


def extract_comment_text(comment: dict[str, Any]) -> tuple[str, str]:
    if "body" in comment:
        return (comment.get("body") or ""), "body"
    if "text" in comment:
        return (comment.get("text") or ""), "text"
    return "", "absent"


def normalize_comments(
    raw_post: dict[str, Any],
    *,
    include_text: bool = False,
) -> dict[str, Any]:
    status = comments_status(raw_post)
    post_author = raw_post.get("author")
    raw_comments = raw_post.get("comments") if "comments" in raw_post else None
    records: list[dict[str, Any]] = []
    if isinstance(raw_comments, list):
        for original_index, comment in enumerate(raw_comments):
            if not isinstance(comment, dict):
                records.append(
                    {
                        "original_index": original_index,
                        "malformed": True,
                        "body_nonempty": False,
                        "timestamp_valid": False,
                        "is_author": False,
                        "author_role_basis": "malformed_comment",
                    }
                )
                continue
            text, text_field = extract_comment_text(comment)
            text_stripped = text.strip()
            parsed_ts = parse_created_utc(comment.get("created_utc"))
            role = infer_author_role(post_author, comment.get("author"), comment.get("is_submitter"))
            record = {
                "original_index": original_index,
                "malformed": False,
                "text_field": text_field,
                "body_len": len(text),
                "body_nonempty": bool(text_stripped),
                "body_sha256": sha256_text(text) if text_stripped or text else sha256_text(text),
                "timestamp_raw": parsed_ts["raw"],
                "timestamp_valid": parsed_ts["valid"],
                "timestamp_parsed_utc": parsed_ts["parsed_utc"],
                "timestamp_epoch_seconds": parsed_ts["epoch_seconds"],
                "timestamp_reason": parsed_ts["reason"],
                **role,
            }
            if include_text:
                record["text"] = text
            records.append(record)

    chronological = sort_for_trajectory(records)
    chrono_index = {id(item): index for index, item in enumerate(chronological)}
    for record in records:
        record["chronological_index"] = chrono_index.get(id(record))

    nonempty = [c for c in chronological if c.get("body_nonempty")]
    author_replies = [c for c in nonempty if c.get("is_author")]
    community = [c for c in nonempty if not c.get("is_author")]
    valid_ts = [c for c in nonempty if c.get("timestamp_valid")]
    return {
        "comments_status": status,
        "n_stored": len(records) if status not in {COMMENTS_ABSENT, COMMENTS_MALFORMED} else 0,
        "n_nonempty": len(nonempty),
        "n_empty_body": sum(1 for c in records if not c.get("body_nonempty") and not c.get("malformed")),
        "n_author_replies_nonempty": len(author_replies),
        "n_community_replies_nonempty": len(community),
        "n_valid_timestamps_nonempty": len(valid_ts),
        "n_invalid_or_missing_timestamps": sum(1 for c in records if not c.get("timestamp_valid")),
        "chronological_original_indices": [c["original_index"] for c in chronological],
        "comments_original_order": records,
        "comments_chronological": chronological,
    }


def sort_for_trajectory(comments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Valid timestamps ascending; original_index is the only tie-breaker.

    Invalid or missing timestamps sort after valid ones, still by original index.
    Index order is not elapsed time.
    """

    def key(comment: dict[str, Any]) -> tuple[int, float, int]:
        valid = bool(comment.get("timestamp_valid"))
        epoch = comment.get("timestamp_epoch_seconds")
        original = int(comment.get("original_index") or 0)
        return (0 if valid else 1, float(epoch) if valid and epoch is not None else 0.0, original)

    return sorted(comments, key=key)


def original_string_sort(raw_comments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """trajectory/build.py sort: str(created_utc or '')."""
    return sorted(raw_comments, key=lambda comment: str(comment.get("created_utc") or ""))


PUBLIC_COMMENT_FIELDS = (
    "original_index",
    "chronological_index",
    "malformed",
    "text_field",
    "body_len",
    "body_nonempty",
    "body_sha256",
    "timestamp_raw",
    "timestamp_valid",
    "timestamp_parsed_utc",
    "timestamp_reason",
    "is_author",
    "author_role_basis",
    "is_submitter_parsed",
    "name_match",
    "author_name_present",
    "role_conflict_is_submitter_vs_name",
    "cohort_split",
    "cohort_is_author_field",
    "cohort_role_conflict",
)


def public_comment(record: dict[str, Any]) -> dict[str, Any]:
    """Drop usernames and raw text; keep hashes, roles, timestamps, and positions."""
    return {key: record.get(key) for key in PUBLIC_COMMENT_FIELDS if key in record}


def normalize_pre_split_role_comments(
    raw_post: dict[str, Any],
    *,
    include_text: bool = False,
) -> dict[str, Any]:
    """Normalize cohort posts that already split author_replies / community_comments.

    Timestamps are parsed when present and flagged when missing. Roles come from the
    stored split, not from a reconstructed comments array. Index order is not elapsed time.
    """
    author_key_present = "author_replies" in raw_post
    community_key_present = "community_comments" in raw_post
    if not author_key_present and not community_key_present:
        return {
            "comments_status": COMMENTS_ABSENT,
            "comment_layout": "pre_split_author_community",
            "n_stored": 0,
            "n_nonempty": 0,
            "n_empty_body": 0,
            "n_author_replies_nonempty": 0,
            "n_community_replies_nonempty": 0,
            "n_valid_timestamps_nonempty": 0,
            "n_invalid_or_missing_timestamps": 0,
            "chronological_original_indices": [],
            "comments_original_order": [],
            "comments_chronological": [],
        }

    author_list = raw_post.get("author_replies") if author_key_present else []
    community_list = raw_post.get("community_comments") if community_key_present else []
    if author_list is None:
        author_list = []
    if community_list is None:
        community_list = []
    if not isinstance(author_list, list) or not isinstance(community_list, list):
        return {
            "comments_status": COMMENTS_MALFORMED,
            "comment_layout": "pre_split_author_community",
            "n_stored": 0,
            "n_nonempty": 0,
            "n_empty_body": 0,
            "n_author_replies_nonempty": 0,
            "n_community_replies_nonempty": 0,
            "n_valid_timestamps_nonempty": 0,
            "n_invalid_or_missing_timestamps": 0,
            "chronological_original_indices": [],
            "comments_original_order": [],
            "comments_chronological": [],
        }

    records: list[dict[str, Any]] = []
    combined = [("author_replies", True, comment) for comment in author_list] + [
        ("community_comments", False, comment) for comment in community_list
    ]
    if not combined:
        status = COMMENTS_PRESENT_EMPTY
    else:
        status = COMMENTS_PRESENT_NONEMPTY

    for original_index, (split, is_author_split, comment) in enumerate(combined):
        if not isinstance(comment, dict):
            records.append(
                {
                    "original_index": original_index,
                    "malformed": True,
                    "body_nonempty": False,
                    "timestamp_valid": False,
                    "is_author": is_author_split,
                    "author_role_basis": "malformed_comment",
                    "cohort_split": split,
                }
            )
            continue
        text, text_field = extract_comment_text(comment)
        text_stripped = text.strip()
        parsed_ts = parse_created_utc(comment.get("created_utc"))
        stored_is_author = comment.get("is_author")
        conflict = stored_is_author is not None and bool(stored_is_author) != bool(is_author_split)
        record = {
            "original_index": original_index,
            "malformed": False,
            "text_field": text_field,
            "body_len": len(text),
            "body_nonempty": bool(text_stripped),
            "body_sha256": sha256_text(text),
            "timestamp_raw": parsed_ts["raw"],
            "timestamp_valid": parsed_ts["valid"],
            "timestamp_parsed_utc": parsed_ts["parsed_utc"],
            "timestamp_epoch_seconds": parsed_ts["epoch_seconds"],
            "timestamp_reason": parsed_ts["reason"],
            "is_author": is_author_split,
            "author_role_basis": "cohort_pre_split_author" if is_author_split else "cohort_pre_split_community",
            "author_name_present": bool(str(comment.get("author") or "").strip()),
            "cohort_split": split,
            "cohort_is_author_field": stored_is_author,
            "cohort_role_conflict": conflict,
        }
        if include_text:
            record["text"] = text
        records.append(record)

    chronological = sort_for_trajectory(records)
    chrono_index = {id(item): index for index, item in enumerate(chronological)}
    for record in records:
        record["chronological_index"] = chrono_index.get(id(record))
    nonempty = [c for c in chronological if c.get("body_nonempty")]
    author_replies = [c for c in nonempty if c.get("is_author")]
    community = [c for c in nonempty if not c.get("is_author")]
    return {
        "comments_status": status,
        "comment_layout": "pre_split_author_community",
        "n_stored": len(records),
        "n_nonempty": len(nonempty),
        "n_empty_body": sum(1 for c in records if not c.get("body_nonempty") and not c.get("malformed")),
        "n_author_replies_nonempty": len(author_replies),
        "n_community_replies_nonempty": len(community),
        "n_valid_timestamps_nonempty": sum(1 for c in nonempty if c.get("timestamp_valid")),
        "n_invalid_or_missing_timestamps": sum(1 for c in records if not c.get("timestamp_valid")),
        "chronological_original_indices": [c["original_index"] for c in chronological],
        "comments_original_order": records,
        "comments_chronological": chronological,
    }
