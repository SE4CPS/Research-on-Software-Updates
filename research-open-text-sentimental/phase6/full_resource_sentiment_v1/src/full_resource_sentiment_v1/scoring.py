"""Score one ReleaseTrain post with package 0.2.1. Missing text is null, not 0.0.

Title, body, and combined original-post scores are obtained by calling
score_post with the unused field empty. Nonempty replies are scored through
to_package_post; empty bodies are skipped. Package text snippets are dropped.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from releasetrain_sentiment import score_post

from .adapter import extract_post_body, to_package_post
from .comments import comments_status, extract_comment_text
from .timestamps import parse_created_utc

PACKAGE_VERSION = "0.2.1"
CAPTURE_ID = "2026-10-22-main"


def is_reddit_url(url: Any) -> bool:
    host = urlparse(str(url or "")).hostname or ""
    return host == "reddit.com" or host.endswith(".reddit.com")


def role_label(is_author: bool, basis: str) -> str:
    if is_author:
        return "author"
    if basis == "blank_names_not_matched":
        return "unresolved"
    return "community"


def components_without_text(item: dict[str, Any]) -> dict[str, float | str]:
    return {
        "compound": float(item["compound"]),
        "pos": float(item["pos"]),
        "neu": float(item["neu"]),
        "neg": float(item["neg"]),
        "label": str(item["label"]),
    }


def score_package_fields(title: str, description: str) -> dict[str, float | str]:
    """Primary VADER implementation: releasetrain-sentiment 0.2.1 score_post."""
    scored = score_post({"title": title, "description": description, "comments": []})
    return components_without_text(scored[0])


def score_observed_text(text: str) -> dict[str, float | str] | None:
    if not str(text or "").strip():
        return None
    return score_package_fields(text, "")


def length_band(n_chars: int, *, kind: str) -> str:
    if kind == "title":
        if n_chars < 20:
            return "lt_20"
        if n_chars <= 50:
            return "from_20_to_50"
        return "gt_50"
    if n_chars < 100:
        return "lt_100"
    if n_chars <= 500:
        return "from_100_to_500"
    return "gt_500"


def date_band(created_month: str | None) -> str:
    month = created_month or "missing"
    if month == "2026-10":
        return "oct2026"
    if month == "missing":
        return "missing"
    return "older"


def duration_band(hours: float | None) -> str | None:
    if hours is None:
        return None
    if hours < 1:
        return "lt_1h"
    if hours <= 24:
        return "from_1_to_24h"
    return "gt_24h"


def _raw_comments(raw_post: dict[str, Any]) -> list[Any]:
    comments = raw_post.get("comments")
    return comments if isinstance(comments, list) else []


def score_raw_post(raw_post: dict[str, Any]) -> dict[str, Any]:
    """Score title, body, combined post, and nonempty comments. No usernames/text."""
    title = "" if raw_post.get("title") is None else str(raw_post.get("title") or "")
    body, body_field, body_field_present = extract_post_body(raw_post)
    title_nonempty = bool(title.strip())
    body_nonempty = bool(body.strip())
    title_scores = score_observed_text(title)
    body_scores = score_observed_text(body)
    if title_nonempty or body_nonempty:
        combined_scores = score_package_fields(title, body)
    else:
        combined_scores = None

    mapped = to_package_post(raw_post, include_empty_comment_bodies=False)
    comment_items = score_post(mapped)[1:]
    orig_indices = mapped["adapter_meta"]["chronological_original_indices"]
    role_bases = mapped["adapter_meta"]["comment_role_basis"]
    is_authors = mapped["adapter_meta"]["comment_is_author"]
    raw_comments = _raw_comments(raw_post)

    comment_rows: list[dict[str, Any]] = []
    skipped_empty: list[dict[str, Any]] = []
    for original_index, comment in enumerate(raw_comments):
        if not isinstance(comment, dict):
            continue
        text, _field = extract_comment_text(comment)
        if not str(text or "").strip():
            skipped_empty.append({"original_index": original_index})

    author_compounds: list[float] = []
    community_compounds: list[float] = []
    unresolved_compounds: list[float] = []
    timed: list[dict[str, Any]] = []

    for chrono_i, item in enumerate(comment_items):
        original_index = int(orig_indices[chrono_i])
        basis = str(role_bases[chrono_i])
        is_author = bool(is_authors[chrono_i])
        role = role_label(is_author, basis)
        raw_comment = raw_comments[original_index] if original_index < len(raw_comments) else {}
        parsed = parse_created_utc(raw_comment.get("created_utc") if isinstance(raw_comment, dict) else None)
        compound = float(item["compound"])
        row = {
            "original_index": original_index,
            "chronological_index": chrono_i,
            "timestamp_utc": parsed["parsed_utc"],
            "timestamp_valid": parsed["valid"],
            "timestamp_epoch_seconds": parsed["epoch_seconds"],
            "role": role,
            "author_role_basis": basis,
            "is_author": is_author,
            **components_without_text(item),
        }
        comment_rows.append(row)
        if is_author:
            author_compounds.append(compound)
        else:
            community_compounds.append(compound)
            if role == "unresolved":
                unresolved_compounds.append(compound)
        if parsed["valid"]:
            timed.append(row)

    post_created = parse_created_utc(raw_post.get("created_utc"))
    n_author = len(author_compounds)
    n_community = len(community_compounds)
    n_unresolved = len(unresolved_compounds)
    n_nonempty = len(comment_rows)
    n_timed = len(timed)
    distinct_times = {row["timestamp_utc"] for row in timed}

    first_timed = timed[0] if timed else None
    last_timed = timed[-1] if timed else None
    last_minus_first = None
    if first_timed is not None and last_timed is not None and n_timed >= 2:
        last_minus_first = float(last_timed["compound"] - first_timed["compound"])

    elapsed_hours = None
    if n_timed >= 2 and len(distinct_times) >= 2:
        elapsed_hours = (
            float(last_timed["timestamp_epoch_seconds"] - first_timed["timestamp_epoch_seconds"]) / 3600.0
        )

    post_to_first_hours = None
    post_to_first_eligible = bool(post_created["valid"] and first_timed is not None)
    if post_to_first_eligible:
        post_to_first_hours = (
            float(first_timed["timestamp_epoch_seconds"] - post_created["epoch_seconds"]) / 3600.0
        )

    body_minus_title = None
    if body_scores is not None and title_scores is not None:
        body_minus_title = float(body_scores["compound"] - title_scores["compound"])

    stored = raw_comments
    try:
        num_comments_reported = int(raw_post.get("num_comments"))
    except (TypeError, ValueError):
        num_comments_reported = None
    count_gap = num_comments_reported is not None and num_comments_reported > len(stored)

    return {
        "title": title_scores,
        "body": body_scores,
        "combined": combined_scores,
        "title_nonempty": title_nonempty,
        "body_nonempty": body_nonempty,
        "title_len": len(title),
        "body_len": len(body),
        "title_len_band": length_band(len(title), kind="title") if title_nonempty else None,
        "body_len_band": length_band(len(body), kind="body") if body_nonempty else None,
        "body_field": body_field,
        "body_field_present": body_field_present,
        "comments_status": comments_status(raw_post),
        "n_stored_comments": len(stored) if comments_status(raw_post) not in {"absent", "malformed"} else 0,
        "n_nonempty_comments": n_nonempty,
        "n_empty_comment_bodies": len(skipped_empty),
        "n_author_replies_nonempty": n_author,
        "n_community_replies_nonempty": n_community,
        "n_unresolved_replies_nonempty": n_unresolved,
        "n_timed_nonempty": n_timed,
        "n_distinct_reply_times": len(distinct_times),
        "author_mean": _mean_or_none(author_compounds),
        "community_mean": _mean_or_none(community_compounds),
        "unresolved_mean": _mean_or_none(unresolved_compounds),
        "author_shift": early_late_or_none(author_compounds),
        "community_shift": early_late_or_none(community_compounds),
        "order_eligible": n_timed >= 2,
        "elapsed_eligible": len(distinct_times) >= 2,
        "author_reply_eligible": n_author >= 1,
        "both_roles_eligible": n_author >= 1 and n_community >= 1,
        "first_reply_compound": None if first_timed is None else float(first_timed["compound"]),
        "last_reply_compound": None if last_timed is None else float(last_timed["compound"]),
        "last_minus_first": last_minus_first,
        "order_slope_normalized": normalized_position_slope(
            [row["compound"] for row in timed]
        ),
        "elapsed_hours": elapsed_hours,
        "elapsed_band": duration_band(elapsed_hours),
        "post_timestamp_valid": post_created["valid"],
        "post_created_utc": post_created["parsed_utc"],
        "post_to_first_reply_eligible": post_to_first_eligible,
        "post_to_first_reply_hours": post_to_first_hours,
        "body_minus_title": body_minus_title,
        "count_gap": bool(count_gap),
        "num_comments_reported": num_comments_reported,
        "comment_rows": comment_rows,
        "skipped_empty_comment_bodies": skipped_empty,
        "n_comments_scored": n_nonempty,
        "n_comments_skipped_empty": len(skipped_empty),
        "used_api_sentiment_score": False,
        "used_api_positive_score": False,
        "package_version": PACKAGE_VERSION,
        "adapter": "to_package_post",
    }


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def early_late_or_none(values: list[float]) -> float | None:
    """Package early/late helper; null when fewer than four role comments."""
    if len(values) < 4:
        return None
    mid = len(values) // 2
    early = sum(values[:mid]) / mid
    late = sum(values[mid:]) / (len(values) - mid)
    return float(late - early)


def normalized_position_slope(values: list[float]) -> float | None:
    """OLS slope of compound vs position in [0, 1]. Requires >=3 timed replies."""
    n = len(values)
    if n < 3:
        return None
    xs = [i / (n - 1) for i in range(n)]
    x_mean = sum(xs) / n
    y_mean = sum(values) / n
    num = sum((xs[i] - x_mean) * (values[i] - y_mean) for i in range(n))
    den = sum((xs[i] - x_mean) ** 2 for i in range(n))
    if den == 0:
        return None
    return float(num / den)
