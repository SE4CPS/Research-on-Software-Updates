"""Harmonized estimands: matched text units and explicit eligibility.

Package-native and original C1-prime values are different estimands. This
module scores the same nonempty comments after the documented chronological
sort and role rule, then reports original-like and package-like aggregates
without forcing them to match.
"""

from __future__ import annotations

from typing import Any

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .adapter import extract_post_body
from .comments import normalize_comments
from .original_semantics import (
    compound_score,
    early_late_shift,
    linear_slope,
    mean_abs_step,
    mean_or_zero,
)


def _mean_or_missing(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def score_texts(analyzer: SentimentIntensityAnalyzer, raw_post: dict[str, Any]) -> dict[str, Any]:
    title = "" if raw_post.get("title") is None else str(raw_post.get("title") or "")
    body, body_field, body_field_present = extract_post_body(raw_post)
    title_s = title.strip()
    body_s = body.strip()
    concat = f"{title}\n\n{body}".strip()
    return {
        "body_field": body_field,
        "body_field_present": body_field_present,
        "title_nonempty": bool(title_s),
        "body_nonempty": bool(body_s),
        "concat_nonempty": bool(concat),
        "title_compound": compound_score(analyzer, title),
        "body_compound": compound_score(analyzer, body) if body_s else None,
        "package_post_concat_compound": compound_score(analyzer, concat) if concat else None,
    }


def harmonized_comment_tracks(
    analyzer: SentimentIntensityAnalyzer,
    raw_post: dict[str, Any],
) -> dict[str, Any]:
    normalized = normalize_comments(raw_post, include_text=True)
    author_values: list[float] = []
    community_values: list[float] = []
    for comment in normalized["comments_chronological"]:
        if not comment.get("body_nonempty"):
            continue
        text = comment.get("text") or ""
        value = compound_score(analyzer, text)
        if comment.get("is_author"):
            author_values.append(value)
        else:
            community_values.append(value)
    return {
        "comments_status": normalized["comments_status"],
        "n_stored": normalized["n_stored"],
        "n_nonempty": normalized["n_nonempty"],
        "n_empty_body": normalized["n_empty_body"],
        "n_invalid_or_missing_timestamps": normalized["n_invalid_or_missing_timestamps"],
        "chronological_original_indices": normalized["chronological_original_indices"],
        "author_comment_compounds": author_values,
        "community_comment_compounds": community_values,
    }


def harmonized_measures(
    analyzer: SentimentIntensityAnalyzer,
    raw_post: dict[str, Any],
) -> dict[str, Any]:
    texts = score_texts(analyzer, raw_post)
    tracks = harmonized_comment_tracks(analyzer, raw_post)
    original_like_author: list[float] = []
    if texts["body_compound"] is not None:
        original_like_author.append(texts["body_compound"])
    original_like_author.extend(tracks["author_comment_compounds"])
    title_fallback = False
    if not original_like_author:
        original_like_author.append(texts["title_compound"])
        title_fallback = True

    author_comments = tracks["author_comment_compounds"]
    community_comments = tracks["community_comment_compounds"]
    return {
        **texts,
        "comments_status": tracks["comments_status"],
        "n_stored_comments": tracks["n_stored"],
        "n_nonempty_comments": tracks["n_nonempty"],
        "n_empty_comment_bodies": tracks["n_empty_body"],
        "n_invalid_or_missing_timestamps": tracks["n_invalid_or_missing_timestamps"],
        "n_author_comments_nonempty": len(author_comments),
        "n_community_comments_nonempty": len(community_comments),
        "author_comment_mean": _mean_or_missing(author_comments),
        "community_comment_mean": _mean_or_missing(community_comments),
        "author_comment_mean_is_observed": len(author_comments) > 0,
        "community_comment_mean_is_observed": len(community_comments) > 0,
        "author_shift_nonempty": early_late_shift(author_comments),
        "community_shift_nonempty": early_late_shift(community_comments),
        "author_shift_eligible": len(author_comments) >= 4,
        "community_shift_eligible": len(community_comments) >= 4,
        "original_like_author_track_mean": mean_or_zero(original_like_author),
        "original_like_author_n": len(original_like_author),
        "original_like_title_fallback": title_fallback,
        "original_like_used_body": texts["body_compound"] is not None,
        "original_like_author_trend": linear_slope(original_like_author),
        "original_like_author_volatility": mean_abs_step(original_like_author),
        "original_like_author_shift": early_late_shift(original_like_author),
        "package_like_author_avg": mean_or_zero(author_comments),
        "package_like_community_avg": mean_or_zero(community_comments),
        "package_like_author_avg_is_observed": len(author_comments) > 0,
        "package_like_community_avg_is_observed": len(community_comments) > 0,
    }


def discrepancy_reasons(
    *,
    original_row: dict[str, Any],
    package_summary: dict[str, Any],
    package_elig: dict[str, Any],
    harmonized: dict[str, Any],
    package_post_meta: dict[str, Any],
    original_string_indices: list[int],
    parsed_indices: list[int],
) -> list[str]:
    reasons: list[str] = []
    if original_row.get("title_fallback") or harmonized.get("original_like_title_fallback"):
        reasons.append("original_author_track_title_fallback")
    if original_row.get("used_body_as_first_author_point") or harmonized.get("original_like_used_body"):
        reasons.append("original_author_track_includes_post_body")
    reasons.append("package_author_avg_uses_author_comments_only")
    if not package_elig.get("author_avg_is_observed"):
        reasons.append("package_author_avg_zero_is_unobserved")
    if not package_elig.get("community_avg_is_observed"):
        reasons.append("package_community_avg_zero_is_unobserved")
    if package_summary.get("author_shift") == 0.0 and not package_elig.get("author_shift_eligible_package_native"):
        reasons.append("package_author_shift_zero_insufficient_n")
    if package_summary.get("community_shift") == 0.0 and not package_elig.get("community_shift_eligible_package_native"):
        reasons.append("package_community_shift_zero_insufficient_n")
    if original_string_indices != parsed_indices:
        reasons.append("comment_order_string_sort_vs_parsed_timestamp")
    if package_post_meta.get("n_empty_comment_bodies_skipped", 0) == 0 and any(
        not flag for flag in package_post_meta.get("comment_body_nonempty") or []
    ):
        reasons.append("package_native_includes_empty_comment_bodies")
    reasons.append("package_post_score_is_title_plus_body_concat")
    if original_row.get("community_mean_is_observed") is False:
        reasons.append("original_community_mean_zero_is_unobserved")
    return reasons
