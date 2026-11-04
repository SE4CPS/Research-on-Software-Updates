"""Map ReleaseTrain raw posts to releasetrain-sentiment 0.2.1 score_post input.

Do not use fetch.to_scoreable_post for these records: that helper looks for
selftext/description, not author_description, and treats blank author names as
a match.
"""

from __future__ import annotations

from typing import Any

from .comments import comments_status, extract_comment_text, normalize_comments
from .roles import infer_author_role


def extract_post_body(raw_post: dict[str, Any]) -> tuple[str, str, bool]:
    """Return (body, field_name, field_present). Prefer author_description."""
    for field in ("author_description", "post_body", "selftext", "description"):
        if field in raw_post:
            value = raw_post.get(field)
            return ("" if value is None else str(value), field, True)
    return "", "absent", False


def to_package_post(
    raw_post: dict[str, Any],
    *,
    include_empty_comment_bodies: bool = True,
) -> dict[str, Any]:
    """Build the package Post shape: title, description, comments[{text,author,is_author}].

    Comments are given in verified chronological order (parsed timestamps,
    original-index tie-break). The original post is not added as a comment;
    score_post prepends title+description as source='post'.
    """
    title = "" if raw_post.get("title") is None else str(raw_post.get("title") or "")
    description, body_field, body_field_present = extract_post_body(raw_post)
    normalized = normalize_comments(raw_post, include_text=True)
    raw_comments = raw_post.get("comments") if isinstance(raw_post.get("comments"), list) else []
    by_original = {index: comment for index, comment in enumerate(raw_comments) if isinstance(comment, dict)}

    package_comments: list[dict[str, Any]] = []
    skipped_empty = 0
    for record in normalized["comments_chronological"]:
        if record.get("malformed"):
            continue
        original = by_original.get(record["original_index"], {})
        text, _field = extract_comment_text(original)
        if not text.strip() and not include_empty_comment_bodies:
            skipped_empty += 1
            continue
        role = infer_author_role(raw_post.get("author"), original.get("author"), original.get("is_submitter"))
        package_comments.append(
            {
                "text": text,
                "author": "" if original.get("author") is None else str(original.get("author") or ""),
                "is_author": bool(role["is_author"]),
                "original_index": record["original_index"],
                "author_role_basis": role["author_role_basis"],
                "body_nonempty": bool(record.get("body_nonempty")),
            }
        )

    return {
        "title": title,
        "description": description,
        "comments": [
            {"text": item["text"], "author": item["author"], "is_author": item["is_author"]}
            for item in package_comments
        ],
        "adapter_meta": {
            "body_field": body_field,
            "body_field_present": body_field_present,
            "comments_status": comments_status(raw_post),
            "n_comments_passed": len(package_comments),
            "n_empty_comment_bodies_skipped": skipped_empty,
            "chronological_original_indices": [item["original_index"] for item in package_comments],
            "comment_is_author": [item["is_author"] for item in package_comments],
            "comment_role_basis": [item["author_role_basis"] for item in package_comments],
            "comment_body_nonempty": [item["body_nonempty"] for item in package_comments],
            "used_fetch_to_scoreable_post": False,
            "post_kept_separate_from_author_replies": True,
        },
    }


def package_eligibility(package_post: dict[str, Any]) -> dict[str, Any]:
    """Interpret trajectory_summary zeros. 0.0 is not observed neutrality."""
    comments = package_post.get("comments") or []
    author_nonempty = [
        c for c in comments if c.get("is_author") and str(c.get("text") or "").strip()
    ]
    community_nonempty = [
        c for c in comments if (not c.get("is_author")) and str(c.get("text") or "").strip()
    ]
    author_any = [c for c in comments if c.get("is_author")]
    community_any = [c for c in comments if not c.get("is_author")]
    return {
        "n_author_comments_passed": len(author_any),
        "n_community_comments_passed": len(community_any),
        "n_author_comments_nonempty": len(author_nonempty),
        "n_community_comments_nonempty": len(community_nonempty),
        "author_avg_is_observed": len(author_any) > 0,
        "community_avg_is_observed": len(community_any) > 0,
        "author_avg_nonempty_is_observed": len(author_nonempty) > 0,
        "community_avg_nonempty_is_observed": len(community_nonempty) > 0,
        "author_shift_eligible": len(author_nonempty) >= 4,
        "community_shift_eligible": len(community_nonempty) >= 4,
        "package_shift_uses_all_passed_comments_including_empty": True,
        "author_shift_eligible_package_native": len(author_any) >= 4,
        "community_shift_eligible_package_native": len(community_any) >= 4,
    }
