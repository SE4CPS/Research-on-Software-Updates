"""Author-role inference with an explicit basis. Blank names never match."""

from __future__ import annotations

from typing import Any


def parse_bool(value: Any) -> bool | None:
    if value is True or value is False:
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)) and value in (0, 1) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "t", "yes", "1"}:
            return True
        if lowered in {"false", "f", "no", "0"}:
            return False
        if lowered == "":
            return None
    return None


def infer_author_role(
    post_author: Any,
    comment_author: Any,
    is_submitter_raw: Any,
) -> dict[str, Any]:
    """is_submitter True wins; otherwise exact nonempty name match.

    Blank or missing names are not treated as author replies. The original
    C1-prime helper uses bool(is_submitter) or (nonempty post author and
    stripped equality). Package fetch.to_scoreable_post matches blank names
    and does not strip; this function does not follow that fetch helper.
    """
    is_submitter = parse_bool(is_submitter_raw)
    post_name = "" if post_author is None else str(post_author).strip()
    comment_name = "" if comment_author is None else str(comment_author).strip()
    name_match = bool(post_name) and bool(comment_name) and post_name == comment_name
    conflict = is_submitter is True and bool(post_name) and bool(comment_name) and post_name != comment_name

    if is_submitter is True:
        basis = "is_submitter"
        is_author = True
    elif name_match:
        basis = "name_match"
        is_author = True
    elif not post_name and not comment_name:
        basis = "blank_names_not_matched"
        is_author = False
    else:
        basis = "not_author"
        is_author = False

    return {
        "is_author": is_author,
        "author_role_basis": basis,
        "is_submitter_parsed": is_submitter,
        "name_match": name_match,
        "blank_post_author": not bool(post_name),
        "blank_comment_author": not bool(comment_name),
        "role_conflict_is_submitter_vs_name": conflict,
        "author_name_present": bool(comment_name),
    }


def original_is_author(post_author: Any, comment: dict[str, Any]) -> bool:
    """Exact original C1-prime rule from trajectory/build.py."""
    author_name = (post_author or "").strip() if isinstance(post_author, str) or post_author else str(post_author or "").strip()
    return bool(comment.get("is_submitter")) or (
        bool(author_name) and str(comment.get("author") or "").strip() == author_name
    )
