"""Join locally loaded sources into per-ID normalized records.

Public exports use hashes and flags only. Protected reserved/fresh files contribute
membership flags, never outcomes. March and October snapshots are kept side by side.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any, Iterable, Iterator

from .adapter import extract_post_body
from .comments import (
    normalize_comments,
    normalize_pre_split_role_comments,
    public_comment,
)
from .hashing import sha256_text
from .schema import (
    ANALYSIS_SNAPSHOT_RULES,
    LABEL_FIELD_NAMES,
    RECORD_REQUIRED_KEYS,
    SNAPSHOT_REQUIRED_KEYS,
    schema_document,
)

PROTECTED_OUTCOME_FIELDS = (
    "s2_target",
    "s2_in_scope",
    "about_software",
    "update_related",
    "discussion_type",
    "label_provenance",
    "confidence",
    "phase2_about_software",
    "phase2_update_related",
    "phase2_discussion_type",
    "corrected_label",
)

SOURCE_KEYS = (
    "historical_labels",
    "historical_raw_tps",
    "historical_raw_gds",
    "c1_prime_saved",
    "engagement_cohort_324",
    "phase2_master",
    "october5_api_pages",
    "development_v2",
    "reddit_development",
    "reserved_test",
    "fresh_500",
)


def source_id_sets(loaded: dict[str, Any]) -> dict[str, set[str]]:
    return {
        "historical_labels": set(loaded["historical_labels"]["inventory"]["ids"]),
        "historical_raw_tps": set(loaded["historical_raw"]["tps"]["inventory"]["ids"]),
        "historical_raw_gds": set(loaded["historical_raw"]["gds"]["inventory"]["ids"]),
        "c1_prime_saved": set(loaded["c1_prime"]["inventory"]["ids"]),
        "engagement_cohort_324": set(loaded["cohort"]["ids"]),
        "phase2_master": set(loaded["master"]["ids"]),
        "october5_api_pages": set(loaded["oct5"]["ids"]),
        "development_v2": set(loaded["development"]["ids"]),
        "reddit_development": set(loaded["reddit_development"]["ids"]),
        "reserved_test": set(loaded["reserved"]["ids"]),
        "fresh_500": set(loaded["fresh"]["ids"]),
    }


def union_reddit_ids(loaded: dict[str, Any]) -> list[str]:
    union: set[str] = set()
    for ids in source_id_sets(loaded).values():
        union.update(ids)
    return sorted(union)


def _parse_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_optional_bool(value: Any) -> bool | None:
    if value is True or value is False:
        return value
    if value is None or value == "":
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _comment_bundle(raw_post: dict[str, Any], *, include_text: bool, pre_split: bool) -> dict[str, Any]:
    if pre_split:
        return normalize_pre_split_role_comments(raw_post, include_text=include_text)
    return normalize_comments(raw_post, include_text=include_text)


def build_text_snapshot(
    raw_post: dict[str, Any],
    *,
    source_key: str,
    source_file: str,
    capture_time: Any,
    analysis_priority: str,
    occurrence_index: int = 0,
    include_comment_text: bool = False,
    pre_split: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    title = "" if raw_post.get("title") is None else str(raw_post.get("title") or "")
    body, body_field, body_field_present = extract_post_body(raw_post)
    bundle = _comment_bundle(raw_post, include_text=include_comment_text, pre_split=pre_split)
    comments_public = [public_comment(item) for item in bundle["comments_original_order"]]
    snapshot = {
        "source_key": source_key,
        "source_file": source_file,
        "capture_time": capture_time,
        "analysis_priority": analysis_priority,
        "occurrence_index": occurrence_index,
        "title_present": bool(title.strip()),
        "title_sha256": sha256_text(title),
        "title_len": len(title),
        "body_field": body_field,
        "body_field_present": body_field_present,
        "body_present": bool(str(body or "").strip()),
        "body_sha256": sha256_text(body),
        "body_len": len(body),
        "comments_status": bundle["comments_status"],
        "comment_layout": bundle.get("comment_layout", "raw_comments_array"),
        "n_stored_comments": bundle["n_stored"],
        "n_nonempty_comments": bundle["n_nonempty"],
        "n_empty_comment_bodies": bundle["n_empty_body"],
        "n_author_replies_nonempty": bundle["n_author_replies_nonempty"],
        "n_community_replies_nonempty": bundle["n_community_replies_nonempty"],
        "n_valid_timestamps_nonempty": bundle["n_valid_timestamps_nonempty"],
        "n_invalid_or_missing_timestamps": bundle["n_invalid_or_missing_timestamps"],
        "num_comments_field": raw_post.get("num_comments", raw_post.get("n_comments_stored")),
        "chronological_original_indices": bundle["chronological_original_indices"],
        "author_shift_eligible": bundle["n_author_replies_nonempty"] >= 4,
        "community_shift_eligible": bundle["n_community_replies_nonempty"] >= 4,
        "comments": comments_public,
    }
    if extra:
        snapshot.update(extra)
    return snapshot


def _snapshot_conflicts(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    title_conflict = False
    body_conflict = False
    for i, left in enumerate(snapshots):
        for right in snapshots[i + 1 :]:
            title_diff = left.get("title_sha256") != right.get("title_sha256")
            body_diff = left.get("body_sha256") != right.get("body_sha256")
            if not title_diff and not body_diff:
                continue
            title_conflict = title_conflict or title_diff
            body_conflict = body_conflict or body_diff
            pairs.append(
                {
                    "left_source_key": left["source_key"],
                    "left_occurrence_index": left.get("occurrence_index", 0),
                    "right_source_key": right["source_key"],
                    "right_occurrence_index": right.get("occurrence_index", 0),
                    "title_hash_conflict": title_diff,
                    "body_hash_conflict": body_diff,
                }
            )
            left.setdefault("text_incompatible_with", [])
            right.setdefault("text_incompatible_with", [])
            left["text_incompatible_with"].append(
                {"source_key": right["source_key"], "occurrence_index": right.get("occurrence_index", 0)}
            )
            right["text_incompatible_with"].append(
                {"source_key": left["source_key"], "occurrence_index": left.get("occurrence_index", 0)}
            )
    return {
        "title_hash_conflict": title_conflict,
        "body_hash_conflict": body_conflict,
        "n_conflicting_pairs": len(pairs),
        "conflicting_snapshot_pairs": pairs,
    }


def choose_analysis_snapshot(record: dict[str, Any], analysis: str) -> dict[str, Any] | None:
    """Return the documented snapshot for an analysis name; never invent a live replacement."""
    preferred_keys = ANALYSIS_SNAPSHOT_RULES.get(analysis)
    if not preferred_keys:
        raise KeyError(f"unknown analysis snapshot rule: {analysis}")
    snapshots = record.get("snapshots") or []
    for source_key in preferred_keys:
        matches = [snap for snap in snapshots if snap.get("source_key") == source_key]
        if matches:
            return min(matches, key=lambda snap: int(snap.get("occurrence_index") or 0))
    return None


def _labels_for_id(loaded: dict[str, Any], reddit_id: str, memberships: list[str]) -> dict[str, Any]:
    labels = {name: None for name in LABEL_FIELD_NAMES}
    if "historical_labels" in memberships:
        row = loaded["historical_labels"]["by_id"].get(reddit_id) or {}
        labels["historical_corrected_label"] = _parse_optional_int(row.get("corrected_label"))
    if "phase2_master" in memberships:
        row = loaded["master"]["by_id"].get(reddit_id) or {}
        labels["phase2_about_software"] = row.get("phase2_about_software")
        labels["phase2_update_related"] = row.get("phase2_update_related")
        labels["phase2_discussion_type"] = row.get("phase2_discussion_type")
        labels["phase2_review_status"] = row.get("phase2_review_status")
    if "development_v2" in memberships:
        row = loaded["development"]["by_id"].get(reddit_id) or {}
        labels["s2_in_scope"] = _parse_optional_bool(row.get("s2_in_scope"))
        labels["s2_target"] = _parse_optional_int(row.get("s2_target"))
    if "october5_api_pages" in memberships:
        labels["api_candidate_route"] = loaded["oct5"]["candidate_route_by_id"].get(reddit_id)
        predicted = loaded["oct5"]["metadata_predicted_by_id"].get(reddit_id)
        labeled = loaded["oct5"]["metadata_labeled_by_id"].get(reddit_id)
        labels["api_metadata_predicted"] = deepcopy(predicted) if predicted is not None else None
        labels["api_metadata_labeled"] = deepcopy(labeled) if labeled is not None else None
    labels["split_reserved"] = "reserved_test" in memberships
    labels["split_fresh"] = "fresh_500" in memberships
    # Reserved/fresh files are membership-only. Do not read their row bodies or outcome fields.
    return labels


def _snapshots_for_id(
    loaded: dict[str, Any],
    reddit_id: str,
    memberships: list[str],
    *,
    include_comment_text: bool,
) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    if "historical_raw_tps" in memberships:
        post = loaded["historical_raw"]["tps"]["by_id"][reddit_id]
        snapshots.append(
            build_text_snapshot(
                post,
                source_key="historical_raw_tps",
                source_file=loaded["historical_raw"]["tps"]["file"],
                capture_time=loaded["historical_raw"]["tps"]["capture_time"],
                analysis_priority="historical_reproduction",
                include_comment_text=include_comment_text,
            )
        )
    if "historical_raw_gds" in memberships:
        post = loaded["historical_raw"]["gds"]["by_id"][reddit_id]
        snapshots.append(
            build_text_snapshot(
                post,
                source_key="historical_raw_gds",
                source_file=loaded["historical_raw"]["gds"]["file"],
                capture_time=loaded["historical_raw"]["gds"]["capture_time"],
                analysis_priority="historical_reproduction",
                include_comment_text=include_comment_text,
            )
        )
    if "october5_api_pages" in memberships:
        occurrences = loaded["oct5"].get("occurrences_by_id", {}).get(reddit_id) or []
        if not occurrences:
            post = loaded["oct5"]["by_id"][reddit_id]
            occurrences = [
                {
                    "page_name": None,
                    "source_file": loaded["oct5"]["pages"][0]["file"] if loaded["oct5"].get("pages") else "",
                    "route": loaded["oct5"]["candidate_route_by_id"].get(reddit_id),
                    "capture_time": loaded["oct5"]["pagination_manifest"].get("collected_at_utc"),
                    "post": post,
                }
            ]
        for index, item in enumerate(occurrences):
            snapshots.append(
                build_text_snapshot(
                    item["post"],
                    source_key="october5_api_pages",
                    source_file=item["source_file"],
                    capture_time=item["capture_time"],
                    analysis_priority="master_source_conversations",
                    occurrence_index=index,
                    include_comment_text=include_comment_text,
                    extra={"page_name": item.get("page_name"), "api_candidate_route": item.get("route")},
                )
            )
    if "engagement_cohort_324" in memberships:
        post = loaded["cohort"]["by_id"][reddit_id]
        snapshots.append(
            build_text_snapshot(
                post,
                source_key="engagement_cohort_324",
                source_file=loaded["cohort"]["file"],
                capture_time=loaded["cohort"].get("capture_time"),
                analysis_priority="engagement_selected_not_representative",
                include_comment_text=include_comment_text,
                pre_split=True,
                extra={"not_representative": True},
            )
        )
    if "phase2_master" in memberships:
        post = loaded["master"]["by_id"][reddit_id]
        snapshots.append(
            build_text_snapshot(
                post,
                source_key="phase2_master",
                source_file=loaded["master"]["file"],
                capture_time=None,
                analysis_priority="phase2_reviewed_text_not_raw_comments",
                include_comment_text=include_comment_text,
            )
        )
    if "development_v2" in memberships:
        post = loaded["development"]["by_id"][reddit_id]
        snapshots.append(
            build_text_snapshot(
                post,
                source_key="development_v2",
                source_file=loaded["development"]["file"],
                capture_time=None,
                analysis_priority="development_text_not_raw_comments",
                include_comment_text=include_comment_text,
            )
        )
    if "reddit_development" in memberships:
        post = loaded["reddit_development"]["by_id"][reddit_id]
        snapshots.append(
            build_text_snapshot(
                post,
                source_key="reddit_development",
                source_file=loaded["reddit_development"]["file"],
                capture_time=None,
                analysis_priority="reddit_url_development_text_not_raw_comments",
                include_comment_text=include_comment_text,
            )
        )
    return snapshots


def build_normalized_record(
    loaded: dict[str, Any],
    reddit_id: str,
    *,
    include_comment_text: bool = False,
    id_sets: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    if not reddit_id or not str(reddit_id).strip():
        raise ValueError("reddit_id must be nonempty")
    reddit_id = str(reddit_id).strip()
    id_sets = id_sets or source_id_sets(loaded)
    memberships = [key for key in SOURCE_KEYS if reddit_id in id_sets[key]]
    duplicate_id_occurrences: dict[str, int] = {}
    if reddit_id in id_sets["october5_api_pages"]:
        duplicate_id_occurrences["october5_api_pages"] = len(
            loaded["oct5"].get("occurrences_by_id", {}).get(reddit_id) or [loaded["oct5"]["by_id"][reddit_id]]
        )
    if reddit_id in id_sets["historical_raw_tps"]:
        duplicate_id_occurrences["historical_raw_tps"] = 1
    if reddit_id in id_sets["historical_raw_gds"]:
        duplicate_id_occurrences["historical_raw_gds"] = 1
    snapshots = _snapshots_for_id(loaded, reddit_id, memberships, include_comment_text=include_comment_text)
    conflicts = _snapshot_conflicts(snapshots)
    labels = _labels_for_id(loaded, reddit_id, memberships)
    availability = {}
    for snap in snapshots:
        availability[f"{snap['source_key']}#{snap.get('occurrence_index', 0)}"] = {
            "comments_status": snap["comments_status"],
            "n_stored_comments": snap["n_stored_comments"],
            "n_nonempty_comments": snap["n_nonempty_comments"],
            "n_author_replies_nonempty": snap["n_author_replies_nonempty"],
            "n_community_replies_nonempty": snap["n_community_replies_nonempty"],
            "n_valid_timestamps_nonempty": snap["n_valid_timestamps_nonempty"],
            "n_invalid_or_missing_timestamps": snap["n_invalid_or_missing_timestamps"],
            "author_shift_eligible": snap["author_shift_eligible"],
            "community_shift_eligible": snap["community_shift_eligible"],
        }
    record = {
        "reddit_id": reddit_id,
        "source_memberships": memberships,
        "duplicate_id_occurrences": duplicate_id_occurrences,
        "snapshots": snapshots,
        "snapshot_conflicts": conflicts,
        "labels": labels,
        "comment_availability_by_snapshot": availability,
        "protected_outcomes_copied": False,
        "analysis_snapshot_choice": {
            name: (
                None
                if (chosen := choose_analysis_snapshot({"snapshots": snapshots}, name)) is None
                else chosen["source_key"]
            )
            for name in ANALYSIS_SNAPSHOT_RULES
        },
    }
    return record


def iter_normalized_records(
    loaded: dict[str, Any],
    *,
    reddit_ids: Iterable[str] | None = None,
    include_comment_text: bool = False,
) -> Iterator[dict[str, Any]]:
    """Yield one joined record per nonempty Reddit ID, deterministic sorted order."""
    id_sets = source_id_sets(loaded)
    ids = sorted({str(rid).strip() for rid in reddit_ids if str(rid).strip()}) if reddit_ids is not None else union_reddit_ids(loaded)
    for reddit_id in ids:
        yield build_normalized_record(
            loaded,
            reddit_id,
            include_comment_text=include_comment_text,
            id_sets=id_sets,
        )


def record_matches_schema(record: dict[str, Any]) -> list[str]:
    """Return missing-field messages; empty list means the materialized record matches the schema."""
    problems: list[str] = []
    for key in RECORD_REQUIRED_KEYS:
        if key not in record:
            problems.append(f"missing top-level field {key}")
    labels = record.get("labels") or {}
    for key in LABEL_FIELD_NAMES:
        if key not in labels:
            problems.append(f"missing label field {key}")
    for snap in record.get("snapshots") or []:
        for key in SNAPSHOT_REQUIRED_KEYS:
            if key not in snap:
                problems.append(f"snapshot {snap.get('source_key')} missing {key}")
        for comment in snap.get("comments") or []:
            for key in ("original_index", "chronological_index", "body_nonempty", "timestamp_valid", "is_author", "author_role_basis"):
                if key not in comment:
                    problems.append(f"comment missing {key}")
            if "author" in comment or "text" in comment:
                problems.append("public comment leaked username or raw text")
        if "api_candidate_route" in labels and labels.get("api_candidate_route") in {0, 1}:
            problems.append("api_candidate_route must not occupy a human TPS/GDS label")
    if labels.get("split_fresh") and loaded_protected_outcome_present(labels) and "development_v2" not in record.get("source_memberships", []):
        problems.append("fresh split unexpectedly has development-only s2 fields")
    schema_id = schema_document()["schema_id"]
    if schema_id != "full_resource_sentiment_v1.normalized_record":
        problems.append("schema_id drifted")
    return problems


def loaded_protected_outcome_present(labels: dict[str, Any]) -> bool:
    return labels.get("s2_target") is not None or labels.get("s2_in_scope") is not None


def audit_normalized_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    membership_counts: Counter[str] = Counter()
    combination_counts: Counter[str] = Counter()
    source_comment_stats: dict[str, Counter[str]] = {}
    n_title_conflict = 0
    n_body_conflict = 0
    n_reserved = 0
    n_fresh = 0
    n_reserved_with_s2 = 0
    n_fresh_with_s2 = 0
    n_protected_text_leak = 0
    for record in rows:
        for source in record["source_memberships"]:
            membership_counts[source] += 1
        combination_counts["|".join(record["source_memberships"])] += 1
        conflicts = record.get("snapshot_conflicts") or {}
        if conflicts.get("title_hash_conflict"):
            n_title_conflict += 1
        if conflicts.get("body_hash_conflict"):
            n_body_conflict += 1
        if record["labels"].get("split_reserved"):
            n_reserved += 1
            if record["labels"].get("s2_target") is not None:
                n_reserved_with_s2 += 1
        if record["labels"].get("split_fresh"):
            n_fresh += 1
            if record["labels"].get("s2_target") is not None:
                n_fresh_with_s2 += 1
        for snap in record.get("snapshots") or []:
            stats = source_comment_stats.setdefault(snap["source_key"], Counter())
            stats["n_snapshots"] += 1
            stats[f"comments_status:{snap['comments_status']}"] += 1
            if snap["n_nonempty_comments"] > 0:
                stats["posts_with_nonempty_comments"] += 1
            if snap["n_author_replies_nonempty"] > 0:
                stats["posts_with_author_reply"] += 1
            if snap["n_community_replies_nonempty"] > 0:
                stats["posts_with_community_reply"] += 1
            if snap["n_author_replies_nonempty"] > 0 and snap["n_community_replies_nonempty"] > 0:
                stats["posts_with_both_roles"] += 1
            if snap["n_valid_timestamps_nonempty"] > 0:
                stats["posts_with_valid_comment_timestamps"] += 1
            if snap["n_invalid_or_missing_timestamps"] > 0:
                stats["posts_with_invalid_or_missing_comment_timestamps"] += 1
            if snap["author_shift_eligible"]:
                stats["posts_author_shift_eligible"] += 1
            if snap["community_shift_eligible"]:
                stats["posts_community_shift_eligible"] += 1
            for comment in snap.get("comments") or []:
                if "text" in comment or "author" in comment:
                    n_protected_text_leak += 1
    return {
        "n_records": len(rows),
        "n_by_source_membership": dict(membership_counts),
        "n_membership_combinations": len(combination_counts),
        "membership_combination_counts": dict(sorted(combination_counts.items(), key=lambda item: (-item[1], item[0]))),
        "snapshot_conflicts": {
            "n_records_title_hash_conflict": n_title_conflict,
            "n_records_body_hash_conflict": n_body_conflict,
        },
        "comment_availability_by_source": {key: dict(value) for key, value in sorted(source_comment_stats.items())},
        "protected": {
            "n_split_reserved": n_reserved,
            "n_split_fresh": n_fresh,
            "n_reserved_with_s2_target": n_reserved_with_s2,
            "n_fresh_with_s2_target": n_fresh_with_s2,
            "protected_outcomes_copied": False,
            "n_public_comments_with_text_or_username": n_protected_text_leak,
        },
        "documented_snapshot_rules": ANALYSIS_SNAPSHOT_RULES,
        "notes": [
            "Each record is one unique nonempty reddit_id. Overlapping sources are memberships, not extra threads.",
            "March and October snapshots are stored separately. Incompatible title/body hashes are flagged.",
            "Reserved/fresh contribute split flags only. Development s2 fields are used only when the ID is in development_v2.",
            "The engagement cohort is reply-selected and not representative.",
        ],
    }
