"""Deterministic read-only loaders. Duplicate IDs are flagged, never silently merged."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .adapter import extract_post_body
from .comments import comments_status, normalize_comments
from .hashing import sha256_file, sha256_text
from .io_utils import nonempty_reddit_id, read_csv_dicts, read_json, read_jsonl
from .paths import (
    C1_PRIME_METADATA,
    C1_PRIME_METRICS,
    COHORT_324,
    DEVELOPMENT_V2,
    FRESH_500,
    HISTORICAL_LABELS,
    HISTORICAL_RAW_GDS,
    HISTORICAL_RAW_TPS,
    MASTER,
    OCT5_API_DIR,
    OCT5_PAGE_FILES,
    OCT5_PAGINATION_MANIFEST,
    PACKAGE_TARBALL,
    REDDIT_DEVELOPMENT,
    RESERVED_TEST,
    all_input_files,
)


def _id_inventory(records: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [nonempty_reddit_id(row) for row in records]
    nonempty = [i for i in ids if i]
    counts = Counter(nonempty)
    duplicates = sorted(rid for rid, n in counts.items() if n > 1)
    return {
        "n_rows": len(records),
        "n_blank_ids": sum(1 for i in ids if not i),
        "n_nonempty_ids": len(nonempty),
        "n_unique_ids": len(counts),
        "duplicate_ids": duplicates,
        "n_duplicate_ids": len(duplicates),
        "ids": sorted(counts),
        "id_counts": {rid: counts[rid] for rid in sorted(counts) if counts[rid] > 1},
    }


def _text_coverage_from_api_posts(posts: list[dict[str, Any]]) -> dict[str, Any]:
    title_n = 0
    body_n = 0
    comments_absent = 0
    comments_empty = 0
    comments_nonempty = 0
    malformed = 0
    stored = 0
    nonempty_comments = 0
    author_reply_posts = 0
    community_reply_posts = 0
    both_roles = 0
    invalid_ts_posts = 0
    created = []
    for post in posts:
        if str(post.get("title") or "").strip():
            title_n += 1
        _body, _field, present = extract_post_body(post)
        if present and str(_body or "").strip():
            body_n += 1
        status = comments_status(post)
        if status == "absent":
            comments_absent += 1
        elif status == "present_empty":
            comments_empty += 1
        elif status == "malformed":
            malformed += 1
        else:
            comments_nonempty += 1
        if isinstance(post.get("comments"), list) or status == "present_empty":
            norm = normalize_comments(post)
            stored += norm["n_stored"]
            nonempty_comments += norm["n_nonempty"]
            if norm["n_author_replies_nonempty"] > 0:
                author_reply_posts += 1
            if norm["n_community_replies_nonempty"] > 0:
                community_reply_posts += 1
            if norm["n_author_replies_nonempty"] > 0 and norm["n_community_replies_nonempty"] > 0:
                both_roles += 1
            if norm["n_invalid_or_missing_timestamps"] > 0:
                invalid_ts_posts += 1
        ts = str(post.get("created_utc") or "").strip()
        if ts:
            created.append(ts)
    created_sorted = sorted(created)
    return {
        "title_present": title_n,
        "body_present": body_n,
        "comments_key_absent_or_null": comments_absent,
        "comments_present_empty": comments_empty,
        "comments_present_nonempty": comments_nonempty,
        "comments_malformed": malformed,
        "sum_stored_comments": stored,
        "sum_nonempty_comments": nonempty_comments,
        "posts_with_author_reply": author_reply_posts,
        "posts_with_community_reply": community_reply_posts,
        "posts_with_both_roles": both_roles,
        "posts_with_invalid_or_missing_comment_timestamp": invalid_ts_posts,
        "created_utc_min": created_sorted[0] if created_sorted else None,
        "created_utc_max": created_sorted[-1] if created_sorted else None,
    }


def file_manifest_entry(path, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    entry = {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else None,
        "sha256": sha256_file(path) if path.exists() else None,
    }
    if extra:
        entry.update(extra)
    return entry


def load_historical_labels() -> dict[str, Any]:
    rows = read_csv_dicts(HISTORICAL_LABELS)
    inv = _id_inventory(rows)
    by_id: dict[str, dict[str, Any]] = {}
    label_counts: Counter[int] = Counter()
    for row in rows:
        rid = nonempty_reddit_id(row)
        if not rid:
            continue
        label = int(row["corrected_label"])
        label_counts[label] += 1
        by_id.setdefault(rid, row)
    return {
        "source_key": "historical_labels",
        "file": str(HISTORICAL_LABELS),
        "inventory": inv,
        "by_id": by_id,
        "corrected_label_counts": {str(k): int(v) for k, v in sorted(label_counts.items())},
        "label_provenance": "historical_corrected_label from updated_labeled_dataset_unique.csv",
    }


def load_historical_raw() -> dict[str, Any]:
    tps_payload = read_json(HISTORICAL_RAW_TPS)
    gds_payload = read_json(HISTORICAL_RAW_GDS)
    tps_posts = list(tps_payload.get("data") or [])
    gds_posts = list(gds_payload.get("data") or [])
    tps_inv = _id_inventory(tps_posts)
    gds_inv = _id_inventory(gds_posts)
    tps_ids = set(tps_inv["ids"])
    gds_ids = set(gds_inv["ids"])
    tps_by_id = {nonempty_reddit_id(p): p for p in tps_posts if nonempty_reddit_id(p)}
    gds_by_id = {nonempty_reddit_id(p): p for p in gds_posts if nonempty_reddit_id(p)}
    # Original load_raw_posts_by_id writes TPS then GDS, so GDS overwrites overlaps.
    merged: dict[str, dict[str, Any]] = dict(tps_by_id)
    overwritten = sorted(rid for rid in gds_by_id if rid in merged)
    merged.update(gds_by_id)
    return {
        "source_key": "historical_raw",
        "tps": {
            "file": str(HISTORICAL_RAW_TPS),
            "capture_time": tps_payload.get("fetched_at_utc"),
            "filters": tps_payload.get("filters"),
            "header_total": tps_payload.get("total"),
            "inventory": tps_inv,
            "coverage": _text_coverage_from_api_posts(tps_posts),
            "by_id": tps_by_id,
        },
        "gds": {
            "file": str(HISTORICAL_RAW_GDS),
            "capture_time": gds_payload.get("fetched_at_utc"),
            "filters": gds_payload.get("filters"),
            "header_total": gds_payload.get("total"),
            "inventory": gds_inv,
            "coverage": _text_coverage_from_api_posts(gds_posts),
            "by_id": gds_by_id,
        },
        "tps_gds_id_overlap": sorted(tps_ids & gds_ids),
        "n_tps_gds_id_overlap": len(tps_ids & gds_ids),
        "union_ids": sorted(tps_ids | gds_ids),
        "n_union_ids": len(tps_ids | gds_ids),
        "original_merge_rule": "tps_response.json then gds_response.json; GDS overwrites identical redditId",
        "ids_where_gds_overwrites_tps": overwritten,
        "merged_by_id_original_rule": merged,
    }


def load_c1_prime() -> dict[str, Any]:
    rows = read_csv_dicts(C1_PRIME_METRICS)
    inv = _id_inventory(rows)
    by_id = {}
    for row in rows:
        rid = nonempty_reddit_id(row)
        if rid:
            by_id[rid] = row
    metadata = read_json(C1_PRIME_METADATA)
    return {
        "source_key": "c1_prime_saved",
        "file": str(C1_PRIME_METRICS),
        "metadata_file": str(C1_PRIME_METADATA),
        "inventory": inv,
        "by_id": by_id,
        "build_metadata": metadata,
    }


def load_cohort_324() -> dict[str, Any]:
    payload = read_json(COHORT_324)
    posts = list(payload.get("all_analyzed_posts") or [])
    inv = _id_inventory(posts)
    by_id: dict[str, dict[str, Any]] = {}
    for post in posts:
        rid = nonempty_reddit_id(post)
        if rid and rid not in by_id:
            by_id[rid] = post
    return {
        "source_key": "engagement_cohort_324",
        "file": str(COHORT_324),
        "capture_time": (payload.get("analysis_metadata") or {}).get("date"),
        "analysis_metadata": payload.get("analysis_metadata"),
        "inventory": inv,
        "by_id": by_id,
        "note": (
            "Selected for reply activity; not representative of all posts. "
            "Comments are already split into author_replies and community_comments."
        ),
        "ids": inv["ids"],
    }


def load_master() -> dict[str, Any]:
    records = read_json(MASTER)
    if not isinstance(records, list):
        raise TypeError("master_records.json is not a list")
    inv = _id_inventory(records)
    by_id = {}
    review_status = Counter()
    source_family = Counter()
    uncertain = []
    for row in records:
        rid = nonempty_reddit_id(row)
        if not rid:
            continue
        by_id[rid] = row
        status = str(row.get("phase2_review_status") or "")
        review_status[status] += 1
        source_family[str(row.get("source_family") or "")] += 1
        if status == "manu_assisted_uncertain":
            uncertain.append(rid)
    return {
        "source_key": "phase2_master",
        "file": str(MASTER),
        "inventory": inv,
        "by_id": by_id,
        "phase2_review_status_counts": dict(review_status),
        "source_family_counts": dict(source_family),
        "uncertain_review_ids": sorted(uncertain),
        "n_uncertain_review": len(uncertain),
        "ids": inv["ids"],
    }


def load_oct5_pages() -> dict[str, Any]:
    pagination = read_json(OCT5_PAGINATION_MANIFEST)
    pages = []
    all_posts: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    route_by_id: dict[str, str] = {}
    predicted_by_id: dict[str, Any] = {}
    labeled_by_id: dict[str, Any] = {}
    occurrences: dict[str, list[str]] = defaultdict(list)
    occurrences_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    page_meta_by_file = {
        str(page.get("file") or ""): page for page in (pagination.get("pages") or []) if isinstance(page, dict)
    }
    for name in OCT5_PAGE_FILES:
        path = OCT5_API_DIR / name
        payload = read_json(path)
        posts = list(payload.get("data") or [])
        route = "gds" if name.startswith("gds") else "tps"
        inv = _id_inventory(posts)
        page_meta = page_meta_by_file.get(name) or {}
        capture_time = page_meta.get("header_date") or pagination.get("collected_at_utc")
        for post in posts:
            rid = nonempty_reddit_id(post)
            if not rid:
                continue
            occurrences[rid].append(name)
            occurrences_by_id[rid].append(
                {
                    "page_name": name,
                    "source_file": str(path),
                    "route": route,
                    "capture_time": capture_time,
                    "post": post,
                }
            )
            all_posts.append(post)
            if rid not in by_id:
                by_id[rid] = post
                route_by_id[rid] = route
                metadata = post.get("metadata") if isinstance(post.get("metadata"), dict) else {}
                predicted_by_id[rid] = metadata.get("predicted")
                labeled_by_id[rid] = metadata.get("labeled")
        pages.append(
            {
                "file": str(path),
                "name": name,
                "route": route,
                "pagination": payload.get("pagination"),
                "criteria_keys": list(payload.get("criteria").keys()) if isinstance(payload.get("criteria"), dict) else None,
                "inventory": {k: inv[k] for k in ("n_rows", "n_blank_ids", "n_unique_ids", "n_duplicate_ids", "duplicate_ids")},
                "coverage": _text_coverage_from_api_posts(posts),
            }
        )
    inv_all = _id_inventory(all_posts)
    return {
        "source_key": "october5_api_pages",
        "pagination_manifest": pagination,
        "pages": pages,
        "inventory": inv_all,
        "by_id": by_id,
        "candidate_route_by_id": route_by_id,
        "metadata_predicted_by_id": predicted_by_id,
        "metadata_labeled_by_id": labeled_by_id,
        "repeated_ids_across_pages": {rid: files for rid, files in occurrences.items() if len(files) > 1},
        "occurrences_by_id": dict(occurrences_by_id),
        "ids": inv_all["ids"],
        "note": "API TPS/GDS route and metadata.predicted are candidate/source membership, never a human TPS/GDS label. Repeated page hits are retained in occurrences_by_id; by_id is the first page only.",
    }


def load_jsonl_ids(path, source_key: str, *, ids_only: bool) -> dict[str, Any]:
    rows = read_jsonl(path)
    inv = _id_inventory(rows)
    payload: dict[str, Any] = {
        "source_key": source_key,
        "file": str(path),
        "inventory": inv,
        "ids": inv["ids"],
        "ids_only": ids_only,
    }
    if ids_only:
        payload["note"] = "Protected ID list loaded for overlap/partition auditing only. Outcomes were not scored or used for decisions."
        return payload
    by_id = {}
    for row in rows:
        rid = nonempty_reddit_id(row)
        if rid and rid not in by_id:
            by_id[rid] = row
    payload["by_id"] = by_id
    return payload


def snapshot_compact(raw_post: dict[str, Any], source_key: str, source_file: str, capture_time: Any) -> dict[str, Any]:
    title = "" if raw_post.get("title") is None else str(raw_post.get("title") or "")
    body, body_field, body_present_field = extract_post_body(raw_post)
    norm = normalize_comments(raw_post)
    return {
        "source_key": source_key,
        "source_file": source_file,
        "capture_time": capture_time,
        "title_present": bool(title.strip()),
        "title_sha256": sha256_text(title),
        "title_len": len(title),
        "body_field": body_field,
        "body_field_present": body_present_field,
        "body_present": bool(body.strip()),
        "body_sha256": sha256_text(body),
        "body_len": len(body),
        "comments_status": norm["comments_status"],
        "n_stored_comments": norm["n_stored"],
        "n_nonempty_comments": norm["n_nonempty"],
        "n_author_replies_nonempty": norm["n_author_replies_nonempty"],
        "n_community_replies_nonempty": norm["n_community_replies_nonempty"],
        "n_invalid_or_missing_timestamps": norm["n_invalid_or_missing_timestamps"],
        "num_comments_field": raw_post.get("num_comments"),
        "chronological_original_indices": norm["chronological_original_indices"],
    }


def source_file_manifest() -> dict[str, Any]:
    entries = []
    for path in all_input_files():
        entries.append(file_manifest_entry(path))
    missing = [e["path"] for e in entries if not e["exists"]]
    return {
        "n_files": len(entries),
        "n_missing": len(missing),
        "missing": missing,
        "files": entries,
        "package_tarball": file_manifest_entry(PACKAGE_TARBALL, {"pin": "releasetrain-sentiment==0.2.1"}),
    }
