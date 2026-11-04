"""Broad Reddit-URL scoring pipeline for frozen capture 2026-10-22-main."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

from .analyses_af import run_all_analyses
from .hashing import sha256_file
from .io_utils import iter_jsonl, read_json
from .paths import (
    BOOTSTRAP_SEED,
    BROAD_CAPTURE_DIR,
    BROAD_COMPACT_INDEX,
    BROAD_RELEVANCE_DENOMS,
    BROAD_SENTIMENT_RUN_DIR,
    LOCAL_COMPACT_RECORDS,
    N_BOOTSTRAP,
    PACKAGE_TARBALL,
    WORKSPACE,
)
from .privacy import scan_paths
from .scoring import date_band, is_reddit_url, score_raw_post

PUBLIC_THREAD_DROP = {"timed_compounds", "comment_rows", "skipped_empty_comment_bodies"}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False))
            handle.write("\n")
    os.replace(temp, path)


def git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=WORKSPACE,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def code_fingerprint() -> str:
    files = sorted((WORKSPACE / "src" / "full_resource_sentiment_v1").glob("*.py"))
    files.append(WORKSPACE / "scripts" / "run_broad_sentiment.py")
    digest = hashlib.sha256()
    for path in files:
        if not path.exists():
            continue
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def verify_capture_pages(capture_dir: Path) -> dict[str, Any]:
    manifest = read_json(capture_dir / "manifest.json")
    if not manifest.get("complete"):
        raise ValueError("capture is not complete")
    pages = []
    prior_next = None
    for index, page in enumerate(manifest["pages"], start=1):
        if page["page"] != index or page.get("request_cursor") != prior_next:
            raise ValueError(f"broken page/cursor sequence at {index}")
        path = capture_dir / page["file"]
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != page["sha256"] or len(raw) != page["bytes"]:
            raise ValueError(f"page hash mismatch {page['file']}")
        pages.append(
            {
                "page": index,
                "file": page["file"],
                "sha256": digest,
                "bytes": len(raw),
                "rows": page["rows"],
            }
        )
        prior_next = page.get("next_cursor")
    return {
        "capture_id": manifest["capture_id"],
        "n_pages": len(pages),
        "pages": pages,
        "raw_manifest_sha256": sha256_file(capture_dir / "manifest.json"),
        "started_utc": manifest.get("started_utc"),
        "finished_utc": manifest.get("finished_utc"),
    }


def load_compact_index(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for row in iter_jsonl(path):
        rid = str(row["reddit_id"])
        if rid in rows:
            raise ValueError(f"duplicate compact reddit_id {rid}")
        rows[rid] = row
    return rows


def load_local_development_labels(path: Path) -> dict[str, dict[str, Any]]:
    """Read development labels only. Protected outcomes are never copied."""
    out: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("protected_outcomes_copied"):
                raise ValueError(f"protected outcomes copied for {record.get('reddit_id')}")
            memberships = set(record.get("source_memberships") or [])
            rid = str(record["reddit_id"])
            flags = {
                "source_memberships": sorted(memberships),
                "development_membership": "development_v2" in memberships,
                "reserved_membership": "reserved_test" in memberships,
                "fresh_membership": "fresh_500" in memberships,
            }
            if flags["reserved_membership"] or flags["fresh_membership"]:
                flags["s2_in_scope"] = None
                flags["s2_target"] = None
                flags["f_eligible"] = False
                flags["f_exclusion"] = "protected_membership"
            elif "development_v2" in memberships:
                labels = record.get("labels") or {}
                in_scope = labels.get("s2_in_scope") is True
                target = labels.get("s2_target")
                flags["s2_in_scope"] = bool(labels.get("s2_in_scope") is True)
                if in_scope:
                    if target not in (0, 1):
                        raise ValueError(f"in-scope development row missing target: {rid}")
                    flags["s2_target"] = int(target)
                    flags["f_eligible"] = True
                    flags["f_exclusion"] = None
                else:
                    flags["s2_target"] = None
                    flags["f_eligible"] = False
                    flags["f_exclusion"] = "out_of_scope_or_unlabeled"
            else:
                flags["s2_in_scope"] = None
                flags["s2_target"] = None
                flags["f_eligible"] = False
                flags["f_exclusion"] = "no_reviewed_development"
            out[rid] = flags
    return out


def f_fields(local: dict[str, Any] | None, compact: dict[str, Any]) -> dict[str, Any]:
    memberships = list(compact.get("local_memberships") or [])
    if local:
        memberships = list(local.get("source_memberships") or memberships)
    memberships_set = set(memberships)
    reserved = "reserved_test" in memberships_set
    fresh = "fresh_500" in memberships_set
    development = "development_v2" in memberships_set
    if local is None:
        return {
            "development_membership": development,
            "reserved_membership": reserved,
            "fresh_membership": fresh,
            "s2_in_scope": None,
            "s2_target": None,
            "f_eligible": False,
            "f_exclusion": "no_reviewed_development" if not development else "missing_local_record",
        }
    if local["reserved_membership"] or local["fresh_membership"] or reserved or fresh:
        if local.get("f_eligible"):
            raise ValueError("protected membership cannot be F-eligible")
        return {
            "development_membership": development,
            "reserved_membership": True if reserved or local["reserved_membership"] else False,
            "fresh_membership": True if fresh or local["fresh_membership"] else False,
            "s2_in_scope": None,
            "s2_target": None,
            "f_eligible": False,
            "f_exclusion": "protected_membership",
        }
    return {
        "development_membership": development,
        "reserved_membership": False,
        "fresh_membership": False,
        "s2_in_scope": local.get("s2_in_scope"),
        "s2_target": local.get("s2_target") if local.get("f_eligible") else None,
        "f_eligible": bool(local.get("f_eligible")),
        "f_exclusion": local.get("f_exclusion"),
    }


def build_output_rows(
    raw_post: dict[str, Any],
    *,
    compact: dict[str, Any],
    local: dict[str, Any] | None,
    capture_page: int,
    capture_row_index: int,
    capture_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    rid = str(raw_post["redditId"]).strip()
    scored = score_raw_post(raw_post)
    if compact["author_replies"] != scored["n_author_replies_nonempty"]:
        raise ValueError(f"author-reply count mismatch for {rid}")
    if compact["community_replies"] != scored["n_community_replies_nonempty"]:
        raise ValueError(f"community-reply count mismatch for {rid}")
    if compact["nonempty_comments"] != scored["n_nonempty_comments"]:
        raise ValueError(f"nonempty-comment count mismatch for {rid}")
    if bool(compact["reported_more_than_stored"]) != bool(scored["count_gap"]):
        raise ValueError(f"count-gap mismatch for {rid}")
    if bool(compact["body_nonempty"]) != bool(scored["body_nonempty"]):
        raise ValueError(f"body-nonempty mismatch for {rid}")

    source = compact.get("source")
    labels = f_fields(local, compact)
    created_month = compact.get("created_month")
    post = {
        "reddit_id": rid,
        "capture_id": capture_id,
        "capture_page": capture_page,
        "capture_row_index": capture_row_index,
        "source": source,
        "source_is_reddit": str(source).lower() == "reddit",
        "reddit_url": True,
        "created_month": created_month,
        "date_band": date_band(created_month),
        "title_nonempty": scored["title_nonempty"],
        "body_nonempty": scored["body_nonempty"],
        "title_len": scored["title_len"],
        "body_len": scored["body_len"],
        "title_len_band": scored["title_len_band"],
        "body_len_band": scored["body_len_band"],
        "title_compound": None if scored["title"] is None else scored["title"]["compound"],
        "title_pos": None if scored["title"] is None else scored["title"]["pos"],
        "title_neu": None if scored["title"] is None else scored["title"]["neu"],
        "title_neg": None if scored["title"] is None else scored["title"]["neg"],
        "title_label": None if scored["title"] is None else scored["title"]["label"],
        "body_compound": None if scored["body"] is None else scored["body"]["compound"],
        "body_pos": None if scored["body"] is None else scored["body"]["pos"],
        "body_neu": None if scored["body"] is None else scored["body"]["neu"],
        "body_neg": None if scored["body"] is None else scored["body"]["neg"],
        "body_label": None if scored["body"] is None else scored["body"]["label"],
        "combined_compound": None if scored["combined"] is None else scored["combined"]["compound"],
        "combined_pos": None if scored["combined"] is None else scored["combined"]["pos"],
        "combined_neu": None if scored["combined"] is None else scored["combined"]["neu"],
        "combined_neg": None if scored["combined"] is None else scored["combined"]["neg"],
        "combined_label": None if scored["combined"] is None else scored["combined"]["label"],
        "body_minus_title": scored["body_minus_title"],
        "comments_status": scored["comments_status"],
        "n_stored_comments": scored["n_stored_comments"],
        "n_nonempty_comments": scored["n_nonempty_comments"],
        "n_empty_comment_bodies": scored["n_empty_comment_bodies"],
        "n_author_replies_nonempty": scored["n_author_replies_nonempty"],
        "n_community_replies_nonempty": scored["n_community_replies_nonempty"],
        "n_unresolved_replies_nonempty": scored["n_unresolved_replies_nonempty"],
        "count_gap": scored["count_gap"],
        "num_comments_reported": scored["num_comments_reported"],
        "package_version": scored["package_version"],
        "used_api_sentiment_score": False,
        "used_api_positive_score": False,
        **labels,
    }
    comments = []
    for row in scored["comment_rows"]:
        comments.append(
            {
                "comment_key": f"{rid}:{row['original_index']}",
                "reddit_id": rid,
                "capture_page": capture_page,
                **row,
            }
        )
    thread = {
        "reddit_id": rid,
        "capture_id": capture_id,
        "source": source,
        "source_is_reddit": str(source).lower() == "reddit",
        "created_month": created_month,
        "date_band": date_band(created_month),
        "title_compound": post["title_compound"],
        "body_compound": post["body_compound"],
        "combined_compound": post["combined_compound"],
        "body_minus_title": scored["body_minus_title"],
        "title_len_band": scored["title_len_band"],
        "body_len_band": scored["body_len_band"],
        "title_nonempty": scored["title_nonempty"],
        "body_nonempty": scored["body_nonempty"],
        "n_stored_comments": scored["n_stored_comments"],
        "n_nonempty_comments": scored["n_nonempty_comments"],
        "n_empty_comment_bodies": scored["n_empty_comment_bodies"],
        "n_author_replies_nonempty": scored["n_author_replies_nonempty"],
        "n_community_replies_nonempty": scored["n_community_replies_nonempty"],
        "n_unresolved_replies_nonempty": scored["n_unresolved_replies_nonempty"],
        "n_timed_nonempty": scored["n_timed_nonempty"],
        "n_distinct_reply_times": scored["n_distinct_reply_times"],
        "author_mean": scored["author_mean"],
        "community_mean": scored["community_mean"],
        "unresolved_mean": scored["unresolved_mean"],
        "author_shift": scored["author_shift"],
        "community_shift": scored["community_shift"],
        "order_eligible": scored["order_eligible"],
        "elapsed_eligible": scored["elapsed_eligible"],
        "author_reply_eligible": scored["author_reply_eligible"],
        "both_roles_eligible": scored["both_roles_eligible"],
        "first_reply_compound": scored["first_reply_compound"],
        "last_reply_compound": scored["last_reply_compound"],
        "last_minus_first": scored["last_minus_first"],
        "order_slope_normalized": scored["order_slope_normalized"],
        "elapsed_hours": scored["elapsed_hours"],
        "elapsed_band": scored["elapsed_band"],
        "post_timestamp_valid": scored["post_timestamp_valid"],
        "post_to_first_reply_eligible": scored["post_to_first_reply_eligible"],
        "post_to_first_reply_hours": scored["post_to_first_reply_hours"],
        "count_gap": scored["count_gap"],
        "timed_compounds": [row["compound"] for row in scored["comment_rows"] if row["timestamp_valid"]],
        **labels,
    }
    return post, comments, thread


def public_thread_row(thread: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in thread.items() if key not in PUBLIC_THREAD_DROP}


def select_handcheck_ids(threads: list[dict[str, Any]], *, n: int = 30, seed: int = BOOTSTRAP_SEED) -> list[str]:
    import numpy as np

    rng = np.random.default_rng(seed)
    slices = [
        lambda row: row.get("created_month") == "2026-10",
        lambda row: row.get("created_month") != "2026-10",
        lambda row: bool(row.get("count_gap")),
        lambda row: not row.get("count_gap"),
        lambda row: bool(row.get("author_reply_eligible")) and not row.get("both_roles_eligible"),
        lambda row: not row.get("author_reply_eligible") and (row.get("n_community_replies_nonempty") or 0) >= 1,
        lambda row: bool(row.get("both_roles_eligible")),
        lambda row: not row.get("author_reply_eligible") and (row.get("n_community_replies_nonempty") or 0) == 0,
        lambda row: bool(row.get("body_nonempty")),
        lambda row: not row.get("body_nonempty"),
        lambda row: bool(row.get("source_is_reddit")),
        lambda row: not row.get("source_is_reddit"),
    ]
    chosen: list[str] = []
    used: set[str] = set()
    ids = [row["reddit_id"] for row in threads]
    for predicate in slices:
        candidates = [row["reddit_id"] for row in threads if predicate(row) and row["reddit_id"] not in used]
        if not candidates:
            continue
        pick = str(candidates[int(rng.integers(0, len(candidates)))])
        chosen.append(pick)
        used.add(pick)
    remaining = [rid for rid in ids if rid not in used]
    rng.shuffle(remaining)
    for rid in remaining:
        if len(chosen) >= n:
            break
        chosen.append(str(rid))
        used.add(rid)
    return chosen[:n]


def expected_denominators() -> dict[str, Any]:
    payload = read_json(BROAD_RELEVANCE_DENOMS)
    return payload["groups"]


def denominator_report(threads: list[dict[str, Any]]) -> dict[str, Any]:
    def count(pred) -> int:
        return sum(1 for row in threads if pred(row))

    actual = {
        "posts": len(threads),
        "title_nonempty": count(lambda r: r.get("title_nonempty")),
        "body_nonempty": count(lambda r: r.get("body_nonempty")),
        "nonempty_comments_ge1": count(lambda r: (r.get("n_nonempty_comments") or 0) >= 1),
        "author_replies_ge1": count(lambda r: r.get("author_reply_eligible")),
        "both_roles_ge1": count(lambda r: r.get("both_roles_eligible")),
        "order_eligible": count(lambda r: r.get("order_eligible")),
        "elapsed_eligible": count(lambda r: r.get("elapsed_eligible")),
        "count_gap": count(lambda r: r.get("count_gap")),
        "f_eligible": count(lambda r: r.get("f_eligible") is True),
        "f_troubleshooting": count(lambda r: r.get("f_eligible") is True and r.get("s2_target") == 1),
        "f_other": count(lambda r: r.get("f_eligible") is True and r.get("s2_target") == 0),
        "f_body": count(lambda r: r.get("f_eligible") is True and r.get("body_compound") is not None),
        "f_comments": count(lambda r: r.get("f_eligible") is True and (r.get("n_nonempty_comments") or 0) >= 1),
        "f_author": count(lambda r: r.get("f_eligible") is True and r.get("author_reply_eligible")),
        "f_both_roles": count(lambda r: r.get("f_eligible") is True and r.get("both_roles_eligible")),
        "f_order": count(lambda r: r.get("f_eligible") is True and r.get("order_eligible")),
        "protected_in_f": count(
            lambda r: r.get("f_eligible") is True and (r.get("reserved_membership") or r.get("fresh_membership"))
        ),
        "author_mean_null_when_no_author": count(
            lambda r: not r.get("author_reply_eligible") and r.get("author_mean") is None
        ),
        "author_mean_zero_with_no_author": count(
            lambda r: not r.get("author_reply_eligible") and r.get("author_mean") == 0.0
        ),
        "n_empty_comment_bodies": sum(int(r.get("n_empty_comment_bodies") or 0) for r in threads),
        "unique_ids": len({r["reddit_id"] for r in threads}),
    }
    groups = expected_denominators()
    expected = {
        "posts": groups["broad_reddit_url"]["n"],
        "title_nonempty": groups["broad_reddit_url"]["title_nonempty"],
        "body_nonempty": groups["broad_reddit_url"]["body_nonempty"],
        "nonempty_comments_ge1": groups["broad_reddit_url"]["nonempty_comments_ge1"],
        "author_replies_ge1": groups["broad_reddit_url"]["author_replies_ge1"],
        "both_roles_ge1": groups["broad_reddit_url"]["both_roles_ge1"],
        "order_eligible": groups["broad_reddit_url"]["order_eligible"],
        "elapsed_eligible": groups["broad_reddit_url"]["elapsed_eligible"],
        "count_gap": groups["broad_reddit_url"]["reported_more_than_stored"],
        "f_eligible": groups["dev_in_scope"]["n"],
        "f_troubleshooting": groups["dev_in_scope_tps"]["n"],
        "f_other": groups["dev_in_scope_other"]["n"],
        "f_body": groups["dev_in_scope"]["body_nonempty"],
        "f_comments": groups["dev_in_scope"]["nonempty_comments_ge1"],
        "f_author": groups["dev_in_scope"]["author_replies_ge1"],
        "f_both_roles": groups["dev_in_scope"]["both_roles_ge1"],
        "f_order": groups["dev_in_scope"]["order_eligible"],
    }
    mismatches = {key: {"expected": expected[key], "actual": actual[key]} for key in expected if expected[key] != actual[key]}
    return {"actual": actual, "expected": expected, "mismatches": mismatches}


def input_contract_hashes(capture_dir: Path, compact_path: Path, local_path: Path) -> dict[str, str]:
    return {
        "capture_manifest_sha256": sha256_file(capture_dir / "manifest.json"),
        "compact_index_sha256": sha256_file(compact_path),
        "local_compact_sha256": sha256_file(local_path),
        "relevance_denominators_sha256": sha256_file(BROAD_RELEVANCE_DENOMS),
        "package_tarball_sha256": sha256_file(PACKAGE_TARBALL),
        "code_fingerprint": code_fingerprint(),
        "releasetrain_sentiment": version("releasetrain-sentiment"),
        "vaderSentiment": version("vaderSentiment"),
    }


def run_broad_sentiment(
    *,
    capture_dir: Path = BROAD_CAPTURE_DIR,
    compact_path: Path = BROAD_COMPACT_INDEX,
    local_path: Path = LOCAL_COMPACT_RECORDS,
    run_dir: Path = BROAD_SENTIMENT_RUN_DIR,
    validate_frozen_denominators: bool = True,
) -> dict[str, Any]:
    started = time.monotonic()
    started_utc = utcnow()
    run_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = run_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    capture = verify_capture_pages(capture_dir)
    hashes = input_contract_hashes(capture_dir, compact_path, local_path)
    state_path = run_dir / "run_state.json"
    completed: set[int] = set()
    if state_path.exists():
        prior = read_json(state_path)
        if prior.get("status") == "complete":
            if prior.get("input_hashes") == hashes:
                return read_json(run_dir / "manifest.json")
            raise ValueError("refusing to overwrite a completed run with different input/code hashes")
        if prior.get("input_hashes") != hashes:
            raise ValueError("incomplete run exists with different input/code hashes; pass a new run directory")
        completed = set(prior.get("completed_pages") or [])
    compact = load_compact_index(compact_path)
    n_url = sum(1 for row in compact.values() if row.get("reddit_url"))
    if validate_frozen_denominators:
        if len(compact) != 37729:
            raise ValueError(f"compact unique IDs {len(compact)} != 37729")
        if n_url != 37596:
            raise ValueError(f"compact reddit-URL IDs {n_url} != 37596")
    local = load_local_development_labels(local_path)

    posts: list[dict[str, Any]] = []
    comments: list[dict[str, Any]] = []
    threads: list[dict[str, Any]] = []
    seen: set[str] = set()
    counts = {
        "non_reddit_url_skipped": 0,
        "duplicate_reddit_url_skipped": 0,
        "pages_resumed": 0,
        "pages_scored": 0,
        "empty_comment_bodies": 0,
    }
    for page in capture["pages"]:
        part_posts = parts_dir / f"page_{page['page']:05d}_posts.jsonl"
        part_comments = parts_dir / f"page_{page['page']:05d}_comments.jsonl"
        part_threads = parts_dir / f"page_{page['page']:05d}_threads.jsonl"
        if page["page"] in completed and part_posts.exists() and part_threads.exists():
            page_posts = [json.loads(line) for line in part_posts.read_text(encoding="utf-8").splitlines() if line.strip()]
            page_comments = (
                [json.loads(line) for line in part_comments.read_text(encoding="utf-8").splitlines() if line.strip()]
                if part_comments.exists()
                else []
            )
            page_threads = [json.loads(line) for line in part_threads.read_text(encoding="utf-8").splitlines() if line.strip()]
            for row in page_posts:
                if row["reddit_id"] in seen:
                    raise ValueError(f"duplicate resumed id {row['reddit_id']}")
                seen.add(row["reddit_id"])
            posts.extend(page_posts)
            comments.extend(page_comments)
            threads.extend(page_threads)
            counts["pages_resumed"] += 1
            continue
        page_posts = []
        page_comments = []
        page_threads = []
        payload = json.loads((capture_dir / page["file"]).read_text(encoding="utf-8"))
        rows = payload.get("data")
        if not isinstance(rows, list) or len(rows) != page["rows"]:
            raise ValueError(f"page schema/count failure {page['file']}")
        for row_index, raw in enumerate(rows):
            if not isinstance(raw, dict) or not raw.get("redditId"):
                raise ValueError(f"missing Reddit ID on page {page['page']} row {row_index}")
            if not is_reddit_url(raw.get("url")):
                counts["non_reddit_url_skipped"] += 1
                continue
            rid = str(raw["redditId"]).strip()
            if rid in seen:
                counts["duplicate_reddit_url_skipped"] += 1
                continue
            if rid not in compact or not compact[rid].get("reddit_url"):
                raise ValueError(f"reddit-URL post missing from compact index: {rid}")
            seen.add(rid)
            post, comment_rows, thread = build_output_rows(
                raw,
                compact=compact[rid],
                local=local.get(rid),
                capture_page=page["page"],
                capture_row_index=row_index,
                capture_id=capture["capture_id"],
            )
            page_posts.append(post)
            page_comments.extend(comment_rows)
            page_threads.append(thread)
            counts["empty_comment_bodies"] += int(post["n_empty_comment_bodies"])
        atomic_write_jsonl(part_posts, page_posts)
        atomic_write_jsonl(part_comments, page_comments)
        atomic_write_jsonl(part_threads, page_threads)
        posts.extend(page_posts)
        comments.extend(page_comments)
        threads.extend(page_threads)
        completed.add(page["page"])
        counts["pages_scored"] += 1
        atomic_write_json(
            state_path,
            {
                "status": "in_progress",
                "input_hashes": hashes,
                "completed_pages": sorted(completed),
                "n_posts": len(posts),
                "n_comments": len(comments),
                "updated_utc": utcnow(),
            },
        )
        print(f"scored page {page['page']}/{capture['n_pages']} posts={len(posts)} comments={len(comments)}", flush=True)

    posts.sort(key=lambda row: row["reddit_id"])
    comments.sort(key=lambda row: (row["reddit_id"], row["original_index"]))
    threads.sort(key=lambda row: row["reddit_id"])
    if len(posts) != len({row["reddit_id"] for row in posts}):
        raise ValueError("post_scores are not unique by reddit_id")
    if validate_frozen_denominators:
        denoms = denominator_report(threads)
        if denoms["mismatches"]:
            raise ValueError(f"denominator mismatch: {denoms['mismatches']}")
        if denoms["actual"]["protected_in_f"]:
            raise ValueError("protected membership entered F")
        if denoms["actual"]["author_mean_zero_with_no_author"]:
            raise ValueError("author_mean is 0.0 for a thread with no author reply")
        if denoms["actual"]["unique_ids"] != denoms["actual"]["posts"]:
            raise ValueError("non-unique thread reddit_id")
    else:
        denoms = {
            "actual": {
                "posts": len(threads),
                "unique_ids": len({row["reddit_id"] for row in threads}),
                "protected_in_f": sum(
                    1
                    for row in threads
                    if row.get("f_eligible") and (row.get("reserved_membership") or row.get("fresh_membership"))
                ),
                "author_mean_zero_with_no_author": sum(
                    1
                    for row in threads
                    if not row.get("author_reply_eligible") and row.get("author_mean") == 0.0
                ),
            },
            "expected": {},
            "mismatches": {},
        }
        if denoms["actual"]["protected_in_f"]:
            raise ValueError("protected membership entered F")
        if denoms["actual"]["author_mean_zero_with_no_author"]:
            raise ValueError("author_mean is 0.0 for a thread with no author reply")

    analyses = run_all_analyses([dict(row) for row in threads])
    handcheck_ids = select_handcheck_ids(threads)
    handcheck = []
    by_id = {row["reddit_id"]: row for row in threads}
    posts_by_id = {row["reddit_id"]: row for row in posts}
    for rid in handcheck_ids:
        thread = by_id[rid]
        post = posts_by_id[rid]
        handcheck.append(
            {
                "reddit_id": rid,
                "created_month": thread.get("created_month"),
                "count_gap": thread.get("count_gap"),
                "body_nonempty": thread.get("body_nonempty"),
                "source_is_reddit": thread.get("source_is_reddit"),
                "n_author_replies_nonempty": thread.get("n_author_replies_nonempty"),
                "n_community_replies_nonempty": thread.get("n_community_replies_nonempty"),
                "title_compound": post.get("title_compound"),
                "body_compound": post.get("body_compound"),
                "combined_compound": post.get("combined_compound"),
                "author_mean": thread.get("author_mean"),
                "community_mean": thread.get("community_mean"),
            }
        )

    empty_bodies = []
    for row in threads:
        if row.get("n_empty_comment_bodies"):
            empty_bodies.append({"reddit_id": row["reddit_id"], "n_empty_comment_bodies": row["n_empty_comment_bodies"]})

    public_threads = [public_thread_row(row) for row in threads]
    atomic_write_jsonl(run_dir / "post_scores.jsonl", posts)
    atomic_write_jsonl(run_dir / "comment_scores.jsonl", comments)
    atomic_write_jsonl(run_dir / "thread_metrics.jsonl", public_threads)
    for letter, payload in analyses.items():
        atomic_write_json(run_dir / f"analysis_{letter}.json", payload)
    output_hashes = {
        "post_scores.jsonl": sha256_file(run_dir / "post_scores.jsonl"),
        "comment_scores.jsonl": sha256_file(run_dir / "comment_scores.jsonl"),
        "thread_metrics.jsonl": sha256_file(run_dir / "thread_metrics.jsonl"),
        **{f"analysis_{letter}.json": sha256_file(run_dir / f"analysis_{letter}.json") for letter in analyses},
    }
    privacy_hits = scan_paths(
        [
            run_dir / "post_scores.jsonl",
            run_dir / "comment_scores.jsonl",
            run_dir / "thread_metrics.jsonl",
            *[run_dir / f"analysis_{letter}.json" for letter in analyses],
        ]
    )
    if privacy_hits:
        raise ValueError(f"public outputs contain forbidden text/URL fields: {privacy_hits[:20]}")

    elapsed = time.monotonic() - started
    manifest = {
        "status": "awaiting_codex_review",
        "complete": True,
        "capture_id": capture["capture_id"],
        "run_started_utc": started_utc,
        "run_finished_utc": utcnow(),
        "elapsed_seconds": elapsed,
        "seed": BOOTSTRAP_SEED,
        "n_bootstrap": N_BOOTSTRAP,
        "package_version": hashes["releasetrain_sentiment"],
        "vader_version": hashes["vaderSentiment"],
        "adapter": "full_resource_sentiment_v1.adapter.to_package_post",
        "git_revision": git_revision(),
        "input_hashes": hashes,
        "capture_pages": capture["pages"],
        "n_post_scores": len(posts),
        "n_comment_scores": len(comments),
        "n_thread_metrics": len(public_threads),
        "counts": counts,
        "denominators": denoms,
        "empty_saved_comment_bodies": empty_bodies,
        "handcheck_sample_30": handcheck,
        "output_sha256": output_hashes,
        "privacy_scan_hits": privacy_hits,
        "live_api_called": False,
        "protected_outcomes_read": False,
        "validated_by_cursor": False,
        "notes": [
            "Workstream G scoring only. Status is Awaiting Codex review, not self-validated.",
            "Missing replies remain null. Package trajectory_summary zeros were not used as observed means.",
            "API sentiment_score and positiveScore were not substituted.",
        ],
    }
    atomic_write_json(run_dir / "manifest.json", manifest)
    atomic_write_json(run_dir / "hashes.json", output_hashes)
    atomic_write_json(run_dir / "handcheck_sample_30.json", {"seed": BOOTSTRAP_SEED, "n": len(handcheck), "rows": handcheck})
    write_run_docs(run_dir, manifest, analyses, write_workspace_copies=validate_frozen_denominators)
    atomic_write_json(
        state_path,
        {
            "status": "complete",
            "input_hashes": hashes,
            "completed_pages": sorted(completed),
            "finished_utc": manifest["run_finished_utc"],
            "output_sha256": output_hashes,
        },
    )
    return manifest


def write_run_docs(
    run_dir: Path,
    manifest: dict[str, Any],
    analyses: dict[str, Any],
    *,
    write_workspace_copies: bool = True,
) -> None:
    denoms = manifest["denominators"]["actual"]
    expected = manifest["denominators"]["expected"]
    a = analyses["A"]["primary_body_minus_title"]
    b = analyses["B"]["primary_author_mean_minus_combined"]
    c = analyses["C"]["primary_community_minus_author"]
    d = analyses["D"]["primary_last_minus_first"]
    e = analyses["E"]["primary_last_minus_first"]
    f = analyses["F"]["primary_combined_post"]["comparison"]
    run_md = f"""# Broad sentiment scoring run

Capture: `{manifest['capture_id']}`
Status: **Awaiting Codex review** (not self-validated)

All commands are run from:

```
{WORKSPACE}
```

## Command

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/run_broad_sentiment.py
```

Elapsed seconds: {manifest['elapsed_seconds']:.1f}
Package: releasetrain-sentiment=={manifest['package_version']}
VADER: vaderSentiment=={manifest['vader_version']}
Seed: {manifest['seed']}; thread bootstrap resamples: {manifest['n_bootstrap']}
Live API called: {manifest['live_api_called']}
Protected outcomes read: {manifest['protected_outcomes_read']}

## Outputs

Directory: `{run_dir}`

- post_scores.jsonl ({manifest['n_post_scores']} rows)
- comment_scores.jsonl ({manifest['n_comment_scores']} rows; no text/usernames)
- thread_metrics.jsonl ({manifest['n_thread_metrics']} rows)
- analysis_A.json … analysis_F.json
- manifest.json, hashes.json, handcheck_sample_30.json

## Denominators

| Measure | Expected | Actual |
|---|---:|---:|
| Reddit-URL posts | {expected.get('posts')} | {denoms.get('posts')} |
| Nonempty body | {expected.get('body_nonempty')} | {denoms.get('body_nonempty')} |
| Author replies >=1 | {expected.get('author_replies_ge1')} | {denoms.get('author_replies_ge1')} |
| Both roles | {expected.get('both_roles_ge1')} | {denoms.get('both_roles_ge1')} |
| Order eligible | {expected.get('order_eligible')} | {denoms.get('order_eligible')} |
| Elapsed eligible | {expected.get('elapsed_eligible')} | {denoms.get('elapsed_eligible')} |
| F in-scope | {expected.get('f_eligible')} | {denoms.get('f_eligible')} |
| F troubleshooting / other | {expected.get('f_troubleshooting')} / {expected.get('f_other')} | {denoms.get('f_troubleshooting')} / {denoms.get('f_other')} |
"""
    report = f"""# Workstream G execution report

Date: {manifest['run_finished_utc'][:10]}. Executor: Cursor. Status for Codex: **Awaiting Codex review**. This report does not mark Workstream G completed and validated.

Workspace: `{WORKSPACE}`
Run directory: `{run_dir}`

## What was executed

Scored the frozen Reddit-URL frame from capture `{manifest['capture_id']}` with `releasetrain-sentiment=={manifest['package_version']}` via `to_package_post`. Analyses A–F used thread-level equal weight and 2,000 thread-bootstrap intervals, seed {manifest['seed']}. No live API calls. Protected reserved/fresh outcomes were not read. Count-gap posts were retained.

## Commands

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/run_broad_sentiment.py
```

Elapsed: {manifest['elapsed_seconds']:.1f} seconds.
Pages scored this invocation: {manifest['counts']['pages_scored']}; resumed: {manifest['counts']['pages_resumed']}.
Posts: {manifest['n_post_scores']}. Comments scored: {manifest['n_comment_scores']}. Empty bodies skipped: {manifest['counts']['empty_comment_bodies']}.
Non-Reddit URL rows skipped: {manifest['counts']['non_reddit_url_skipped']}. Duplicate Reddit-URL IDs skipped: {manifest['counts']['duplicate_reddit_url_skipped']}.

## A–F point estimates (descriptive only)

These are not paper claims. Codex must independently recompute before acceptance.

- A body−title mean: {a.get('mean')} CI {a.get('mean_ci95')} (n={a.get('n')})
- B author-reply mean − combined post mean: {b.get('mean')} CI {b.get('mean_ci95')} (n={b.get('n')})
- C community mean − author mean: {c.get('mean')} CI {c.get('mean_ci95')} (n={c.get('n')})
- D last−first reply: {d.get('mean')} CI {d.get('mean_ci95')} (n={d.get('n')})
- E last−first among elapsed-eligible: {e.get('mean')} CI {e.get('mean_ci95')} (n={e.get('n')})
- F troubleshooting − other combined post: {f.get('mean_diff')} CI {f.get('mean_diff_ci95')} (n_a={f.get('n_a')}, n_b={f.get('n_b')}); Cliff's delta {f.get('cliffs_delta')}

## Integrity

- Input/page hashes: see `manifest.json` `input_hashes` and `capture_pages`.
- Output SHA-256: `hashes.json`.
- Privacy scan hits: {manifest['privacy_scan_hits']}.
- Git revision: `{manifest.get('git_revision')}`.
- Hand-check 30 IDs: `handcheck_sample_30.json` (no text).
- Empty saved comment bodies: {manifest['empty_saved_comment_bodies']}.

## Limitations

Broad IDs are a dated candidate frame, not human-verified update posts. Analysis F uses only the reviewed in-scope overlap and does not estimate troubleshooting prevalence. Cadence metrics describe observed saved replies only.
"""
    atomic_write_text(run_dir / "RUN.md", run_md)
    atomic_write_text(run_dir / "EXECUTION_REPORT.md", report)
    if write_workspace_copies:
        atomic_write_text(WORKSPACE / "BROAD_SENTIMENT_RUN.md", run_md)
        atomic_write_text(WORKSPACE / "BROAD_SENTIMENT_EXECUTION_REPORT.md", report)
