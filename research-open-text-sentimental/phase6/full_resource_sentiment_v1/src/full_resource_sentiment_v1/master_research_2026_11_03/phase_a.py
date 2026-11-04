"""Phase A: inventory, hashes, protected overlap, text/thread manifest (metadata)."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from full_resource_sentiment_v1.paths import FRESH_500, RESERVED_TEST, WORKSPACE
from full_resource_sentiment_v1.master_research_2026_11_03.common import (
    CAPTURE_DIR,
    COMMENT_SCORES,
    DATASET_CSV,
    DATASET_JSONL,
    EXPECTED_CSV,
    EXPECTED_JSONL,
    FROZEN_DEF,
    OUT_ROOT,
    POST_SCORES,
    RESTRICTED_ROOT,
    STAGE1_DIR,
    THREAD_METRICS,
    load_jsonl,
    sha256_file,
    sha256_text,
    utcnow,
    write_json,
    write_jsonl,
)


def protected_ids() -> set[str]:
    out: set[str] = set()
    for path in (RESERVED_TEST, FRESH_500):
        for row in load_jsonl(path):
            rid = row.get("reddit_id") or row.get("id")
            if rid:
                out.add(str(rid))
    return out


def run_inventory() -> dict[str, Any]:
    rows = load_jsonl(DATASET_JSONL)
    ids = [str(r["reddit_id"]) for r in rows]
    discrepancies = []
    if len(ids) != 6425 or len(set(ids)) != 6425:
        discrepancies.append(f"id count {len(ids)} unique {len(set(ids))}")
    jh, ch = sha256_file(DATASET_JSONL), sha256_file(DATASET_CSV)
    if jh != EXPECTED_JSONL:
        discrepancies.append("jsonl hash mismatch")
    if ch != EXPECTED_CSV:
        discrepancies.append("csv hash mismatch")
    prot = protected_ids()
    overlap = sorted(set(ids) & prot)
    if overlap:
        discrepancies.append(f"protected overlap {len(overlap)}")

    by_source = Counter(
        (r.get("final_relevance"), r.get("final_discussion"), r.get("decision_source")) for r in rows
    )
    reply: dict[str, Any] = {}
    for label in ("TPS", "GDS", "Other", "Unclear"):
        q = [r for r in rows if r.get("final_relevance") == "relevant" and r.get("final_discussion") == label]
        sent = [(r.get("sentiment") or {}) for r in q]
        reply[label] = {
            "n": len(q),
            "author_ge1": sum(int(s.get("n_author_replies_nonempty") or 0) >= 1 for s in sent),
            "author_ge4": sum(int(s.get("n_author_replies_nonempty") or 0) >= 4 for s in sent),
            "community_ge1": sum(int(s.get("n_community_replies_nonempty") or 0) >= 1 for s in sent),
            "frozen_def": sum(1 for r in q if r.get("sentiment_definition") == FROZEN_DEF),
        }

    out = {
        "utc": utcnow(),
        "ok": not discrepancies,
        "discrepancies": discrepancies,
        "n": len(rows),
        "unique": len(set(ids)),
        "jsonl_sha256": jh,
        "csv_sha256": ch,
        "protected_overlap": len(overlap),
        "sentiment_definition": dict(Counter(str(r.get("sentiment_definition")) for r in rows)),
        "decision_status": dict(Counter(str(r.get("decision_status")) for r in rows)),
        "counts_by_relevance_purpose_source": [
            {"relevance": a, "purpose": b, "source": c, "n": v} for (a, b, c), v in sorted(by_source.items())
        ],
        "reply_coverage": reply,
        "stage1_dir_exists": STAGE1_DIR.exists(),
        "input_hashes": {
            "post_scores": sha256_file(POST_SCORES),
            "thread_metrics": sha256_file(THREAD_METRICS),
            "comment_scores": sha256_file(COMMENT_SCORES),
        },
        "interpretation": "Availability and provenance only; not validation or prevalence.",
    }
    dest = OUT_ROOT / "phase_a_inventory"
    dest.mkdir(parents=True, exist_ok=True)
    write_json(dest / "INVENTORY.json", out)
    return out


def build_text_thread_manifest() -> dict[str, Any]:
    """Build restricted per-ID text hashes + public metadata (no raw text in public)."""
    frame = {str(r["reddit_id"]): r for r in load_jsonl(DATASET_JSONL)}
    comments_by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(COMMENT_SCORES):
        rid = str(row["reddit_id"])
        if rid in frame:
            comments_by[rid].append(row)

    # Title/body from capture pages (hash only publicly)
    texts: dict[str, dict[str, Any]] = {}
    for path in sorted(CAPTURE_DIR.glob("page_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("data") or []:
            rid = str(row.get("redditId") or "")
            if rid not in frame:
                continue
            title = "" if row.get("title") is None else str(row.get("title"))
            body = "" if row.get("author_description") is None else str(row.get("author_description"))
            prev = texts.get(rid)
            if prev is None or len(body) > len(prev["body"]):
                texts[rid] = {
                    "title": title,
                    "body": body,
                    "snapshot": f"broad_capture:{path.name}",
                    "n_raw_comments": len(row.get("comments") or []) if isinstance(row.get("comments"), list) else 0,
                    "num_comments_reported": row.get("num_comments"),
                }

    public_rows = []
    restricted_rows = []
    for rid, meta in frame.items():
        doc = texts.get(rid) or {"title": "", "body": "", "snapshot": "missing_capture", "n_raw_comments": 0}
        title, body = doc["title"], doc["body"]
        clist = sorted(
            comments_by.get(rid, []),
            key=lambda c: (
                c.get("timestamp_epoch_seconds") is None,
                c.get("timestamp_epoch_seconds") or 0,
                c.get("chronological_index") or 0,
            ),
        )
        roles = Counter(str(c.get("role")) for c in clist)
        public_rows.append(
            {
                "reddit_id": rid,
                "final_relevance": meta.get("final_relevance"),
                "final_discussion": meta.get("final_discussion"),
                "decision_source": meta.get("decision_source"),
                "sentiment_definition": meta.get("sentiment_definition"),
                "title_sha256": sha256_text(title),
                "body_sha256": sha256_text(body),
                "title_chars": len(title),
                "body_chars": len(body),
                "title_nonempty": bool(title.strip()),
                "body_nonempty": bool(body.strip()),
                "snapshot": doc.get("snapshot"),
                "n_scored_comments": len(clist),
                "n_author_scored": roles.get("author", 0),
                "n_community_scored": roles.get("community", 0),
                "n_timed_valid": sum(1 for c in clist if c.get("timestamp_valid")),
                "count_gap": (meta.get("sentiment") or {}).get("count_gap"),
            }
        )
        restricted_rows.append(
            {
                "reddit_id": rid,
                "title": title,
                "body": body,
                "snapshot": doc.get("snapshot"),
                "title_sha256": sha256_text(title),
                "body_sha256": sha256_text(body),
                "comments_meta": [
                    {
                        "comment_key": c.get("comment_key"),
                        "role": c.get("role"),
                        "chronological_index": c.get("chronological_index"),
                        "timestamp_valid": c.get("timestamp_valid"),
                        "compound": c.get("compound"),
                    }
                    for c in clist
                ],
            }
        )

    dest = OUT_ROOT / "phase_a_inventory"
    write_jsonl(dest / "text_thread_manifest_public.jsonl", public_rows)
    write_jsonl(RESTRICTED_ROOT / "text_thread_manifest_restricted.jsonl", restricted_rows)
    summary = {
        "utc": utcnow(),
        "n_public": len(public_rows),
        "n_with_capture_text": sum(1 for r in public_rows if r["snapshot"] != "missing_capture"),
        "n_missing_capture": sum(1 for r in public_rows if r["snapshot"] == "missing_capture"),
        "n_with_scored_comments": sum(1 for r in public_rows if r["n_scored_comments"] > 0),
        "public_manifest_sha256": sha256_file(dest / "text_thread_manifest_public.jsonl"),
        "restricted_manifest_sha256": sha256_file(RESTRICTED_ROOT / "text_thread_manifest_restricted.jsonl"),
        "note": "Raw title/body only in restricted directory; public file has hashes/lengths/roles.",
    }
    write_json(dest / "MANIFEST_SUMMARY.json", summary)
    return summary
