"""Read-only load/join for Stage-1 corrected VADER analysis."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from full_resource_sentiment_v1.paths import FRESH_500, RESERVED_TEST, WORKSPACE
from full_resource_sentiment_v1.tps_gds_sentiment_v2.dictionary import (
    BODY_BANDS,
    EXPECTED,
    FROZEN_DEF,
    HISTORICAL_DEF,
    TITLE_BANDS,
)

DATASET_JSONL = WORKSPACE / "results/tps_gds_full_ai_dataset_2026-11-02/full_dataset.jsonl"
DATASET_CSV = WORKSPACE / "results/tps_gds_full_ai_dataset_2026-11-02/full_dataset.csv"
POST_SCORES = WORKSPACE / "results/broad_sentiment_2026-10-22-main/post_scores.jsonl"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def protected_ids() -> set[str]:
    out: set[str] = set()
    for path in (RESERVED_TEST, FRESH_500):
        for row in load_jsonl(path):
            rid = row.get("reddit_id") or row.get("id")
            if rid:
                out.add(str(rid))
    return out


def body_band(n: int | None) -> str | None:
    if n is None:
        return None
    for name, pred in BODY_BANDS:
        if pred(int(n)):
            return name
    return None


def title_band(n: int | None) -> str | None:
    if n is None:
        return None
    for name, pred in TITLE_BANDS:
        if pred(int(n)):
            return name
    return None


def reply_bucket(n_author: int | None, n_community: int | None, n_stored: int | None) -> str:
    total = None
    if n_author is not None or n_community is not None:
        total = int(n_author or 0) + int(n_community or 0)
    elif n_stored is not None:
        total = int(n_stored)
    if total is None:
        return "unknown"
    if total <= 0:
        return "0"
    if total == 1:
        return "1"
    if total <= 3:
        return "2_3"
    if total <= 10:
        return "4_10"
    return "11_plus"


def flatten(row: dict[str, Any], lengths: dict[str, dict[str, Any]]) -> dict[str, Any]:
    sent = row.get("sentiment") or {}
    rid = str(row["reddit_id"])
    length = lengths.get(rid) or {}
    title_len = length.get("title_len")
    body_len = length.get("body_len")
    # Prefer post_scores lengths; fall back unavailable.
    if title_len is None and sent.get("title_nonempty") and length.get("title_len") is None:
        title_len = None
    n_author = sent.get("n_author_replies_nonempty")
    n_community = sent.get("n_community_replies_nonempty")
    author_mean = sent.get("author_mean")
    combined = sent.get("combined_compound")
    post_to_author = None
    if author_mean is not None and combined is not None:
        post_to_author = float(author_mean) - float(combined)

    existing = row.get("decision_status") == "existing_decision"
    if row.get("decision_status") not in {"existing_decision", "model_decision"}:
        existing = row.get("decision_source") != "gpt-6.1-terra_ai_classification"

    return {
        "reddit_id": rid,
        "final_relevance": row.get("final_relevance"),
        "final_discussion": row.get("final_discussion"),
        "decision_source": row.get("decision_source"),
        "decision_status": row.get("decision_status"),
        "decision_confidence": row.get("decision_confidence"),
        "selection_stratum": row.get("selection_stratum"),
        "sentiment_definition": row.get("sentiment_definition"),
        "is_existing_decision": bool(existing),
        "is_new_model_decision": row.get("decision_source") == "gpt-6.1-terra_ai_classification",
        "title_compound": sent.get("title_compound"),
        "body_compound": sent.get("body_compound"),
        "combined_compound": combined,
        "author_mean": author_mean,
        "community_mean": sent.get("community_mean"),
        "author_shift": sent.get("author_shift"),
        "original_post_to_author_mean": post_to_author,
        "last_minus_first": sent.get("last_minus_first"),
        "order_slope_normalized": sent.get("order_slope_normalized"),
        "elapsed_hours": sent.get("elapsed_hours"),
        "order_eligible": sent.get("order_eligible"),
        "elapsed_eligible": sent.get("elapsed_eligible"),
        "count_gap": sent.get("count_gap"),
        "title_nonempty": sent.get("title_nonempty"),
        "body_nonempty": sent.get("body_nonempty"),
        "n_author_replies_nonempty": n_author,
        "n_community_replies_nonempty": n_community,
        "n_stored_comments": sent.get("n_stored_comments"),
        "title_len": title_len,
        "body_len": body_len,
        "body_band": body_band(int(body_len) if body_len is not None else None),
        "title_band": title_band(int(title_len) if title_len is not None else None),
        "reply_bucket": reply_bucket(n_author, n_community, sent.get("n_stored_comments")),
        "author_shift_eligible": int(n_author or 0) >= 4 and sent.get("author_shift") is not None,
        "post_to_author_eligible": post_to_author is not None,
    }


def load_lengths() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(POST_SCORES):
        out[str(row["reddit_id"])] = {
            "title_len": row.get("title_len"),
            "body_len": row.get("body_len"),
        }
    return out


def verify_and_load() -> dict[str, Any]:
    jsonl_hash = sha256_file(DATASET_JSONL)
    csv_hash = sha256_file(DATASET_CSV)
    discrepancies: list[str] = []
    if jsonl_hash != EXPECTED["jsonl_sha256"]:
        discrepancies.append(f"jsonl hash mismatch: {jsonl_hash}")
    if csv_hash != EXPECTED["csv_sha256"]:
        discrepancies.append(f"csv hash mismatch: {csv_hash}")

    raw = load_jsonl(DATASET_JSONL)
    ids = [str(r["reddit_id"]) for r in raw]
    if len(ids) != len(set(ids)):
        discrepancies.append("duplicate reddit_id")
    if len(ids) != EXPECTED["n_frame"]:
        discrepancies.append(f"n={len(ids)} != {EXPECTED['n_frame']}")

    sdef = Counter(str(r.get("sentiment_definition")) for r in raw)
    if sdef.get(FROZEN_DEF, 0) != EXPECTED["sentiment_frozen"]:
        discrepancies.append(f"frozen def count {sdef.get(FROZEN_DEF)}")
    if sdef.get(HISTORICAL_DEF, 0) != EXPECTED["sentiment_historical"]:
        discrepancies.append(f"historical def count {sdef.get(HISTORICAL_DEF)}")

    overlap = sorted(set(ids) & protected_ids())
    if overlap:
        discrepancies.append(f"protected overlap n={len(overlap)}")

    lengths = load_lengths()
    rows = [flatten(r, lengths) for r in raw]
    frozen_rel = [
        r
        for r in rows
        if r["final_relevance"] == "relevant" and r["sentiment_definition"] == FROZEN_DEF
    ]
    n_tps = sum(1 for r in frozen_rel if r["final_discussion"] == "TPS")
    n_gds = sum(1 for r in frozen_rel if r["final_discussion"] == "GDS")
    if n_tps != EXPECTED["frozen_relevant_tps"] or n_gds != EXPECTED["frozen_relevant_gds"]:
        discrepancies.append(f"frozen relevant TPS/GDS {n_tps}/{n_gds}")

    return {
        "ok": not discrepancies,
        "discrepancies": discrepancies,
        "n": len(rows),
        "jsonl_sha256": jsonl_hash,
        "csv_sha256": csv_hash,
        "post_scores_sha256": sha256_file(POST_SCORES),
        "sentiment_definition_counts": dict(sdef),
        "protected_overlap": len(overlap),
        "n_frozen_relevant_tps": n_tps,
        "n_frozen_relevant_gds": n_gds,
        "n_with_body_len": sum(1 for r in rows if r["body_len"] is not None),
        "rows": rows,
    }
