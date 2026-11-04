"""Read-only load and verification of the combined 6,425-ID table."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from full_resource_sentiment_v1.paths import FRESH_500, RESERVED_TEST, WORKSPACE
from full_resource_sentiment_v1.tps_gds_sentiment_analysis.dictionary import (
    EXPECTED_COUNTS,
    FROZEN_DEF,
    HISTORICAL_DEF,
)

DATASET_DIR = WORKSPACE / "results/tps_gds_full_ai_dataset_2026-11-02"
JSONL = DATASET_DIR / "full_dataset.jsonl"
CSV = DATASET_DIR / "full_dataset.csv"
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
    ids: set[str] = set()
    for path in (RESERVED_TEST, FRESH_500):
        for row in load_jsonl(path):
            rid = row.get("reddit_id") or row.get("id")
            if rid:
                ids.add(str(rid))
    return ids


def flatten_row(row: dict[str, Any], lengths: dict[str, dict[str, Any]]) -> dict[str, Any]:
    sent = row.get("sentiment") or {}
    rid = str(row["reddit_id"])
    length = lengths.get(rid) or {}
    title_c = sent.get("title_compound")
    body_c = sent.get("body_compound")
    body_minus = sent.get("body_minus_title")
    if body_minus is None and title_c is not None and body_c is not None:
        body_minus = float(body_c) - float(title_c)
    n_author = sent.get("n_author_replies_nonempty")
    n_community = sent.get("n_community_replies_nonempty")
    n_stored = sent.get("n_stored_comments")
    reply_total = None
    if n_author is not None or n_community is not None:
        reply_total = int(n_author or 0) + int(n_community or 0)
    existing = row.get("decision_status") == "existing_decision" or row.get("decision_source") != (
        "gpt-6.1-terra_ai_classification"
    )
    # Prefer explicit decision_status when present.
    if row.get("decision_status") == "existing_decision":
        existing = True
    elif row.get("decision_status") == "model_decision":
        existing = False
    conf = str(row.get("decision_confidence") or "").lower()
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
        "confidence_band": "high" if conf == "high" else ("lower_or_missing" if conf else "missing"),
        "title_compound": title_c,
        "body_compound": body_c,
        "combined_compound": sent.get("combined_compound"),
        "body_minus_title": body_minus,
        "author_mean": sent.get("author_mean"),
        "community_mean": sent.get("community_mean"),
        "author_shift": sent.get("author_shift"),
        "community_shift": sent.get("community_shift"),
        "first_reply_compound": sent.get("first_reply_compound"),
        "last_reply_compound": sent.get("last_reply_compound"),
        "last_minus_first": sent.get("last_minus_first"),
        "order_slope_normalized": sent.get("order_slope_normalized"),
        "order_eligible": sent.get("order_eligible"),
        "elapsed_hours": sent.get("elapsed_hours"),
        "elapsed_eligible": sent.get("elapsed_eligible"),
        "count_gap": sent.get("count_gap"),
        "title_nonempty": sent.get("title_nonempty"),
        "body_nonempty": sent.get("body_nonempty"),
        "n_author_replies_nonempty": n_author,
        "n_community_replies_nonempty": n_community,
        "n_stored_comments": n_stored,
        "reply_role_total": reply_total,
        "title_len": length.get("title_len"),
        "body_len": length.get("body_len"),
        "title_len_band": length.get("title_len_band"),
        "body_len_band": length.get("body_len_band"),
    }


def load_length_index() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(POST_SCORES):
        out[str(row["reddit_id"])] = {
            "title_len": row.get("title_len"),
            "body_len": row.get("body_len"),
            "title_len_band": row.get("title_len_band"),
            "body_len_band": row.get("body_len_band"),
        }
    return out


def verify_and_load() -> dict[str, Any]:
    jsonl_hash = sha256_file(JSONL)
    csv_hash = sha256_file(CSV)
    discrepancies: list[str] = []
    if jsonl_hash != EXPECTED_COUNTS["full_dataset_jsonl_sha256"]:
        discrepancies.append(f"jsonl hash {jsonl_hash} != expected")
    if csv_hash != EXPECTED_COUNTS["full_dataset_csv_sha256"]:
        discrepancies.append(f"csv hash {csv_hash} != expected")

    raw = load_jsonl(JSONL)
    ids = [str(r["reddit_id"]) for r in raw]
    if len(ids) != len(set(ids)):
        discrepancies.append("duplicate reddit_id")
    if len(ids) != EXPECTED_COUNTS["n_frame"]:
        discrepancies.append(f"n={len(ids)} != {EXPECTED_COUNTS['n_frame']}")

    rel = Counter(str(r.get("final_relevance")) for r in raw)
    for k, v in EXPECTED_COUNTS["relevance"].items():
        if rel.get(k, 0) != v:
            discrepancies.append(f"relevance {k}: {rel.get(k, 0)} != {v}")
    disc = Counter(str(r.get("final_discussion")) for r in raw if r.get("final_relevance") == "relevant")
    for k, v in EXPECTED_COUNTS["relevant_discussion"].items():
        if disc.get(k, 0) != v:
            discrepancies.append(f"discussion {k}: {disc.get(k, 0)} != {v}")
    sdef = Counter(str(r.get("sentiment_definition")) for r in raw)
    for k, v in EXPECTED_COUNTS["sentiment_definition"].items():
        if sdef.get(k, 0) != v:
            discrepancies.append(f"sentiment_definition {k}: {sdef.get(k, 0)} != {v}")

    n_existing = sum(1 for r in raw if r.get("decision_status") == "existing_decision")
    n_new = sum(1 for r in raw if r.get("decision_status") == "model_decision")
    if n_existing != EXPECTED_COUNTS["existing_decisions"]:
        # Fall back to source-based if status naming differs.
        n_existing_src = sum(1 for r in raw if r.get("decision_source") != "gpt-6.1-terra_ai_classification")
        n_new_src = sum(1 for r in raw if r.get("decision_source") == "gpt-6.1-terra_ai_classification")
        if n_existing_src != EXPECTED_COUNTS["existing_decisions"] or n_new_src != EXPECTED_COUNTS["new_model_decisions"]:
            discrepancies.append(
                f"existing/new decisions status={n_existing}/{n_new} source={n_existing_src}/{n_new_src}"
            )
        else:
            n_existing, n_new = n_existing_src, n_new_src
    elif n_new != EXPECTED_COUNTS["new_model_decisions"]:
        discrepancies.append(f"new model decisions {n_new} != {EXPECTED_COUNTS['new_model_decisions']}")

    prot = protected_ids()
    overlap = sorted(set(ids) & prot)
    if overlap:
        discrepancies.append(f"protected overlap n={len(overlap)}")

    lengths = load_length_index()
    rows = [flatten_row(r, lengths) for r in raw]
    # Keep every row in the master analysis table.
    assert len(rows) == len(raw)

    return {
        "ok": not discrepancies,
        "discrepancies": discrepancies,
        "n": len(rows),
        "jsonl_sha256": jsonl_hash,
        "csv_sha256": csv_hash,
        "relevance_counts": dict(rel),
        "relevant_discussion_counts": dict(disc),
        "sentiment_definition_counts": dict(sdef),
        "n_existing_decisions": n_existing,
        "n_new_model_decisions": n_new,
        "protected_overlap": len(overlap),
        "n_with_length_join": sum(1 for r in rows if r["title_len"] is not None or r["body_len"] is not None),
        "rows": rows,
        "frozen_def": FROZEN_DEF,
        "historical_def": HISTORICAL_DEF,
    }
