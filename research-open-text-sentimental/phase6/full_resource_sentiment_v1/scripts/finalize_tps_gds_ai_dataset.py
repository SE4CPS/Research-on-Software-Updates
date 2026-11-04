#!/usr/bin/env python3
"""Join full Terra predictions into one source-preserving ID-level dataset."""

from __future__ import annotations

import hashlib
import json
import csv
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
INVENTORY = BASE / "results/tps_gds_all_source_inventory_2026-11-02/all_source_inventory.jsonl"
COMBINED = BASE / "results/tps_gds_combined_pool_2026-11-02/combined_pool.jsonl"
RUN = BASE / "restricted/tps_gds_full_classification_2026-11-02"
PRED = RUN / "full_predictions_gpt-6_1-terra.jsonl"
OUT = BASE / "results/tps_gds_full_ai_dataset_2026-11-02"
POST_SCORES = BASE / "results/broad_sentiment_2026-10-22-main/post_scores.jsonl"
THREAD_METRICS = BASE / "results/broad_sentiment_2026-10-22-main/thread_metrics.jsonl"
HISTORICAL_SCORES = BASE / "results/historical_542_original_vs_package.jsonl"

SENTIMENT_FIELDS = (
    "title_compound", "body_compound", "combined_compound", "body_minus_title",
    "author_mean", "community_mean", "author_shift", "community_shift",
    "first_reply_compound", "last_reply_compound", "last_minus_first",
    "order_eligible", "order_slope_normalized", "elapsed_eligible", "elapsed_hours",
    "post_timestamp_valid", "n_author_replies_nonempty", "n_community_replies_nonempty",
    "n_stored_comments", "count_gap", "body_nonempty", "title_nonempty",
)


def rows(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    inventory = {x["reddit_id"]: x for x in rows(INVENTORY)}
    combined = {x["reddit_id"]: x for x in rows(COMBINED)}
    prediction_attempts = list(rows(PRED))
    predictions = {x["reddit_id"]: x for x in prediction_attempts}
    broad_post = {x["reddit_id"]: x for x in rows(POST_SCORES)}
    broad_thread = {x["reddit_id"]: x for x in rows(THREAD_METRICS)}
    historical_scores = {x["reddit_id"]: x for x in rows(HISTORICAL_SCORES)}
    missing = set(inventory) - set(combined)
    assert len(inventory) == 6425 and len(combined) == 1964 and len(missing) == 4461
    assert set(predictions) == missing, (len(predictions), len(missing - set(predictions)))
    records = []
    for rid in sorted(inventory):
        i = inventory[rid]
        c, p = combined.get(rid), predictions.get(rid)
        if c:
            relevance = c["relevance"]
            discussion = c["discussion_label"] if relevance == "relevant" else None
            source = c["label_source"]
            confidence = c["label_confidence"]
            status = "existing_decision"
            reason = None
            model = None
        elif p["status"] == "ok":
            label = p["label"]
            relevance = label["relevance"]
            discussion = label["discussion"] if relevance == "relevant" else None
            if relevance == "relevant" and discussion is None:
                discussion = "Unclear"
            source = "gpt-6.1-terra_ai_classification"
            confidence = label["confidence"]
            status = "model_decision"
            reason = label["reason"]
            model = p.get("model")
        else:
            relevance, discussion = "unresolved", None
            source, confidence, status, reason, model = "gpt-6.1-terra_ai_classification", None, p["status"], None, p.get("model")
        if rid in broad_thread:
            sentiment = {k: broad_thread[rid].get(k, broad_post[rid].get(k)) for k in SENTIMENT_FIELDS}
            sentiment_definition = "frozen_broad_release_train_package_scores"
        elif rid in historical_scores:
            h = historical_scores[rid]["harmonized"]
            sentiment = {
                "title_compound": h.get("title_compound"),
                "body_compound": h.get("body_compound"),
                "combined_compound": h.get("package_post_concat_compound"),
                "author_mean": h.get("author_comment_mean") if h.get("author_comment_mean_is_observed") else None,
                "community_mean": h.get("community_comment_mean") if h.get("community_comment_mean_is_observed") else None,
                "author_shift": h.get("author_shift_nonempty") if h.get("author_shift_eligible") else None,
                "community_shift": h.get("community_shift_nonempty") if h.get("community_shift_eligible") else None,
                "n_author_replies_nonempty": h.get("n_author_comments_nonempty"),
                "n_community_replies_nonempty": h.get("n_community_comments_nonempty"),
                "body_nonempty": h.get("body_nonempty"),
                "title_nonempty": h.get("title_nonempty"),
            }
            sentiment_definition = "historical_542_harmonized_package_comparison"
        else:
            sentiment, sentiment_definition = None, None
        records.append({
            **i,
            "final_relevance": relevance,
            "final_discussion": discussion,
            "decision_source": source,
            "decision_confidence": confidence,
            "decision_status": status,
            "model_version": model,
            "model_reason": reason,
            "sentiment": sentiment,
            "sentiment_definition": sentiment_definition,
            "current_strict_tps_gds_eligible": relevance == "relevant" and discussion in ("TPS", "GDS"),
        })
    summary = {
        "unique_ids": len(records),
        "relevance_counts": dict(Counter(x["final_relevance"] for x in records)),
        "relevant_discussion_counts": dict(Counter(x["final_discussion"] for x in records if x["final_relevance"] == "relevant")),
        "decision_source_counts": dict(Counter(x["decision_source"] for x in records)),
        "new_prediction_status_counts": dict(Counter(x["status"] for x in predictions.values())),
        "new_prediction_cost_usd_all_attempts": sum(x.get("cost_usd", 0) for x in prediction_attempts),
        "new_prediction_retry_count": len(prediction_attempts) - len(predictions),
        "strict_tps_gds_eligible": sum(x["current_strict_tps_gds_eligible"] for x in records),
        "sentiment_source_counts": dict(Counter(x["sentiment_definition"] for x in records)),
        "source_sha256": {str(p): sha(p) for p in (INVENTORY, COMBINED, PRED, RUN / "prompt.txt", RUN / "schema.json", POST_SCORES, THREAD_METRICS, HISTORICAL_SCORES)},
        "note": "One ID-level pool. Model labels are AI-assisted and retain confidence, reason, and model version. Historical binary labels are preserved separately. Unresolved is never converted to Other or not_relevant.",
    }
    OUT.mkdir(exist_ok=True)
    with (OUT / "full_dataset.jsonl").open("w", encoding="utf-8") as handle:
        for x in records:
            handle.write(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n")
    flat_fields = ["reddit_id", "final_relevance", "final_discussion", "decision_source",
                   "decision_confidence", "model_version", "model_reason",
                   "in_combined_decision_pool", "in_historical_542",
                   "in_nonprotected_ai_selected_5686", "historical_binary_label",
                   "sentiment_definition", *SENTIMENT_FIELDS]
    with (OUT / "full_dataset.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=flat_fields)
        writer.writeheader()
        for x in records:
            writer.writerow({k: ((x["sentiment"] or {}).get(k) if k in SENTIMENT_FIELDS else x.get(k))
                             for k in flat_fields})
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
