#!/usr/bin/env python3
"""Independent ID, lineage, protection, and score-coverage checks for the final pool."""

from __future__ import annotations

import hashlib
import json
import csv
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parents[1]
FINAL = BASE / "results/tps_gds_full_ai_dataset_2026-11-02"
COMBINED = BASE / "results/tps_gds_combined_pool_2026-11-02/combined_pool.jsonl"
INVENTORY = BASE / "results/tps_gds_all_source_inventory_2026-11-02/all_source_inventory.jsonl"
RUN = BASE / "restricted/tps_gds_full_classification_2026-11-02"
RESERVED = REPO / "phase3/experiment_v1/data/reserved_test.jsonl"
FRESH = REPO / "phase3/fresh_test_20261007/frozen_joint_labels/all_500_reviewed.jsonl"


def rows(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    final_list = list(rows(FINAL / "full_dataset.jsonl"))
    final = {x["reddit_id"]: x for x in final_list}
    with (FINAL / "full_dataset.csv").open(newline="", encoding="utf-8") as handle:
        flat = list(csv.DictReader(handle))
    assert len(flat) == 6425 and {x["reddit_id"] for x in flat} == set(final)
    inventory = {x["reddit_id"]: x for x in rows(INVENTORY)}
    combined = {x["reddit_id"]: x for x in rows(COMBINED)}
    predictions = {x["reddit_id"]: x for x in rows(RUN / "full_predictions_gpt-6_1-terra.jsonl")}
    protected = {x["reddit_id"] for x in rows(RESERVED)} | {x["reddit_id"] for x in rows(FRESH)}
    assert len(final_list) == len(final) == len(inventory) == 6425
    assert set(final) == set(inventory) and not (set(final) & protected)
    assert len(combined) == 1964 and len(predictions) == 4461
    assert set(combined).isdisjoint(predictions) and set(final) == (set(combined) | set(predictions))
    assert all(x["status"] == "ok" for x in predictions.values())
    assert all(final[rid]["final_relevance"] == x["relevance"] and
               final[rid]["final_discussion"] == (x["discussion_label"] if x["relevance"] == "relevant" else None) and
               final[rid]["decision_source"] == x["label_source"] for rid, x in combined.items())
    assert all(final[rid]["final_relevance"] == x["label"]["relevance"] for rid, x in predictions.items())
    assert all(x["final_relevance"] in {"relevant", "not_relevant", "unclear", "unassessable"} for x in final_list)
    assert all(x["final_discussion"] is None for x in final_list if x["final_relevance"] != "relevant")
    assert all(x["final_discussion"] in {"TPS", "GDS", "Other", "Unclear"} for x in final_list if x["final_relevance"] == "relevant")
    assert all(x["sentiment"] is not None for x in final_list)
    def with_reply(x):
        s = x["sentiment"]
        return any((s.get(k) or 0) > 0 for k in ("n_stored_comments", "n_author_replies_nonempty", "n_community_replies_nonempty"))
    report = {
        "checks": "PASS",
        "unique_post_ids": len(final),
        "protected_id_overlap": len(set(final) & protected),
        "existing_decisions_preserved": len(combined),
        "new_model_decisions": len(predictions),
        "relevance": dict(Counter(x["final_relevance"] for x in final_list)),
        "discussion_among_relevant": dict(Counter(x["final_discussion"] for x in final_list if x["final_relevance"] == "relevant")),
        "gds_with_any_saved_reply": sum(with_reply(x) for x in final_list if x["final_discussion"] == "GDS"),
        "gds_with_author_reply": sum((x["sentiment"].get("n_author_replies_nonempty") or 0) > 0 for x in final_list if x["final_discussion"] == "GDS"),
        "gds_order_eligible": sum(x["sentiment"].get("order_eligible") is True for x in final_list if x["final_discussion"] == "GDS"),
        "sha256": {str(p): sha(p) for p in (FINAL / "full_dataset.jsonl", FINAL / "full_dataset.csv", FINAL / "summary.json", COMBINED, INVENTORY, RUN / "prompt.txt", RUN / "missing_packet.jsonl", RESERVED, FRESH)},
    }
    (FINAL / "VALIDATION_REPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
