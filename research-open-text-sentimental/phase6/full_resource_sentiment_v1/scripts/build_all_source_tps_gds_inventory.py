#!/usr/bin/env python3
"""One ID-level inventory of every existing TPS/GDS research source.

Rows without current strict TPS/GDS decisions stay unlabeled; old binary labels
and AI relevance selections are retained as distinct fields on the same row.
Protected membership files contribute IDs only and are never read for outcomes.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parents[1]
COMBINED = BASE / "results/tps_gds_combined_pool_2026-11-02/combined_pool.jsonl"
HISTORICAL = REPO / "tps_gds_classification/data/updated_labeled_dataset_unique.csv"
AI_SELECTED = BASE / "results/ai_relevance_exploratory_full_2026-10-29/exploratory_ai_selected_cohort.jsonl"
RESERVED = REPO / "phase3/experiment_v1/data/reserved_test.jsonl"
FRESH = REPO / "phase3/fresh_test_20261007/frozen_joint_labels/all_500_reviewed.jsonl"
OUT = BASE / "results/tps_gds_all_source_inventory_2026-11-02"


def rows(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    combined = {x["reddit_id"]: x for x in rows(COMBINED)}
    with HISTORICAL.open(newline="", encoding="utf-8") as handle:
        historical = {x["reddit_id"]: x for x in csv.DictReader(handle)}
    selected = {x["reddit_id"]: x for x in rows(AI_SELECTED) if x["protected_membership"] is False}
    reserved = {x["reddit_id"] for x in rows(RESERVED)}
    fresh = {x["reddit_id"] for x in rows(FRESH)}
    protected = reserved | fresh
    assert len(combined) == 1964 and len(historical) == 542 and len(selected) == 5686
    assert not (set(combined) & protected) and not (set(historical) & protected) and not (set(selected) & protected)
    union = set(combined) | set(historical) | set(selected)
    assert len(union) == 6425, len(union)
    assert len(set(combined) & set(historical)) == 19
    assert len(set(combined) & set(selected)) == 1748
    assert len(set(historical) & set(selected)) == 14

    output = []
    for rid in sorted(union):
        c, h, s = combined.get(rid), historical.get(rid), selected.get(rid)
        old = None if h is None else ("historical_TPS" if h["corrected_label"] == "1" else "historical_broad_GDS")
        record = {
            "reddit_id": rid,
            "in_combined_decision_pool": c is not None,
            "in_historical_542": h is not None,
            "in_nonprotected_ai_selected_5686": s is not None,
            "relevance_status": c["relevance"] if c else ("ai_predicted_relevant_unreviewed" if s else "historical_scope_unreconciled"),
            "current_discussion_label": c["discussion_label"] if c else None,
            "current_label_source": c["label_source"] if c else None,
            "historical_binary_label": old,
            "strict_tps_gds_eligible": bool(c and c["relevance"] == "relevant" and c["discussion_label"] in ("TPS", "GDS")),
            "selection_stratum": s["stratum"] if s else None,
        }
        output.append(record)
    summary = {
        "unique_ids": len(output),
        "source_counts": {"combined": len(combined), "historical": len(historical), "ai_selected_nonprotected": len(selected)},
        "overlaps": {
            "combined_historical": len(set(combined) & set(historical)),
            "combined_ai_selected": len(set(combined) & set(selected)),
            "historical_ai_selected": len(set(historical) & set(selected)),
            "all_three": len(set(combined) & set(historical) & set(selected)),
        },
        "strict_tps_gds_eligible": sum(x["strict_tps_gds_eligible"] for x in output),
        "current_discussion_label_counts": dict(Counter(x["current_discussion_label"] for x in output if x["relevance_status"] == "relevant")),
        "historical_binary_counts": dict(Counter(x["historical_binary_label"] for x in output if x["historical_binary_label"])),
        "without_current_discussion_decision": sum(x["current_discussion_label"] is None for x in output),
        "source_sha256": {str(p): sha(p) for p in (COMBINED, HISTORICAL, AI_SELECTED, RESERVED, FRESH)},
        "note": "One row per Reddit ID. Historical broad GDS is not a strict GDS decision. AI-selected relevance is not a TPS/GDS decision. Protected files were read for IDs only, never outcomes.",
    }
    OUT.mkdir(exist_ok=True)
    with (OUT / "all_source_inventory.jsonl").open("w", encoding="utf-8") as handle:
        for record in output:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
