#!/usr/bin/env python3
"""Build one Reddit-ID-level research pool from saved TPS/GDS decisions.

The 179-case decisions are Manu's review entered with AI assistance. The
100-case pilot is Codex's AI-assisted review. Provenance remains explicit.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parents[1]
DEV = REPO / "phase3/revised_development_v2/development_v2.jsonl"
SCORES = BASE / "results/broad_sentiment_2026-10-22-main/post_scores.jsonl"
THREADS = BASE / "results/broad_sentiment_2026-10-22-main/thread_metrics.jsonl"
REVIEW_ARCHIVE = BASE / "restricted/tps_gds_179_manu_confirmed_2026-11-02"
A = REVIEW_ARCHIVE / "assistant_review_A_draft.csv"
B = REVIEW_ARCHIVE / "assistant_review_B_draft.csv"
PILOT = BASE / "results/tps_gds_ai_pilot_2026-11-02"
OUT = BASE / "results/tps_gds_combined_pool_2026-11-02"
BASE_LABELS = {
    "technical_troubleshooting": "TPS",
    "general_dissatisfaction": "GDS",
    "other": "Other",
}


def jsonl(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def csvrows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean_ci(gds, tps, seed=20261102, nboot=5000):
    if not gds or not tps:
        return None
    rng = random.Random(seed)
    boot = []
    for _ in range(nboot):
        a = sum(gds[rng.randrange(len(gds))] for _ in gds) / len(gds)
        b = sum(tps[rng.randrange(len(tps))] for _ in tps) / len(tps)
        boot.append(a - b)
    boot.sort()
    return {"gds_minus_tps": statistics.mean(gds) - statistics.mean(tps),
            "ci95": [boot[125], boot[4874]], "gds_n": len(gds), "tps_n": len(tps)}


def main():
    development = {x["reddit_id"]: x for x in jsonl(DEV)}
    scores = {x["reddit_id"]: x for x in jsonl(SCORES)}
    threads = {x["reddit_id"]: x for x in jsonl(THREADS)}
    reviews_a, reviews_b = csvrows(A), csvrows(B)
    assert len(reviews_a) == len(reviews_b) == 179
    assert [x["reddit_id"] for x in reviews_a] == [x["reddit_id"] for x in reviews_b]
    assert len({x["reddit_id"] for x in reviews_a}) == 179
    for x, y in zip(reviews_a, reviews_b):
        assert x["proposed_label"] == y["proposed_label"]
        assert x["title"] == y["title"] == development[x["reddit_id"]]["title"]
        assert development[x["reddit_id"]]["discussion_type"] == "other"
        assert scores[x["reddit_id"]]["f_eligible"]
        assert x["proposed_label"] in {"TPS", "GDS", "Other", "Unclear"}
    override = {x["reddit_id"]: x["proposed_label"] for x in reviews_a}

    pilot_ids, pilot_labels = csvrows(PILOT / "sample_ids.csv"), csvrows(PILOT / "ai_assisted_labels.csv")
    assert len(pilot_ids) == len(pilot_labels) == 100
    assert [x["case_code"] for x in pilot_ids] == [x["case_code"] for x in pilot_labels]
    assert len({x["reddit_id"] for x in pilot_ids}) == 100

    records = {}
    for rid, score in scores.items():
        if not score["f_eligible"]:
            continue
        assert score["reserved_membership"] is False and score["fresh_membership"] is False
        label = override.get(rid, BASE_LABELS[development[rid]["discussion_type"]])
        records[rid] = {
            "reddit_id": rid, "relevance": "relevant", "discussion_label": label,
            "label_source": "manu_review_ai_assisted_entry" if rid in override else "original_f_review",
            "original_f_label": BASE_LABELS[development[rid]["discussion_type"]],
            "case_code": None, "selection_arm": "reviewed_f_set", "label_confidence": None,
        }
    assert len(records) == 1864
    for x, lab in zip(pilot_ids, pilot_labels):
        rid = x["reddit_id"]
        assert rid not in records
        assert scores[rid]["reserved_membership"] is False and scores[rid]["fresh_membership"] is False
        assert lab["relevance_label"] in {"relevant", "not_relevant", "unclear"}
        assert lab["discussion_type"] in {"TPS", "GDS", "Other", "Unclear"}
        records[rid] = {
            "reddit_id": rid, "relevance": lab["relevance_label"],
            "discussion_label": lab["discussion_type"],
            "label_source": "codex_ai_assisted_pilot", "original_f_label": None,
            "case_code": x["case_code"], "selection_arm": x["selection_arm"],
            "label_confidence": lab["confidence"],
        }
    assert len(records) == 1964

    for rid, row in records.items():
        score, thread = scores[rid], threads[rid]
        row.update({
            "date_band": score["date_band"],
            "title_compound": score["title_compound"],
            "body_compound": score["body_compound"] if score["body_nonempty"] else None,
            "combined_compound": score["combined_compound"],
            "n_comments": score["n_nonempty_comments"],
            "n_author_replies": score["n_author_replies_nonempty"],
            "n_community_replies": score["n_community_replies_nonempty"],
            "post_timestamp_valid": thread["post_timestamp_valid"],
            "order_eligible": thread["order_eligible"],
            "author_mean": thread["author_mean"] if thread["author_reply_eligible"] else None,
            "community_mean": thread["community_mean"] if score["n_community_replies_nonempty"] else None,
            "first_reply_compound": thread["first_reply_compound"] if score["n_nonempty_comments"] else None,
            "last_minus_first": thread["last_minus_first"] if thread["order_eligible"] else None,
            "order_slope_normalized": thread["order_slope_normalized"] if thread["order_eligible"] else None,
            "author_shift": thread["author_shift"],
            "community_shift": thread["community_shift"],
            "elapsed_hours": thread["elapsed_hours"] if thread["elapsed_eligible"] else None,
        })

    relevant = [x for x in records.values() if x["relevance"] == "relevant"]
    assert len(relevant) == 1920
    counts = Counter(x["discussion_label"] for x in relevant)
    assert counts == {"TPS": 1682, "GDS": 76, "Other": 157, "Unclear": 5}, counts
    def cover(label):
        rows = [x for x in relevant if x["discussion_label"] == label]
        return {
            "n": len(rows),
            "with_comments": sum(x["n_comments"] > 0 for x in rows),
            "with_author_replies": sum(x["n_author_replies"] > 0 for x in rows),
            "with_community_replies": sum(x["n_community_replies"] > 0 for x in rows),
            "order_eligible": sum(bool(x["order_eligible"]) for x in rows),
            "source_breakdown": dict(Counter(x["label_source"] for x in rows)),
        }
    measures = {}
    for field in ("title_compound", "body_compound", "combined_compound", "author_mean", "community_mean",
                  "first_reply_compound", "last_minus_first", "order_slope_normalized", "author_shift", "community_shift"):
        g = [float(x[field]) for x in relevant if x["discussion_label"] == "GDS" and x[field] is not None]
        t = [float(x[field]) for x in relevant if x["discussion_label"] == "TPS" and x[field] is not None]
        measures[field] = {"gds_mean": statistics.mean(g) if g else None,
                           "tps_mean": statistics.mean(t) if t else None,
                           "comparison": mean_ci(g, t)}
    summary = {
        "all_unique_posts": len(records),
        "relevance_counts": dict(Counter(x["relevance"] for x in records.values())),
        "relevant_discussion_counts": dict(counts),
        "coverage": {k: cover(k) for k in ("TPS", "GDS", "Other", "Unclear")},
        "measures": measures,
        "source_hashes": {str(p): sha(p) for p in (DEV, SCORES, THREADS, A, B, PILOT / "sample_ids.csv", PILOT / "ai_assisted_labels.csv")},
        "interpretation_constraint": "Single ID-level pool, mixed reviewed/AI-assisted labels and targeted pilot selection. Combined score differences are descriptive, not a representative or independently validated population estimate.",
    }
    OUT.mkdir(exist_ok=True)
    with (OUT / "combined_pool.jsonl").open("w", encoding="utf-8") as handle:
        for rid in sorted(records):
            handle.write(json.dumps(records[rid], sort_keys=True) + "\n")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"all_unique_posts": len(records), "relevance_counts": summary["relevance_counts"], "relevant_discussion_counts": summary["relevant_discussion_counts"], "coverage": summary["coverage"], "measures": summary["measures"]}, indent=2))


if __name__ == "__main__":
    main()
