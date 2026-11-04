#!/usr/bin/env python3
"""Read-only, reviewed F-set TPS/GDS sentiment comparison.

Uses only saved F eligibility and human review discussion-type labels.
Never reads protected evaluation outcomes or AI-assisted pilot labels.
"""

from __future__ import annotations

import hashlib
import json
import random
import statistics
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parents[1]
DEV = REPO / "phase3/revised_development_v2/development_v2.jsonl"
SCORES = BASE / "results/broad_sentiment_2026-10-22-main/post_scores.jsonl"
THREADS = BASE / "results/broad_sentiment_2026-10-22-main/thread_metrics.jsonl"
OUT = BASE / "results/tps_gds_reviewed_sentiment_2026-11-02"
LABELS = {
    "technical_troubleshooting": "TPS",
    "general_dissatisfaction": "GDS",
    "other": "Other",
}
FIELDS = (
    "title_compound", "body_compound", "combined_compound",
    "author_mean", "community_mean", "first_reply_compound",
    "last_minus_first", "order_slope_normalized",
)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(values):
    vals = [float(x) for x in values if x is not None]
    return {
        "n_observed": len(vals),
        "mean": statistics.mean(vals) if vals else None,
        "median": statistics.median(vals) if vals else None,
        "fraction_negative": sum(x < -0.05 for x in vals) / len(vals) if vals else None,
    }


def diff_ci(a, b, seed=20261102, replicates=5000):
    """Difference GDS minus TPS, independent post-level bootstrap."""
    aa = [float(x) for x in a if x is not None]
    bb = [float(x) for x in b if x is not None]
    if not aa or not bb:
        return None
    rng = random.Random(seed)
    draws = []
    for _ in range(replicates):
        ma = sum(aa[rng.randrange(len(aa))] for _ in aa) / len(aa)
        mb = sum(bb[rng.randrange(len(bb))] for _ in bb) / len(bb)
        draws.append(ma - mb)
    draws.sort()
    return {
        "difference_gds_minus_tps": statistics.mean(aa) - statistics.mean(bb),
        "bootstrap_95_percentile": [draws[125], draws[4874]],
        "n_gds": len(aa), "n_tps": len(bb), "replicates": replicates,
    }


def main():
    reviewed = {r["reddit_id"]: r for r in read_jsonl(DEV)}
    scores = {r["reddit_id"]: r for r in read_jsonl(SCORES) if r["f_eligible"]}
    threads = {r["reddit_id"]: r for r in read_jsonl(THREADS) if r["f_eligible"]}
    assert set(scores) == set(threads) and len(scores) == 1864
    groups = {label: [] for label in LABELS.values()}
    for rid, post in scores.items():
        assert not post["reserved_membership"] and not post["fresh_membership"]
        source = reviewed[rid]
        assert source["s2_in_scope"] is True
        label = LABELS[source["discussion_type"]]
        record = {**post, **{k: v for k, v in threads[rid].items() if k not in post}}
        groups[label].append(record)
    assert {k: len(v) for k, v in groups.items()} == {"TPS": 1628, "GDS": 57, "Other": 179}

    output = {
        "population": "human-reviewed, in-scope ReleaseTrain F-set only",
        "counts": {k: len(v) for k, v in groups.items()},
        "source_sha256": {str(p): sha(p) for p in (DEV, SCORES, THREADS)},
        "groups": {}, "gds_minus_tps": {},
        "method": "post-level unweighted means; missing author/community replies excluded from observed-score denominators; independent post bootstrap, 5000 replicates, seed 20261102",
    }
    for label, rows in groups.items():
        coverage = {
            "title_nonempty": sum(bool(r["title_nonempty"]) for r in rows),
            "body_nonempty": sum(bool(r["body_nonempty"]) for r in rows),
            "posts_with_comments": sum(r["n_nonempty_comments"] > 0 for r in rows),
            "posts_with_author_replies": sum(r["n_author_replies_nonempty"] > 0 for r in rows),
            "posts_with_community_replies": sum(r["n_community_replies_nonempty"] > 0 for r in rows),
            "posts_with_both_roles": sum(r["n_author_replies_nonempty"] > 0 and r["n_community_replies_nonempty"] > 0 for r in rows),
            "post_timestamp_valid": sum(bool(r["post_timestamp_valid"]) for r in rows),
            "order_eligible": sum(bool(r["order_eligible"]) for r in rows),
            "elapsed_eligible": sum(bool(r["elapsed_eligible"]) for r in rows),
            "count_gap": sum(bool(r["count_gap"]) for r in rows),
            "date_band": dict(Counter(r["date_band"] for r in rows)),
        }
        output["groups"][label] = {
            "coverage": coverage,
            "metrics": {field: summarize(r[field] for r in rows) for field in FIELDS},
        }
    for field in FIELDS:
        a = [r[field] for r in groups["GDS"]]
        b = [r[field] for r in groups["TPS"]]
        output["gds_minus_tps"][field] = diff_ci(a, b)
    OUT.mkdir(exist_ok=True)
    path = OUT / "analysis.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"counts": output["counts"], "gds_minus_tps": output["gds_minus_tps"]}, indent=2))


if __name__ == "__main__":
    main()
