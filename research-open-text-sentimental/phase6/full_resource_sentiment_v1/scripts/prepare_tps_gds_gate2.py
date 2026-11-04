#!/usr/bin/env python3
"""Prepare a second, disjoint 90-case TPS/GDS check after model comparison."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parents[1]
sys.path.insert(0, str(BASE / "src"))
from full_resource_sentiment_v1.relevance_classify.preprocess import redact_one_way, deterministic_unassessable

RUN = BASE / "restricted/tps_gds_full_classification_2026-11-02"
COMBINED = BASE / "results/tps_gds_combined_pool_2026-11-02/combined_pool.jsonl"
DEV = REPO / "phase3/revised_development_v2/development_v2.jsonl"
SEED = 20261103


def rows(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main():
    prior = set(json.loads((RUN / "gate_truth.json").read_text(encoding="utf-8")))
    combined = {x["reddit_id"]: x for x in rows(COMBINED)}
    docs = {x["reddit_id"]: x for x in rows(DEV)}
    classes = defaultdict(list)
    for rid, x in combined.items():
        if rid in prior or x["selection_arm"] != "reviewed_f_set" or x["relevance"] != "relevant":
            continue
        if x["discussion_label"] in {"TPS", "GDS", "Other"}:
            classes[x["discussion_label"]].append(rid)
    rng = random.Random(SEED)
    selected = []
    for label in ("TPS", "GDS", "Other"):
        selected.extend(rng.sample(sorted(classes[label]), 30))
    assert len(selected) == len(set(selected)) == 90 and not (set(selected) & prior)
    truth = {rid: combined[rid]["discussion_label"] for rid in selected}
    out = RUN / "gate2_packet.jsonl"
    if out.exists() or (RUN / "gate2_truth.json").exists():
        raise RuntimeError("Second gate already prepared; preserve frozen sample")
    with out.open("w", encoding="utf-8") as handle:
        for rid in selected:
            d = docs[rid]
            title, _ = redact_one_way(str(d.get("title") or "")[:1000])
            body, _ = redact_one_way(str(d.get("post_body") or "")[:8000])
            handle.write(json.dumps({"reddit_id": rid, "title": title, "body": body,
                                     "deterministic_unassessable": deterministic_unassessable(title, body) is not None},
                                    ensure_ascii=False, sort_keys=True) + "\n")
    (RUN / "gate2_truth.json").write_text(json.dumps(truth, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (RUN / "gate2_manifest.json").write_text(json.dumps({"seed": SEED, "n": 90,
       "disjoint_from_gate1": True, "packet_sha256": hashlib.sha256(out.read_bytes()).hexdigest()},
       sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print("prepared 90 disjoint cases")


if __name__ == "__main__":
    main()
