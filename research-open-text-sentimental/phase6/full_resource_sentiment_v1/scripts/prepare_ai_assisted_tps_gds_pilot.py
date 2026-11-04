#!/usr/bin/env python3
"""Reproducibly select a bounded, AI-assisted TPS/GDS research pilot.

This script selects candidates only. It does not assign discussion labels.
Source records and earlier research outputs are read-only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parents[1]
RUN = "tps_gds_ai_pilot_2026-11-02"
OUT = BASE / "results" / RUN
PRIVATE = BASE / "restricted" / RUN
SEED = 20261102

SELECTED = BASE / "results/ai_relevance_exploratory_full_2026-10-29/exploratory_ai_selected_cohort.jsonl"
POST_SCORES = BASE / "results/broad_sentiment_2026-10-22-main/post_scores.jsonl"
DEVELOPMENT = REPO / "phase3/revised_development_v2/development_v2.jsonl"
VALIDATION_KEY = BASE / "restricted/independent_validation_pilot_2026-10-29/validation/private_id_key.jsonl"
CAPTURE = BASE / "restricted/broad_release_train/2026-10-22-main"

COMPLAINT = re.compile(
    r"\b(?:hate|hated|worse|worst|awful|terrible|frustrat\w*|annoy\w*|"
    r"disappoint\w*|ruined|unwanted|forced|broken|breaks?|regress\w*|"
    r"slow(?:er)?|unusable|downgrade|rollback|bad|buggy|painful|removed|lost)\b",
    re.I,
)


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    selected_all = {r["reddit_id"]: r for r in rows(SELECTED)}
    selected = {i: r for i, r in selected_all.items() if r["protected_membership"] is False}
    scores = {r["reddit_id"]: r for r in rows(POST_SCORES)}
    reviewed = {r["reddit_id"] for r in rows(DEVELOPMENT) if scores.get(r["reddit_id"], {}).get("f_eligible")}
    validation = {r["reddit_id"] for r in rows(VALIDATION_KEY)}
    base = set(selected) - reviewed
    assert len(selected_all) == 6204 and len(selected) == 5686
    assert len(set(selected) & reviewed) == 1648 and len(base) == 4038
    drawable = base - validation

    by_page: dict[int, set[str]] = defaultdict(set)
    for i in drawable:
        by_page[int(scores[i]["capture_page"])].add(i)
    docs = {}
    for page, ids in by_page.items():
        data = json.loads((CAPTURE / f"page_{page:05d}.json").read_text(encoding="utf-8"))
        for post in data.get("data") or []:
            rid = str(post.get("redditId") or "")
            if rid in ids:
                comments = []
                for n, comment in enumerate(post.get("comments") or []):
                    if not isinstance(comment, dict):
                        continue
                    body = str(comment.get("body") or "").strip()
                    if body:
                        comments.append({
                            "role": "author" if comment.get("is_submitter") else "community",
                            "created_utc": comment.get("created_utc"),
                            "body": body,
                            "original_index": n,
                        })
                comments.sort(key=lambda x: (str(x["created_utc"] or ""), x["original_index"]))
                docs[rid] = {
                    "reddit_id": rid,
                    "title": str(post.get("title") or ""),
                    "body": str(post.get("author_description") or ""),
                    "comments": comments,
                    "post_created_utc": post.get("created_utc"),
                }
    assert set(docs) == drawable, (len(docs), len(drawable))

    rng = random.Random(SEED)
    random_ids = []
    for period, n in (("older", 20), ("october_2026", 20)):
        pool = sorted(i for i in drawable if selected[i]["stratum"] == period)
        random_ids += rng.sample(pool, n)

    remaining = drawable - set(random_ids)
    cue_count = {i: len(COMPLAINT.findall(docs[i]["title"] + " " + docs[i]["body"])) for i in remaining}
    target_pool = {i for i in remaining if cue_count[i] > 0}
    assert len(target_pool) >= 60
    tie = {i: rng.random() for i in target_pool}

    def priority(i: str):
        score = scores[i]
        return (
            -(4 * int(score["n_author_replies_nonempty"] > 0 and score["n_community_replies_nonempty"] > 0)
              + 2 * int(score["n_author_replies_nonempty"] > 0)
              + int(score["n_nonempty_comments"] > 0)
              + min(cue_count[i], 4)),
            tie[i],
            i,
        )

    target_ids = []
    for period in ("older", "october_2026"):
        pool = sorted((i for i in target_pool if selected[i]["stratum"] == period), key=priority)
        assert len(pool) >= 30, (period, len(pool))
        target_ids += pool[:30]

    sample = [(i, "random") for i in random_ids] + [(i, "targeted") for i in target_ids]
    assert len(sample) == 100 and len({i for i, _ in sample}) == 100
    assert not ({i for i, _ in sample} & validation)
    assert not ({i for i, _ in sample} & reviewed)

    OUT.mkdir(parents=True, exist_ok=False)
    PRIVATE.mkdir(parents=True, exist_ok=False)
    with (PRIVATE / "candidate_packet.jsonl").open("w", encoding="utf-8") as handle:
        for n, (i, arm) in enumerate(sample, 1):
            record = {
                "case_code": f"AI-{n:03d}",
                "selection_arm": arm,
                "period": selected[i]["stratum"],
                **docs[i],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (OUT / "sample_ids.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case_code", "reddit_id", "selection_arm", "period", "n_comments", "n_author_replies", "n_community_replies"])
        for n, (i, arm) in enumerate(sample, 1):
            s = scores[i]
            writer.writerow([f"AI-{n:03d}", i, arm, selected[i]["stratum"], s["n_nonempty_comments"], s["n_author_replies_nonempty"], s["n_community_replies_nonempty"]])
    manifest = {
        "run_id": RUN,
        "seed": SEED,
        "candidate_frame": len(selected_all),
        "nonprotected_ai_selected": len(selected),
        "reviewed_overlap": len(set(selected) & reviewed),
        "unreviewed_base": len(base),
        "validation_overlap_excluded": len(base & validation),
        "drawable": len(drawable),
        "target_cue_pool_after_random": len(target_pool),
        "sample_size": 100,
        "arms": dict(Counter(arm for _, arm in sample)),
        "periods": dict(Counter(selected[i]["stratum"] for i, _ in sample)),
        "source_hashes": {str(p): sha(p) for p in (SELECTED, POST_SCORES, DEVELOPMENT, VALIDATION_KEY)},
        "selection_note": "40 period-balanced uniform random; 60 period-balanced, deterministic complaint-cue/comment-availability priority. Targeted arm is not a probability sample and cannot be used unweighted for population estimates.",
    }
    (OUT / "selection_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
