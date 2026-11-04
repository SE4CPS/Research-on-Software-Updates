"""Stage 4: conversation analyses by author-reply depth cohorts."""

from __future__ import annotations

import csv
from collections import defaultdict
from typing import Any

import numpy as np

from full_resource_sentiment_v1.master_research_corrections_2026_11_03.textblob_checks import (
    independent_stratified_mean_diff,
)
from full_resource_sentiment_v1.scoring import early_late_or_none
from full_resource_sentiment_v1.strict_continuation_corrections_2026_11_03 import SEED
from full_resource_sentiment_v1.strict_continuation_corrections_2026_11_03.common import (
    COMMENT_SCORES,
    DATASET_JSONL,
    FROZEN_DEF,
    OUT_ROOT,
    THREAD_METRICS,
    load_jsonl,
    utcnow,
    write_json,
)


def run_stage4() -> dict[str, Any]:
    dest = OUT_ROOT / "stage4_conversation_by_depth"
    dest.mkdir(parents=True, exist_ok=True)
    comments_by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in load_jsonl(COMMENT_SCORES):
        comments_by[str(c["reddit_id"])].append(c)
    tm = {str(r["reddit_id"]): r for r in load_jsonl(THREAD_METRICS)}

    rows = []
    for r in load_jsonl(DATASET_JSONL):
        if r.get("final_relevance") != "relevant":
            continue
        if r.get("final_discussion") not in {"TPS", "GDS"}:
            continue
        if r.get("sentiment_definition") != FROZEN_DEF:
            continue
        rid = str(r["reddit_id"])
        sent = r.get("sentiment") or {}
        clist = sorted(
            comments_by.get(rid, []),
            key=lambda c: (
                c.get("timestamp_epoch_seconds") is None,
                c.get("timestamp_epoch_seconds") or 0,
                c.get("chronological_index") or 0,
            ),
        )
        author = [c for c in clist if c.get("role") == "author" and c.get("compound") is not None]
        community = [c for c in clist if c.get("role") == "community" and c.get("compound") is not None]
        vals = [float(c["compound"]) for c in author]
        cvals = [float(c["compound"]) for c in community]
        post = sent.get("combined_compound")
        n_a = int(sent.get("n_author_replies_nonempty") or 0)
        n_c = int(sent.get("n_community_replies_nonempty") or 0)
        author_mean = float(np.mean(vals)) if vals else None
        community_mean = float(np.mean(cvals)) if cvals else None
        rows.append(
            {
                "reddit_id": rid,
                "label": r["final_discussion"],
                "selection_stratum": r.get("selection_stratum"),
                "decision_source": r.get("decision_source"),
                "count_gap": sent.get("count_gap"),
                "n_author": n_a,
                "n_community": n_c,
                "first_author_minus_post": (vals[0] - float(post)) if vals and post is not None else None,
                "second_minus_first": (vals[1] - vals[0]) if len(vals) >= 2 else None,
                "author_shift_ge4": early_late_or_none(vals) if n_a >= 4 else None,
                "author_minus_community_same_thread": (
                    author_mean - community_mean
                    if author_mean is not None and community_mean is not None
                    else None
                ),
                "post_to_first_reply_hours": (tm.get(rid) or {}).get("post_to_first_reply_hours")
                if (tm.get(rid) or {}).get("post_to_first_reply_eligible")
                else None,
            }
        )

    def cohort(pred):
        return [r for r in rows if pred(r)]

    cohorts = {
        "author_ge1": cohort(lambda r: r["n_author"] >= 1),
        "author_ge2": cohort(lambda r: r["n_author"] >= 2),
        "author_ge4": cohort(lambda r: r["n_author"] >= 4),
    }

    def gds_tps(rs, key):
        g = [float(r[key]) for r in rs if r["label"] == "GDS" and r.get(key) is not None]
        t = [float(r[key]) for r in rs if r["label"] == "TPS" and r.get(key) is not None]
        return independent_stratified_mean_diff(g, t, seed=SEED + 70)

    results = {}
    for cname, rs in cohorts.items():
        results[cname] = {
            "n_tps": sum(1 for r in rs if r["label"] == "TPS"),
            "n_gds": sum(1 for r in rs if r["label"] == "GDS"),
            "first_author_minus_post": gds_tps(rs, "first_author_minus_post"),
            "author_minus_community_same_thread": gds_tps(rs, "author_minus_community_same_thread"),
        }
        if cname in {"author_ge2", "author_ge4"}:
            results[cname]["second_minus_first"] = gds_tps(rs, "second_minus_first")
        if cname == "author_ge4":
            results[cname]["author_shift_ge4"] = gds_tps(rs, "author_shift_ge4")

    # Sensitivity period / count_gap on author_ge1 first_author
    sens = {}
    for period in ("older", "october_2026"):
        rs = [r for r in cohorts["author_ge1"] if r.get("selection_stratum") == period]
        sens[f"period_{period}"] = gds_tps(rs, "first_author_minus_post")
    for gap in (True, False):
        rs = [r for r in cohorts["author_ge1"] if bool(r.get("count_gap")) == gap]
        sens[f"count_gap_{gap}"] = gds_tps(rs, "first_author_minus_post")

    report = {
        "utc": utcnow(),
        "cohort": "frozen comparable only; recovered candidates excluded",
        "definitions": {
            "author_ge1": "first-author-reply sentiment analyses",
            "author_ge2": "conversation-rich / two-point analyses (priority for recovery)",
            "author_ge4": "long trajectory only",
        },
        "results": results,
        "sensitivity_first_author_ge1": sens,
        "guards": {
            "not_resolution": True,
            "comments_not_independent": "thread-level bootstrap only",
            "historical_separate": True,
            "no_formal_survival_without_capture_end_time": True,
        },
    }
    write_json(dest / "CONVERSATION_BY_DEPTH.json", report)
    with (dest / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["cohort", "metric", "n_gds", "n_tps", "mean_diff", "ci_low", "ci_high"])
        for cname, block in results.items():
            for metric, est in block.items():
                if not isinstance(est, dict) or "mean_diff" not in est:
                    continue
                ci = est.get("mean_diff_ci95") or [None, None]
                w.writerow([cname, metric, est.get("n_a"), est.get("n_b"), est.get("mean_diff"), ci[0], ci[1]])
    return report
