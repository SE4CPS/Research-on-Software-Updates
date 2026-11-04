"""Phase 5: conversation analyses by distinct eligibility + decision packet helpers."""

from __future__ import annotations

import csv
from collections import defaultdict
from typing import Any

import numpy as np

from full_resource_sentiment_v1.master_research_corrections_2026_11_03.textblob_checks import (
    independent_stratified_mean_diff,
)
from full_resource_sentiment_v1.scoring import early_late_or_none
from full_resource_sentiment_v1.strict_continuation_2026_11_03 import SEED
from full_resource_sentiment_v1.strict_continuation_2026_11_03.common import (
    COMMENT_SCORES,
    DATASET_JSONL,
    FROZEN_DEF,
    OUT_ROOT,
    THREAD_METRICS,
    load_jsonl,
    utcnow,
    write_json,
)


def run_phase5() -> dict[str, Any]:
    dest = OUT_ROOT / "phase5_conversation"
    dest.mkdir(parents=True, exist_ok=True)

    frame = [
        r
        for r in load_jsonl(DATASET_JSONL)
        if r.get("final_relevance") == "relevant"
        and r.get("final_discussion") in {"TPS", "GDS"}
        and r.get("sentiment_definition") == FROZEN_DEF
    ]
    comments_by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in load_jsonl(COMMENT_SCORES):
        comments_by[str(c["reddit_id"])].append(c)
    tm = {str(r["reddit_id"]): r for r in load_jsonl(THREAD_METRICS)}

    rows = []
    for r in frame:
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
        author_vals = [float(c["compound"]) for c in author]
        community_vals = [float(c["compound"]) for c in community]
        post = sent.get("combined_compound")
        n_author = int(sent.get("n_author_replies_nonempty") or 0)
        n_comm = int(sent.get("n_community_replies_nonempty") or 0)
        post_to_author = (float(np.mean(author_vals)) - float(post)) if author_vals and post is not None else None
        two_point = (author_vals[-1] - author_vals[0]) if len(author_vals) >= 2 else None
        shift4 = early_late_or_none(author_vals)
        community_mean = float(np.mean(community_vals)) if community_vals else None
        p2fr = (tm.get(rid) or {}).get("post_to_first_reply_hours")
        p2fr_ok = bool((tm.get(rid) or {}).get("post_to_first_reply_eligible"))
        rows.append(
            {
                "reddit_id": rid,
                "label": r["final_discussion"],
                "selection_stratum": r.get("selection_stratum"),
                "count_gap": sent.get("count_gap"),
                "n_author": n_author,
                "n_community": n_comm,
                "post_to_author_ge1": post_to_author if n_author >= 1 else None,
                "within_author_two_point_ge2": two_point if n_author >= 2 else None,
                "author_shift_ge4": shift4 if n_author >= 4 else None,
                "community_mean_ge1": community_mean if n_comm >= 1 else None,
                "post_to_first_reply_hours": p2fr if p2fr_ok else None,
            }
        )

    tps = [r for r in rows if r["label"] == "TPS"]
    gds = [r for r in rows if r["label"] == "GDS"]

    def vals(rs, key):
        return [float(r[key]) for r in rs if r.get(key) is not None]

    outcomes = {
        "post_to_author_ge1": {
            "eligibility": "author>=1",
            "n_tps": sum(r["n_author"] >= 1 for r in tps),
            "n_gds": sum(r["n_author"] >= 1 for r in gds),
            "contrast": independent_stratified_mean_diff(
                vals(gds, "post_to_author_ge1"), vals(tps, "post_to_author_ge1"), seed=SEED + 31
            ),
        },
        "within_author_two_point_ge2": {
            "eligibility": "author>=2",
            "n_tps": sum(r["n_author"] >= 2 for r in tps),
            "n_gds": sum(r["n_author"] >= 2 for r in gds),
            "contrast": independent_stratified_mean_diff(
                vals(gds, "within_author_two_point_ge2"),
                vals(tps, "within_author_two_point_ge2"),
                seed=SEED + 32,
            ),
        },
        "author_shift_ge4": {
            "eligibility": "author>=4",
            "n_tps": sum(r["n_author"] >= 4 for r in tps),
            "n_gds": sum(r["n_author"] >= 4 for r in gds),
            "contrast": independent_stratified_mean_diff(
                vals(gds, "author_shift_ge4"), vals(tps, "author_shift_ge4"), seed=SEED + 33
            ),
        },
        "community_mean_ge1": {
            "eligibility": "community>=1",
            "n_tps": sum(r["n_community"] >= 1 for r in tps),
            "n_gds": sum(r["n_community"] >= 1 for r in gds),
            "contrast": independent_stratified_mean_diff(
                vals(gds, "community_mean_ge1"), vals(tps, "community_mean_ge1"), seed=SEED + 34
            ),
        },
        "post_to_first_reply_hours_timed": {
            "eligibility": "package post_to_first_reply_eligible",
            "n_tps": sum(r.get("post_to_first_reply_hours") is not None for r in tps),
            "n_gds": sum(r.get("post_to_first_reply_hours") is not None for r in gds),
            "contrast": independent_stratified_mean_diff(
                vals(gds, "post_to_first_reply_hours"),
                vals(tps, "post_to_first_reply_hours"),
                seed=SEED + 35,
            ),
        },
    }

    # Missingness / period sensitivity for author_shift_ge4 and post_to_author
    sensitivity = {}
    for outcome_key in ("post_to_author_ge1", "author_shift_ge4"):
        sensitivity[outcome_key] = {}
        for period in ("older", "october_2026"):
            g = [r for r in gds if r.get("selection_stratum") == period]
            t = [r for r in tps if r.get("selection_stratum") == period]
            sensitivity[outcome_key][period] = independent_stratified_mean_diff(
                vals(g, outcome_key), vals(t, outcome_key), seed=SEED + 40
            )
        for gap_flag, name in ((True, "count_gap_true"), (False, "count_gap_false")):
            g = [r for r in gds if bool(r.get("count_gap")) == gap_flag]
            t = [r for r in tps if bool(r.get("count_gap")) == gap_flag]
            sensitivity[outcome_key][name] = independent_stratified_mean_diff(
                vals(g, outcome_key), vals(t, outcome_key), seed=SEED + 41
            )

    report = {
        "utc": utcnow(),
        "cohort": "frozen comparable TPS/GDS only; recovered-post candidates excluded",
        "outcomes": outcomes,
        "sensitivity": sensitivity,
        "interpretation_guards": {
            "not_resolution": "Sentiment improvement ≠ verified technical resolution",
            "overlapping_outcomes": (
                "post→author, two-point, and author_shift≥4 overlap in units — do not present as independent evidence"
            ),
            "missing_replies": "unobserved, not neutral",
        },
    }
    write_json(dest / "CONVERSATION_ANALYSES.json", report)

    with (dest / "outcome_contrasts.csv").open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(
            handle,
            fieldnames=["outcome", "eligibility", "n_gds", "n_tps", "mean_diff", "ci_low", "ci_high"],
        )
        w.writeheader()
        for name, block in outcomes.items():
            c = block["contrast"]
            ci = c.get("mean_diff_ci95") or [None, None]
            w.writerow(
                {
                    "outcome": name,
                    "eligibility": block["eligibility"],
                    "n_gds": c.get("n_a"),
                    "n_tps": c.get("n_b"),
                    "mean_diff": c.get("mean_diff"),
                    "ci_low": ci[0],
                    "ci_high": ci[1],
                }
            )

    (dest / "FINDINGS.md").write_text(
        f"""# Phase 5 — conversation analyses

**UTC:** {report['utc']}  
Distinct eligibility per outcome (see CSV). Overlapping D/E estimands are **not** independent evidence. Not resolution.

Author≥4 remains small (GDS n={outcomes['author_shift_ge4']['n_gds']}).
""",
        encoding="utf-8",
    )
    return report
