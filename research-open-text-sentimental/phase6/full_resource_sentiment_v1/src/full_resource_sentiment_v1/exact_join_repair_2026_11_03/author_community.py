"""Title/body baseline + chronological author/community sentiment (exact-join TextBlob where used)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from full_resource_sentiment_v1.exact_join_repair_2026_11_03 import N_BOOT, SEED
from full_resource_sentiment_v1.exact_join_repair_2026_11_03.common import (
    COMMENT_SCORES,
    DATASET_JSONL,
    FROZEN_DEF,
    OUT_ROOT,
    RESTRICTED_ROOT,
    TEXT_MANIFEST,
    THREAD_METRICS,
    load_jsonl,
    utcnow,
    write_json,
    write_jsonl,
)
from full_resource_sentiment_v1.master_research_corrections_2026_11_03.textblob_checks import (
    independent_stratified_mean_diff,
)
from full_resource_sentiment_v1.scoring import early_late_or_none


def run_author_community() -> dict[str, Any]:
    dest = OUT_ROOT / "phase4_author_community"
    dest.mkdir(parents=True, exist_ok=True)

    texts = {str(r["reddit_id"]): r for r in load_jsonl(TEXT_MANIFEST)}
    analysis = load_jsonl(RESTRICTED_ROOT / "analysis_units_exact_join.jsonl")
    tb_by = defaultdict(dict)
    for u in analysis:
        if u.get("textblob") is not None:
            tb_by[u["reddit_id"]][u["unit"]] = float(u["textblob"])

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
        post = sent.get("combined_compound")
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
        n_a = int(sent.get("n_author_replies_nonempty") or 0)
        n_c = int(sent.get("n_community_replies_nonempty") or 0)
        tb = tb_by.get(rid) or {}
        # Post-level baseline: frozen combined VADER; optional mean(title_tb, body_tb) as TextBlob baseline
        title_tb = tb.get("title_only")
        body_tb = tb.get("body_only")
        post_tb = tb.get("original_post")
        tb_baseline = None
        if title_tb is not None and body_tb is not None:
            tb_baseline = float(np.mean([title_tb, body_tb]))
        elif post_tb is not None:
            tb_baseline = post_tb

        first_a_tb = tb.get("first_author_reply")
        second_a_tb = tb.get("second_author_reply")
        first_c_tb = tb.get("first_community_reply")

        author_mean = float(np.mean(vals)) if vals else None
        community_mean = float(np.mean(cvals)) if cvals else None
        rows.append(
            {
                "reddit_id": rid,
                "label": r["final_discussion"],
                "n_author": n_a,
                "n_community": n_c,
                # Event definitions:
                # event0 = original post (title+body segments); event1 = first author reply; event2 = second
                "post_vader_baseline": float(post) if post is not None else None,
                "post_textblob_baseline": tb_baseline,
                "title_textblob": title_tb,
                "body_textblob": body_tb,
                "first_author_minus_post_vader": (vals[0] - float(post)) if vals and post is not None else None,
                "second_minus_first_vader": (vals[1] - vals[0]) if len(vals) >= 2 else None,
                "first_author_minus_post_textblob": (
                    (first_a_tb - tb_baseline) if first_a_tb is not None and tb_baseline is not None else None
                ),
                "second_minus_first_textblob": (
                    (second_a_tb - first_a_tb) if second_a_tb is not None and first_a_tb is not None else None
                ),
                "author_shift_ge4_vader": early_late_or_none(vals) if n_a >= 4 else None,
                "author_minus_community_mean_vader": (
                    author_mean - community_mean
                    if author_mean is not None and community_mean is not None
                    else None
                ),
                "first_author_minus_first_community_vader": (
                    vals[0] - cvals[0] if vals and cvals else None
                ),
                "first_author_minus_first_community_textblob": (
                    (first_a_tb - first_c_tb) if first_a_tb is not None and first_c_tb is not None else None
                ),
                "community_minus_post_mean_vader": (
                    community_mean - float(post) if community_mean is not None and post is not None else None
                ),
                "post_to_first_reply_hours": (tm.get(rid) or {}).get("post_to_first_reply_hours")
                if (tm.get(rid) or {}).get("post_to_first_reply_eligible")
                else None,
                "interpretation_warning": "Sentiment change ≠ issue resolution",
            }
        )

    write_jsonl(RESTRICTED_ROOT / "author_community_thread_rows.jsonl", rows)

    def cohort(pred):
        return [r for r in rows if pred(r)]

    cohorts = {
        "all_frozen": rows,
        "author_ge1": cohort(lambda r: r["n_author"] >= 1),
        "author_ge2": cohort(lambda r: r["n_author"] >= 2),
        "author_ge4": cohort(lambda r: r["n_author"] >= 4),
        "community_ge1": cohort(lambda r: r["n_community"] >= 1),
    }

    def gds_tps(rs, key):
        g = [float(r[key]) for r in rs if r["label"] == "GDS" and r.get(key) is not None]
        t = [float(r[key]) for r in rs if r["label"] == "TPS" and r.get(key) is not None]
        return independent_stratified_mean_diff(g, t, seed=SEED + 70, n_boot=N_BOOT)

    keys = [
        "first_author_minus_post_vader",
        "second_minus_first_vader",
        "first_author_minus_post_textblob",
        "second_minus_first_textblob",
        "author_shift_ge4_vader",
        "author_minus_community_mean_vader",
        "first_author_minus_first_community_vader",
        "first_author_minus_first_community_textblob",
        "community_minus_post_mean_vader",
        "title_textblob",
        "body_textblob",
        "post_vader_baseline",
        "post_textblob_baseline",
    ]
    results = {}
    for cname, rs in cohorts.items():
        block = {
            "n": len(rs),
            "n_gds": sum(1 for r in rs if r["label"] == "GDS"),
            "n_tps": sum(1 for r in rs if r["label"] == "TPS"),
            "contrasts": {},
        }
        for k in keys:
            # author_shift only meaningful in ge4
            if k == "author_shift_ge4_vader" and cname not in {"author_ge4", "all_frozen"}:
                continue
            block["contrasts"][k] = gds_tps(rs, k)
        results[cname] = block

    # Title vs body within-post divergence (descriptive)
    title_body_diffs = []
    for r in rows:
        if r.get("title_textblob") is not None and r.get("body_textblob") is not None:
            title_body_diffs.append(float(r["body_textblob"]) - float(r["title_textblob"]))
    title_body = {
        "n": len(title_body_diffs),
        "mean_body_minus_title": float(np.mean(title_body_diffs)) if title_body_diffs else None,
        "note": "Within-post segment contrast; not a second event",
    }

    report = {
        "utc": utcnow(),
        "event_definitions": {
            "event0_post": "title and body are two text segments of one original post event; baseline = frozen combined VADER and/or TextBlob post/mean(title,body)",
            "event1_first_author_comment": "first chronological author reply (adapter role)",
            "event2_second_author_comment": "second chronological author reply",
        },
        "title_vs_body": title_body,
        "cohort_results": results,
        "interpretation_rule": "Never interpret sentiment change alone as proof of issue resolution",
        "join_basis_for_textblob_replies": "exact capture_page+reddit_id+original_index",
    }
    write_json(dest / "AUTHOR_COMMUNITY_ANALYSES.json", report)
    return report
