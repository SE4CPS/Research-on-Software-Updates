"""Phase 1: reconcile frozen/historical TPS/GDS cohorts and missingness reason codes."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from typing import Any

from full_resource_sentiment_v1.strict_continuation_2026_11_03 import FROZEN_DEF, HIST_DEF
from full_resource_sentiment_v1.strict_continuation_2026_11_03.common import (
    COMMENT_SCORES,
    DATASET_JSONL,
    OUT_ROOT,
    RESTRICTED_ROOT,
    THREAD_METRICS,
    load_jsonl,
    utcnow,
    write_json,
    write_jsonl,
)

EXPECTED_FROZEN = {
    "all": {"TPS": 3028, "GDS": 231},
    "author_ge1": {"TPS": 956, "GDS": 98},
    "author_ge2": {"TPS": 437, "GDS": 66},
    "author_ge4": {"TPS": 115, "GDS": 30},
    "community_ge1": {"TPS": 1489, "GDS": 140},
}


def _gate_reasons(
    *,
    n_author_scoreable: int,
    n_author_empty: int,
    n_author_missing_ts: int,
    n_author_missing_score: int,
    n_community_scoreable: int,
    post_ts_valid: bool,
    threshold: int,
) -> list[str]:
    reasons: list[str] = []
    if n_author_scoreable == 0:
        reasons.append("no_saved_author_reply_scoreable")
    if n_author_empty > 0 and n_author_scoreable == 0:
        reasons.append("author_replies_empty_text_only")
    if n_author_missing_ts > 0:
        reasons.append("author_reply_missing_timestamp")
    if n_author_missing_score > 0:
        reasons.append("author_reply_missing_sentiment_score")
    if n_author_scoreable < threshold:
        reasons.append(f"insufficient_author_replies_for_threshold_{threshold}")
    if n_community_scoreable == 0:
        reasons.append("no_saved_community_reply_scoreable")
    if not post_ts_valid:
        reasons.append("missing_or_invalid_post_timestamp")
    if not reasons:
        reasons.append("eligible")
    return reasons


def run_phase1() -> dict[str, Any]:
    dest = OUT_ROOT / "phase1_cohort_reconciliation"
    dest.mkdir(parents=True, exist_ok=True)

    frame = load_jsonl(DATASET_JSONL)
    comments_by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in load_jsonl(COMMENT_SCORES):
        comments_by[str(c["reddit_id"])].append(c)
    tm = {str(r["reddit_id"]): r for r in load_jsonl(THREAD_METRICS)}

    row_level = []
    aggregates: dict[str, Any] = {
        "frozen": {},
        "historical": {},
        "exact_author_counts": {"frozen": {}, "historical": {}},
    }

    for defn, key in ((FROZEN_DEF, "frozen"), (HIST_DEF, "historical")):
        for lab in ("TPS", "GDS"):
            subset = [
                r
                for r in frame
                if r.get("final_relevance") == "relevant"
                and r.get("final_discussion") == lab
                and r.get("sentiment_definition") == defn
            ]
            thr_counts = {}
            for thr in (0, 1, 2, 4):
                thr_counts[f"author_ge{thr}"] = 0
            community_ge1 = 0
            post_ts_valid_n = 0
            exact = Counter()

            for r in subset:
                rid = str(r["reddit_id"])
                sent = r.get("sentiment") or {}
                clist = comments_by.get(rid, [])
                author_all = [c for c in clist if c.get("role") == "author"]
                author_scoreable = [
                    c
                    for c in author_all
                    if c.get("compound") is not None and str(c.get("compound")) != ""
                ]
                # nonempty text proxy: scored package only retains scoreable; empty tracked if compound missing
                author_empty = [c for c in author_all if c.get("compound") is None]
                author_missing_ts = [
                    c for c in author_scoreable if not c.get("timestamp_valid") or c.get("timestamp_epoch_seconds") is None
                ]
                author_missing_score = author_empty
                community_scoreable = [
                    c for c in clist if c.get("role") == "community" and c.get("compound") is not None
                ]
                # Prefer package n_author_replies_nonempty for threshold alignment with accepted counts
                n_author_pkg = int(sent.get("n_author_replies_nonempty") or 0)
                n_comm_pkg = int(sent.get("n_community_replies_nonempty") or 0)
                post_ts_valid = bool((tm.get(rid) or {}).get("post_timestamp_valid") or sent.get("post_timestamp_valid"))
                if post_ts_valid:
                    post_ts_valid_n += 1
                exact[n_author_pkg] += 1
                for thr in (0, 1, 2, 4):
                    if n_author_pkg >= thr:
                        thr_counts[f"author_ge{thr}"] += 1
                if n_comm_pkg >= 1:
                    community_ge1 += 1

                # Primary exclusion for author>=4 analysis
                reasons4 = _gate_reasons(
                    n_author_scoreable=n_author_pkg,
                    n_author_empty=len(author_empty),
                    n_author_missing_ts=len(author_missing_ts),
                    n_author_missing_score=len(author_missing_score),
                    n_community_scoreable=n_comm_pkg,
                    post_ts_valid=post_ts_valid,
                    threshold=4,
                )
                reasons1 = _gate_reasons(
                    n_author_scoreable=n_author_pkg,
                    n_author_empty=len(author_empty),
                    n_author_missing_ts=len(author_missing_ts),
                    n_author_missing_score=len(author_missing_score),
                    n_community_scoreable=n_comm_pkg,
                    post_ts_valid=post_ts_valid,
                    threshold=1,
                )
                row_level.append(
                    {
                        "reddit_id": rid,
                        "final_discussion": lab,
                        "sentiment_definition": defn,
                        "n_author_pkg": n_author_pkg,
                        "n_community_pkg": n_comm_pkg,
                        "n_author_comments_in_scores": len(author_all),
                        "n_author_scoreable_in_scores": len(author_scoreable),
                        "n_author_missing_timestamp_among_scoreable": len(author_missing_ts),
                        "post_timestamp_valid": post_ts_valid,
                        "reason_codes_author_ge1": reasons1,
                        "reason_codes_author_ge4": reasons4,
                        "primary_exclusion_author_ge4": reasons4[0] if reasons4[0] != "eligible" else "eligible",
                        "note": "Missing replies are unobserved, not neutral sentiment.",
                    }
                )

            aggregates[key][lab] = {
                **thr_counts,
                "community_ge1": community_ge1,
                "post_timestamp_valid": post_ts_valid_n,
                "n": len(subset),
            }
            aggregates["exact_author_counts"][key][lab] = {str(k): v for k, v in sorted(exact.items())}

    # Reconciliation vs expected frozen
    recon = {}
    for metric, expected in EXPECTED_FROZEN.items():
        got = {
            "TPS": aggregates["frozen"]["TPS"]["author_ge0" if metric == "all" else metric]
            if metric != "all"
            else aggregates["frozen"]["TPS"]["n"],
            "GDS": aggregates["frozen"]["GDS"]["author_ge0" if metric == "all" else metric]
            if metric != "all"
            else aggregates["frozen"]["GDS"]["n"],
        }
        if metric == "all":
            got = {"TPS": aggregates["frozen"]["TPS"]["n"], "GDS": aggregates["frozen"]["GDS"]["n"]}
        elif metric == "author_ge1":
            got = {"TPS": aggregates["frozen"]["TPS"]["author_ge1"], "GDS": aggregates["frozen"]["GDS"]["author_ge1"]}
        elif metric == "author_ge2":
            got = {"TPS": aggregates["frozen"]["TPS"]["author_ge2"], "GDS": aggregates["frozen"]["GDS"]["author_ge2"]}
        elif metric == "author_ge4":
            got = {"TPS": aggregates["frozen"]["TPS"]["author_ge4"], "GDS": aggregates["frozen"]["GDS"]["author_ge4"]}
        elif metric == "community_ge1":
            got = {
                "TPS": aggregates["frozen"]["TPS"]["community_ge1"],
                "GDS": aggregates["frozen"]["GDS"]["community_ge1"],
            }
        recon[metric] = {
            "expected": expected,
            "observed": got,
            "match": got == expected,
        }

    # Aggregate exclusion reasons for frozen author>=4
    excl = Counter(
        r["primary_exclusion_author_ge4"]
        for r in row_level
        if r["sentiment_definition"] == FROZEN_DEF and r["primary_exclusion_author_ge4"] != "eligible"
    )

    write_jsonl(RESTRICTED_ROOT / "phase1_row_level_reason_codes.jsonl", row_level)
    report = {
        "utc": utcnow(),
        "aggregates": aggregates,
        "reconciliation_frozen_expected": recon,
        "all_frozen_expected_match": all(v["match"] for v in recon.values()),
        "frozen_author_ge4_exclusion_reason_counts": dict(excl),
        "interpretation": (
            "115/30 is a reply-depth restriction for later−earlier author shift (≥4), "
            "not evidence that other posts have neutral sentiment. Historical 520 kept separate."
        ),
    }
    write_json(dest / "COHORT_RECONCILIATION.json", report)

    with (dest / "cohort_tables.csv").open("w", encoding="utf-8", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["definition", "label", "metric", "n"])
        for defn in ("frozen", "historical"):
            for lab in ("TPS", "GDS"):
                block = aggregates[defn][lab]
                for metric, n in block.items():
                    w.writerow([defn, lab, metric, n])

    (dest / "FINDINGS.md").write_text(
        f"""# Phase 1 — cohort reconciliation

**UTC:** {report['utc']}  
Frozen expected match: **{report['all_frozen_expected_match']}**

## Frozen comparable
- All: TPS {aggregates['frozen']['TPS']['n']} / GDS {aggregates['frozen']['GDS']['n']}
- Author≥1/2/4: TPS {aggregates['frozen']['TPS']['author_ge1']}/{aggregates['frozen']['TPS']['author_ge2']}/{aggregates['frozen']['TPS']['author_ge4']}; GDS {aggregates['frozen']['GDS']['author_ge1']}/{aggregates['frozen']['GDS']['author_ge2']}/{aggregates['frozen']['GDS']['author_ge4']}
- Community≥1: TPS {aggregates['frozen']['TPS']['community_ge1']} / GDS {aggregates['frozen']['GDS']['community_ge1']}

## Historical-only (separate)
- Author≥1/2/4: TPS {aggregates['historical']['TPS']['author_ge1']}/{aggregates['historical']['TPS']['author_ge2']}/{aggregates['historical']['TPS']['author_ge4']}; GDS {aggregates['historical']['GDS']['author_ge1']}/{aggregates['historical']['GDS']['author_ge2']}/{aggregates['historical']['GDS']['author_ge4']}

Missing replies = unobserved, not neutral. Row-level reasons: restricted `phase1_row_level_reason_codes.jsonl`.
""",
        encoding="utf-8",
    )
    return report
