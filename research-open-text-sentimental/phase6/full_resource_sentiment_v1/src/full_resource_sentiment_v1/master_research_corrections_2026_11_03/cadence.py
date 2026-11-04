"""Cadence completion: post→first reply/author gaps, duration, censoring."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import numpy as np

from full_resource_sentiment_v1.master_research_corrections_2026_11_03 import N_BOOT, SEED
from full_resource_sentiment_v1.master_research_corrections_2026_11_03.common import (
    CAPTURE_DIR,
    COMMENT_SCORES,
    DATASET_JSONL,
    FROZEN_DEF,
    OUT_ROOT,
    THREAD_METRICS,
    load_jsonl,
    utcnow,
    write_json,
)
from full_resource_sentiment_v1.master_research_corrections_2026_11_03.textblob_checks import (
    independent_stratified_mean_diff,
)
from full_resource_sentiment_v1.scoring import early_late_or_none


def _parse_created_epoch(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        return None
    # ISO with optional Z
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return None


def _load_post_created(ids: set[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    remaining = set(ids)
    for path in sorted(CAPTURE_DIR.glob("page_*.json")):
        if not remaining:
            break
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("data") or []:
            rid = str(row.get("redditId") or "")
            if rid not in remaining:
                continue
            ep = _parse_created_epoch(row.get("created_utc"))
            if ep is not None:
                prev = out.get(rid)
                if prev is None or ep < prev:
                    out[rid] = ep
            remaining.discard(rid)
    return out


def run_cadence_completion() -> dict[str, Any]:
    dest = OUT_ROOT / "cadence_timing"
    dest.mkdir(parents=True, exist_ok=True)

    frame = []
    for r in load_jsonl(DATASET_JSONL):
        if r.get("final_relevance") != "relevant":
            continue
        if r.get("final_discussion") not in {"TPS", "GDS"}:
            continue
        if r.get("sentiment_definition") != FROZEN_DEF:
            continue
        frame.append(r)
    frame_ids = {str(r["reddit_id"]) for r in frame}
    by_meta = {str(r["reddit_id"]): r for r in frame}

    tm = {str(r["reddit_id"]): r for r in load_jsonl(THREAD_METRICS) if str(r["reddit_id"]) in frame_ids}
    comments_by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in load_jsonl(COMMENT_SCORES):
        rid = str(c["reddit_id"])
        if rid in frame_ids:
            comments_by[rid].append(c)

    post_created = _load_post_created(frame_ids)

    # Capture age proxy: max timed comment epoch across corpus as observation end if needed;
    # per-thread: last timed reply, else missing.
    threads = []
    for rid in sorted(frame_ids):
        meta = by_meta[rid]
        sent = meta.get("sentiment") or {}
        label = meta.get("final_discussion")
        clist = sorted(
            comments_by.get(rid, []),
            key=lambda c: (
                c.get("timestamp_epoch_seconds") is None,
                c.get("timestamp_epoch_seconds") or 0,
                c.get("chronological_index") or 0,
            ),
        )
        timed = [
            c
            for c in clist
            if c.get("timestamp_valid") and c.get("timestamp_epoch_seconds") is not None
        ]
        author = [c for c in clist if c.get("role") == "author" and c.get("compound") is not None]
        author_vals = [float(c["compound"]) for c in author]
        author_timed = [c for c in timed if c.get("role") == "author"]
        community_timed = [c for c in timed if c.get("role") == "community"]

        post_ep = post_created.get(rid)
        t_metrics = tm.get(rid) or {}

        # Prefer package thread metric when eligible
        p2fr_pkg = t_metrics.get("post_to_first_reply_hours")
        p2fr_eligible = bool(t_metrics.get("post_to_first_reply_eligible"))

        first_reply_ep = float(timed[0]["timestamp_epoch_seconds"]) if timed else None
        first_author_ep = float(author_timed[0]["timestamp_epoch_seconds"]) if author_timed else None
        last_reply_ep = float(timed[-1]["timestamp_epoch_seconds"]) if timed else None

        post_to_first_reply_hours = None
        post_to_first_author_hours = None
        observation_duration_hours = None
        if post_ep is not None and first_reply_ep is not None and first_reply_ep >= post_ep:
            post_to_first_reply_hours = (first_reply_ep - post_ep) / 3600.0
        if post_ep is not None and first_author_ep is not None and first_author_ep >= post_ep:
            post_to_first_author_hours = (first_author_ep - post_ep) / 3600.0
        if post_ep is not None and last_reply_ep is not None and last_reply_ep >= post_ep:
            observation_duration_hours = (last_reply_ep - post_ep) / 3600.0

        # Inter-reply gaps (hours) among timed replies
        gaps = []
        for i in range(1, len(timed)):
            a = float(timed[i - 1]["timestamp_epoch_seconds"])
            b = float(timed[i]["timestamp_epoch_seconds"])
            if b >= a:
                gaps.append((b - a) / 3600.0)
        author_gaps = []
        for i in range(1, len(author_timed)):
            a = float(author_timed[i - 1]["timestamp_epoch_seconds"])
            b = float(author_timed[i]["timestamp_epoch_seconds"])
            if b >= a:
                author_gaps.append((b - a) / 3600.0)
        community_gaps = []
        for i in range(1, len(community_timed)):
            a = float(community_timed[i - 1]["timestamp_epoch_seconds"])
            b = float(community_timed[i]["timestamp_epoch_seconds"])
            if b >= a:
                community_gaps.append((b - a) / 3600.0)

        # Right-censoring / missingness flags
        no_observed_reply = len(clist) == 0
        no_timed_reply = len(timed) == 0
        missing_post_ts = post_ep is None
        invalid_reply_before_post = bool(
            post_ep is not None and first_reply_ep is not None and first_reply_ep < post_ep
        )

        threads.append(
            {
                "reddit_id": rid,
                "final_discussion": label,
                "selection_stratum": meta.get("selection_stratum"),
                "decision_source": meta.get("decision_source"),
                "count_gap": sent.get("count_gap"),
                "n_author": len(author_vals),
                "n_replies_scored": len([c for c in clist if c.get("compound") is not None]),
                "n_timed": len(timed),
                "post_created_epoch": post_ep,
                "post_timestamp_valid_pkg": t_metrics.get("post_timestamp_valid"),
                "post_to_first_reply_hours_pkg": p2fr_pkg if p2fr_eligible else None,
                "post_to_first_reply_hours": post_to_first_reply_hours,
                "post_to_first_author_hours": post_to_first_author_hours,
                "observation_duration_hours": observation_duration_hours,
                "median_inter_reply_gap_hours": float(np.median(gaps)) if gaps else None,
                "median_author_gap_hours": float(np.median(author_gaps)) if author_gaps else None,
                "median_community_gap_hours": float(np.median(community_gaps)) if community_gaps else None,
                "author_shift": early_late_or_none(author_vals),  # preserve ≥4 definition via early_late_or_none
                "no_observed_reply": no_observed_reply,
                "no_timed_reply": no_timed_reply,
                "missing_post_timestamp": missing_post_ts,
                "invalid_reply_before_post": invalid_reply_before_post,
                "right_censored_no_timed_reply": no_timed_reply,
            }
        )

    tps = [t for t in threads if t["final_discussion"] == "TPS"]
    gds = [t for t in threads if t["final_discussion"] == "GDS"]

    def vals(rows: list[dict[str, Any]], key: str) -> list[float]:
        return [float(r[key]) for r in rows if r.get(key) is not None]

    timing_fields = (
        "post_to_first_reply_hours",
        "post_to_first_reply_hours_pkg",
        "post_to_first_author_hours",
        "observation_duration_hours",
        "median_inter_reply_gap_hours",
        "median_author_gap_hours",
        "median_community_gap_hours",
        "author_shift",
    )
    contrasts = {
        f: independent_stratified_mean_diff(vals(gds, f), vals(tps, f), seed=SEED + 11)
        for f in timing_fields
    }

    def coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(rows)
        return {
            "n": n,
            "n_missing_post_timestamp": sum(1 for r in rows if r["missing_post_timestamp"]),
            "n_no_observed_reply": sum(1 for r in rows if r["no_observed_reply"]),
            "n_no_timed_reply_right_censored": sum(1 for r in rows if r["right_censored_no_timed_reply"]),
            "n_invalid_reply_before_post": sum(1 for r in rows if r["invalid_reply_before_post"]),
            "n_count_gap": sum(1 for r in rows if r.get("count_gap")),
            "n_post_to_first_reply": sum(1 for r in rows if r.get("post_to_first_reply_hours") is not None),
            "n_post_to_first_author": sum(1 for r in rows if r.get("post_to_first_author_hours") is not None),
            "n_author_shift_ge4": sum(1 for r in rows if r.get("author_shift") is not None),
            "author_ge1": sum(1 for r in rows if int(r.get("n_author") or 0) >= 1),
            "author_ge4": sum(1 for r in rows if int(r.get("n_author") or 0) >= 4),
        }

    # Period sensitivity
    period_sens = {}
    for period in ("older", "october_2026"):
        g = [t for t in gds if t.get("selection_stratum") == period]
        t = [t for t in tps if t.get("selection_stratum") == period]
        period_sens[period] = {
            "coverage_gds": coverage(g),
            "coverage_tps": coverage(t),
            "post_to_first_reply_hours": independent_stratified_mean_diff(
                vals(g, "post_to_first_reply_hours"), vals(t, "post_to_first_reply_hours"), seed=SEED + 12
            ),
            "post_to_first_author_hours": independent_stratified_mean_diff(
                vals(g, "post_to_first_author_hours"), vals(t, "post_to_first_author_hours"), seed=SEED + 13
            ),
            "author_shift": independent_stratified_mean_diff(
                vals(g, "author_shift"), vals(t, "author_shift"), seed=SEED + 14
            ),
        }

    report = {
        "utc": utcnow(),
        "eligibility_predeclared": {
            "cohort": "frozen relevant TPS/GDS with frozen_broad_release_train_package_scores",
            "author_shift": "later−earlier author compounds via early_late_or_none (≥4 author replies)",
            "post_to_author": "preserved as separate estimand in original Phase D; not re-defined here",
            "timing_eligible": "requires valid post created_utc and reply timestamp ≥ post time",
            "right_censoring": "no timed reply observed ⇒ censored/missing waiting time, not zero",
            "not_resolution": "Sentiment/timing improvement ≠ verified technical resolution",
        },
        "coverage": {"TPS": coverage(tps), "GDS": coverage(gds)},
        "contrasts_gds_minus_tps": contrasts,
        "period_sensitivity": period_sens,
        "n_boot": N_BOOT,
        "seed": SEED + 11,
    }
    write_json(dest / "CADENCE_TIMING.json", report)

    with (dest / "gds_minus_tps_timing.csv").open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(
            handle, fieldnames=["field", "mean_diff", "ci_low", "ci_high", "n_gds", "n_tps", "cliffs_delta"]
        )
        w.writeheader()
        for field, est in contrasts.items():
            ci = est.get("mean_diff_ci95") or [None, None]
            w.writerow(
                {
                    "field": field,
                    "mean_diff": est.get("mean_diff"),
                    "ci_low": ci[0],
                    "ci_high": ci[1],
                    "n_gds": est.get("n_a"),
                    "n_tps": est.get("n_b"),
                    "cliffs_delta": est.get("cliffs_delta"),
                }
            )

    (dest / "FINDINGS.md").write_text(
        f"""# Cadence timing completion

**UTC:** {report['utc']}

## Eligibility / coverage
- TPS: {report['coverage']['TPS']}
- GDS: {report['coverage']['GDS']}

## GDS−TPS timing (thread bootstrap; independent impl)
See `gds_minus_tps_timing.csv`. Author shift definition unchanged (≥4). Waiting times are missing when right-censored — not zeros.

## Interpretation
Do not infer issue resolution from sentiment or faster replies.
""",
        encoding="utf-8",
    )
    (dest / "LIMITATIONS.md").write_text(
        """# Cadence timing limitations

- Observation ends at capture; unreplied threads are right-censored.
- Count gaps and deleted comments undercount replies.
- GDS author≥4 remains small (see coverage).
- Capture created_utc may differ slightly from package post_timestamp when both exist.
""",
        encoding="utf-8",
    )
    return report
