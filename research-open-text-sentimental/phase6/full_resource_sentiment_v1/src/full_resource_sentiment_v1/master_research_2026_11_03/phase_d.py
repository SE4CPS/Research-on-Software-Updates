"""Phase D: author/community trajectory and cadence from saved VADER comment scores."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from full_resource_sentiment_v1.inference import bootstrap_spearman, stratified_mean_diff
from full_resource_sentiment_v1.master_research_2026_11_03 import N_BOOT, SEED
from full_resource_sentiment_v1.master_research_2026_11_03.common import (
    COMMENT_SCORES,
    DATASET_JSONL,
    FROZEN_DEF,
    OUT_ROOT,
    load_jsonl,
    utcnow,
    write_json,
)
from full_resource_sentiment_v1.scoring import early_late_or_none


def _thread_summaries(frame_rows: list[dict[str, Any]], comments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {
        str(r["reddit_id"]): r
        for r in frame_rows
        if r.get("final_relevance") == "relevant"
        and r.get("final_discussion") in {"TPS", "GDS", "Other"}
        and r.get("sentiment_definition") == FROZEN_DEF
    }
    comments_by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in comments:
        rid = str(c["reddit_id"])
        if rid in by_id:
            comments_by[rid].append(c)

    out = []
    for rid, meta in by_id.items():
        sent = meta.get("sentiment") or {}
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
        timed = [c for c in clist if c.get("timestamp_valid") and c.get("timestamp_epoch_seconds") is not None]
        first_reply = float(clist[0]["compound"]) if clist and clist[0].get("compound") is not None else None
        last_reply = float(clist[-1]["compound"]) if clist and clist[-1].get("compound") is not None else None
        post = sent.get("combined_compound")
        author_mean = float(np.mean(author_vals)) if author_vals else None
        community_mean = float(np.mean(community_vals)) if community_vals else None
        post_to_author = (author_mean - float(post)) if author_mean is not None and post is not None else None
        author_minus_community = (
            (author_mean - community_mean) if author_mean is not None and community_mean is not None else None
        )
        author_shift = early_late_or_none(author_vals)
        last_minus_first = (last_reply - first_reply) if first_reply is not None and last_reply is not None and len(clist) >= 2 else None
        # elapsed hours among timed
        elapsed = None
        t_first_reply_hours = None
        if len(timed) >= 2:
            ts = [float(c["timestamp_epoch_seconds"]) for c in timed]
            elapsed = (max(ts) - min(ts)) / 3600.0
        # variability
        all_vals = [float(c["compound"]) for c in clist if c.get("compound") is not None]
        variability = float(np.std(all_vals, ddof=1)) if len(all_vals) >= 2 else None
        out.append(
            {
                "reddit_id": rid,
                "final_discussion": meta.get("final_discussion"),
                "decision_source": meta.get("decision_source"),
                "selection_stratum": meta.get("selection_stratum"),
                "count_gap": sent.get("count_gap"),
                "n_author": len(author_vals),
                "n_community": len(community_vals),
                "n_replies": len(all_vals),
                "n_timed": len(timed),
                "combined_compound": post,
                "author_mean": author_mean,
                "community_mean": community_mean,
                "post_to_author_mean": post_to_author,
                "author_minus_community": author_minus_community,
                "author_shift": author_shift,
                "first_reply_compound": first_reply,
                "last_reply_compound": last_reply,
                "last_minus_first": last_minus_first,
                "elapsed_hours_timed_replies": elapsed,
                "sentiment_variability": variability,
                "order_slope_normalized": sent.get("order_slope_normalized"),
            }
        )
    return out


def run_phase_d() -> dict[str, Any]:
    dest = OUT_ROOT / "phase_d_cadence"
    dest.mkdir(parents=True, exist_ok=True)
    frame = load_jsonl(DATASET_JSONL)
    comments = load_jsonl(COMMENT_SCORES)
    threads = _thread_summaries(frame, comments)
    tps = [t for t in threads if t["final_discussion"] == "TPS"]
    gds = [t for t in threads if t["final_discussion"] == "GDS"]
    other = [t for t in threads if t["final_discussion"] == "Other"]

    def vals(rows: list[dict[str, Any]], key: str) -> list[float]:
        return [float(r[key]) for r in rows if r.get(key) is not None]

    fields = (
        "post_to_author_mean",
        "author_minus_community",
        "author_shift",
        "last_minus_first",
        "order_slope_normalized",
        "sentiment_variability",
        "elapsed_hours_timed_replies",
        "author_mean",
        "community_mean",
    )
    contrasts = {
        f: stratified_mean_diff(
            vals(gds, f),
            vals(tps, f),
            n_boot=N_BOOT,
            seed=SEED,
            difference_name="gds_minus_tps",
        )
        for f in fields
    }

    # Eligibility table
    def elig(rows: list[dict[str, Any]]) -> dict[str, int]:
        return {
            "n_rows": len(rows),
            **{f: sum(1 for r in rows if r.get(f) is not None) for f in fields},
            "author_ge1": sum(1 for r in rows if int(r.get("n_author") or 0) >= 1),
            "author_ge4": sum(1 for r in rows if int(r.get("n_author") or 0) >= 4),
            "both_roles": sum(
                1 for r in rows if int(r.get("n_author") or 0) >= 1 and int(r.get("n_community") or 0) >= 1
            ),
        }

    eligibility = {"TPS": elig(tps), "GDS": elig(gds), "Other": elig(other)}

    # Sensitivity: count_gap / existing source
    def slice_contrast(name: str, pred) -> dict[str, Any]:
        tt = [r for r in tps if pred(r)]
        gg = [r for r in gds if pred(r)]
        return {
            "slice": name,
            "n_tps": len(tt),
            "n_gds": len(gg),
            "post_to_author_mean": stratified_mean_diff(
                vals(gg, "post_to_author_mean"),
                vals(tt, "post_to_author_mean"),
                n_boot=N_BOOT,
                seed=SEED,
                difference_name="gds_minus_tps",
            ),
            "author_shift": stratified_mean_diff(
                vals(gg, "author_shift"),
                vals(tt, "author_shift"),
                n_boot=N_BOOT,
                seed=SEED,
                difference_name="gds_minus_tps",
            ),
        }

    sensitivities = [
        slice_contrast("count_gap", lambda r: r.get("count_gap") is True),
        slice_contrast("no_count_gap", lambda r: r.get("count_gap") is False),
        slice_contrast(
            "existing_decision_sources",
            lambda r: r.get("decision_source") != "gpt-6.1-terra_ai_classification",
        ),
        slice_contrast(
            "terra_only",
            lambda r: r.get("decision_source") == "gpt-6.1-terra_ai_classification",
        ),
        slice_contrast("october_2026", lambda r: r.get("selection_stratum") == "october_2026"),
        slice_contrast("older", lambda r: r.get("selection_stratum") == "older"),
    ]

    # Elapsed vs last_minus_first association within GDS/TPS (descriptive)
    associations = {}
    for label, rows in (("TPS", tps), ("GDS", gds)):
        dual = [
            r
            for r in rows
            if r.get("elapsed_hours_timed_replies") is not None and r.get("last_minus_first") is not None
        ]
        if len(dual) >= 5:
            associations[label] = bootstrap_spearman(
                [float(r["elapsed_hours_timed_replies"]) for r in dual],
                [float(r["last_minus_first"]) for r in dual],
                n_boot=N_BOOT,
                seed=SEED,
            )
            associations[label]["n"] = len(dual)
        else:
            associations[label] = {"n": len(dual), "spearman": None}

    report = {
        "utc": utcnow(),
        "unit": "one_thread_per_post",
        "bootstrap": {"n_boot": N_BOOT, "seed": SEED},
        "eligibility": eligibility,
        "gds_minus_tps": contrasts,
        "sensitivities": sensitivities,
        "elapsed_vs_last_minus_first": associations,
        "definitions": {
            "author_shift": "later_minus_earlier_author_replies_ge4",
            "post_to_author_mean": "author_mean_minus_combined_post_ge1",
            "author_minus_community": "within_thread_paired_when_both_observed",
        },
        "not_resolution": True,
    }
    write_json(dest / "CADENCE_ANALYSIS.json", report)

    import csv

    with (dest / "gds_minus_tps_cadence.csv").open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(
            handle,
            fieldnames=["field", "mean_diff", "ci_low", "ci_high", "n_gds", "n_tps", "cliffs_delta"],
        )
        w.writeheader()
        for field, stats in contrasts.items():
            ci = stats.get("mean_diff_ci95") or [None, None]
            w.writerow(
                {
                    "field": field,
                    "mean_diff": stats.get("mean_diff"),
                    "ci_low": ci[0],
                    "ci_high": ci[1],
                    "n_gds": stats.get("n_a"),
                    "n_tps": stats.get("n_b"),
                    "cliffs_delta": stats.get("cliffs_delta"),
                }
            )

    def fmt(s: dict[str, Any] | None) -> str:
        if not s or s.get("mean_diff") is None:
            return "n/a"
        ci = s.get("mean_diff_ci95") or [None, None]
        return f"{s['mean_diff']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] (n={s.get('n_a')}/{s.get('n_b')})"

    (dest / "FINDINGS.md").write_text(
        f"""# Phase D findings — author/community trajectory & cadence

**UTC:** {report['utc']}  
Unit: one frozen-relevant thread per post. Sentiment change ≠ issue resolution.

## Eligibility
- TPS author≥1 / ≥4 / both roles: **{eligibility['TPS']['author_ge1']} / {eligibility['TPS']['author_ge4']} / {eligibility['TPS']['both_roles']}**
- GDS author≥1 / ≥4 / both roles: **{eligibility['GDS']['author_ge1']} / {eligibility['GDS']['author_ge4']} / {eligibility['GDS']['both_roles']}**

## GDS − TPS (thread bootstrap)
- post→author mean: {fmt(contrasts['post_to_author_mean'])}
- author−community (paired): {fmt(contrasts['author_minus_community'])}
- author_shift (≥4): {fmt(contrasts['author_shift'])}
- last−first reply: {fmt(contrasts['last_minus_first'])}
- order slope: {fmt(contrasts['order_slope_normalized'])}
- elapsed hours (timed replies): {fmt(contrasts['elapsed_hours_timed_replies'])}

Sensitivities by count-gap, provenance, and period are in `CADENCE_ANALYSIS.json`.
""",
        encoding="utf-8",
    )
    (dest / "LIMITATIONS.md").write_text(
        """# Phase D limitations

- Capture windows censor unobserved later replies.
- Count gaps mean stored comments may be incomplete.
- Mixed label provenance; exploratory until human validation.
- Do not interpret rising sentiment as technical resolution.
""",
        encoding="utf-8",
    )
    return report
