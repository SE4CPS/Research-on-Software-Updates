"""Stage 0–2 runner: timestamp trajectory package (no manuscript)."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import ELAPSED_BINS_HOURS, MIN_THREADS_PER_CELL, N_BOOT, RUN_ID, SEED
from .common import (
    bin_label,
    bootstrap_gds_minus_tps,
    bootstrap_mean_ci,
    bootstrap_paired_mean_ci,
    load_jsonl,
    parse_created_utc,
    sha256_file,
    sha256_text,
    stable_seed,
    write_json,
    write_jsonl,
)
from . import paths as P
from .figures import make_all_figures
from .prior_trace import run_prior_trace


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def load_cohorts() -> dict[str, Any]:
    frozen: dict[str, str] = {}
    with P.FULL_DATASET.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if (
                r.get("final_relevance") == "relevant"
                and r.get("final_discussion") in {"TPS", "GDS"}
                and r.get("sentiment_definition") == P.FROZEN_DEF
            ):
                frozen[r["reddit_id"]] = r["final_discussion"]

    ai: dict[str, str] = {}
    historical_only: list[str] = []
    union_ids: set[str] = set()
    origin_counts: Counter = Counter()
    with P.UNION_AI.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rid = r["reddit_id"]
            union_ids.add(rid)
            origin_counts[r.get("record_origin", "unknown")] += 1
            if (
                r.get("record_origin") == "ai_assisted_screen_candidate_v2"
                and r.get("final_discussion") in {"TPS", "GDS"}
            ):
                ai[rid] = r["final_discussion"]
            if r.get("record_origin") in {
                "historical_preserved",
                "historical_only",
                "historical",
            } or r.get("broad_frame") is False:
                pass

    # Identify historical-only: union rows without broad post_scores
    post_ids: set[str] = set()
    with P.POST_SCORES.open(encoding="utf-8") as f:
        for line in f:
            post_ids.add(json.loads(line)["reddit_id"])

    for rid in union_ids:
        if rid not in post_ids:
            historical_only.append(rid)

    return {
        "frozen": frozen,
        "ai": ai,
        "union_ids": union_ids,
        "union_n": len(union_ids),
        "historical_only_ids": sorted(historical_only),
        "n_historical_only": len(historical_only),
        "origin_counts": dict(origin_counts),
        "post_ids": post_ids,
    }


def load_thread_metrics() -> dict[str, dict]:
    out = {}
    with P.THREAD_METRICS.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["reddit_id"]] = r
    return out


def load_post_scores(need: set[str]) -> dict[str, dict]:
    out = {}
    with P.POST_SCORES.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["reddit_id"] in need:
                out[r["reddit_id"]] = r
    return out


def load_post_created_utc(need: set[str]) -> dict[str, float]:
    """Map reddit_id -> post created epoch from capture pages."""
    out: dict[str, float] = {}
    remaining = set(need)
    for page in sorted(P.CAPTURE_DIR.glob("page_*.json")):
        if not remaining:
            break
        data = json.loads(page.read_text(encoding="utf-8"))
        for post in data.get("data") or []:
            rid = post.get("redditId") or post.get("reddit_id")
            if rid not in remaining:
                continue
            epoch = parse_created_utc(post.get("created_utc"))
            if epoch is not None:
                out[rid] = epoch
                remaining.discard(rid)
    return out


def load_comments(need: set[str]) -> dict[str, list[dict]]:
    by: dict[str, list[dict]] = defaultdict(list)
    with P.COMMENT_SCORES.open(encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if c["reddit_id"] not in need:
                continue
            by[c["reddit_id"]].append(c)
    # stable chronological order: prefer timestamp, then original_index
    for rid, rows in by.items():
        rows.sort(
            key=lambda x: (
                x.get("timestamp_epoch_seconds") is None,
                x.get("timestamp_epoch_seconds") if x.get("timestamp_epoch_seconds") is not None else 0,
                x.get("original_index") if x.get("original_index") is not None else 10**9,
                x.get("chronological_index") if x.get("chronological_index") is not None else 10**9,
            )
        )
    return by


def role_ordered(comments: list[dict], role: str) -> list[dict]:
    rows = [
        c
        for c in comments
        if c.get("role") == role
        and c.get("compound") is not None
        and c.get("timestamp_valid") is True
        and c.get("timestamp_epoch_seconds") is not None
    ]
    return rows


def expected_gate(cohorts: dict, tm: dict[str, dict]) -> dict[str, Any]:
    failures = []
    eligibility: dict[str, Any] = {}

    if cohorts["union_n"] != P.EXPECTED["union_n"]:
        failures.append(
            f"union_n={cohorts['union_n']} != expected {P.EXPECTED['union_n']}"
        )

    for name, mapping in (("frozen", cohorts["frozen"]), ("ai", cohorts["ai"])):
        exp = P.EXPECTED[name]
        counts = Counter(mapping.values())
        cell = {
            "TPS": counts.get("TPS", 0),
            "GDS": counts.get("GDS", 0),
            "author_ge1": {"TPS": 0, "GDS": 0},
            "author_ge2": {"TPS": 0, "GDS": 0},
            "both_roles": {"TPS": 0, "GDS": 0},
            "missing_tm": 0,
        }
        for rid, lab in mapping.items():
            m = tm.get(rid)
            if m is None:
                cell["missing_tm"] += 1
                continue
            na = int(m.get("n_author_replies_nonempty") or 0)
            nc = int(m.get("n_community_replies_nonempty") or 0)
            if na >= 1:
                cell["author_ge1"][lab] += 1
            if na >= 2:
                cell["author_ge2"][lab] += 1
            if na >= 1 and nc >= 1:
                cell["both_roles"][lab] += 1

        eligibility[name] = cell
        for lab in ("TPS", "GDS"):
            if cell[lab] != exp[lab]:
                failures.append(f"{name} {lab}={cell[lab]} != {exp[lab]}")
            for key in ("author_ge1", "author_ge2", "both_roles"):
                if cell[key][lab] != exp[key][lab]:
                    failures.append(
                        f"{name} {key}[{lab}]={cell[key][lab]} != {exp[key][lab]}"
                    )

    # duplicate ID gate within cohorts
    if len(cohorts["frozen"]) != len(set(cohorts["frozen"])):
        failures.append("frozen duplicate reddit_id")
    if len(cohorts["ai"]) != len(set(cohorts["ai"])):
        failures.append("ai duplicate reddit_id")

    return {"ok": not failures, "failures": failures, "eligibility": eligibility}


def protected_check(union_ids: set[str]) -> dict[str, Any]:
    raw = json.loads(P.PROTECTED_IDS.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "union_ids" in raw:
        ids = set(raw["union_ids"])
    elif isinstance(raw, dict) and "ids" in raw:
        ids = set(raw["ids"])
    elif isinstance(raw, list):
        ids = set(raw)
    else:
        ids = set(raw.get("reddit_ids") or raw.get("protected_ids") or [])
    # Overlap with analytical purpose union (should be 0 for train/screen isolation)
    overlap = sorted(ids & union_ids)
    return {
        "source": str(P.PROTECTED_IDS),
        "source_sha256": sha256_file(P.PROTECTED_IDS),
        "n_protected_ids": len(ids),
        "expected_n": P.EXPECTED["protected_n"],
        "n_overlap_analytical_union": len(overlap),
        "outcomes_loaded": False,
        "note": "ID-only overlap check; protected outcome labels were not read.",
        "ok_count": len(ids) == P.EXPECTED["protected_n"],
    }


def build_event_rows(
    *,
    cohort: str,
    labels: dict[str, str],
    posts: dict[str, dict],
    post_ts: dict[str, float],
    comments_by: dict[str, list[dict]],
    text_hashes: dict[str, str],
) -> list[dict]:
    rows: list[dict] = []
    for rid, lab in labels.items():
        ps = posts.get(rid)
        if ps is None:
            continue
        pts = post_ts.get(rid)
        th = text_hashes.get(rid)
        rows.append(
            {
                "reddit_id": rid,
                "cohort": cohort,
                "label": lab,
                "event_kind": "post",
                "event_index_within_role": 0,
                "role": "post",
                "capture_page": ps.get("capture_page"),
                "original_index": ps.get("capture_row_index"),
                "comment_key": None,
                "timestamp_epoch_seconds": pts,
                "timestamp_valid": pts is not None,
                "elapsed_hours": 0.0 if pts is not None else None,
                "text_hash": th,
                "vader_compound": ps.get("combined_compound"),
                "vader_present": ps.get("combined_compound") is not None,
                "model": "VADER",
            }
        )
        for role in ("author", "community"):
            ordered = role_ordered(comments_by.get(rid, []), role)
            for i, c in enumerate(ordered, start=1):
                cts = c.get("timestamp_epoch_seconds")
                elapsed = None
                if pts is not None and cts is not None:
                    elapsed = (float(cts) - float(pts)) / 3600.0
                rows.append(
                    {
                        "reddit_id": rid,
                        "cohort": cohort,
                        "label": lab,
                        "event_kind": "comment",
                        "event_index_within_role": i,
                        "role": role,
                        "capture_page": c.get("capture_page"),
                        "original_index": c.get("original_index"),
                        "comment_key": c.get("comment_key"),
                        "timestamp_epoch_seconds": cts,
                        "timestamp_valid": bool(c.get("timestamp_valid")),
                        "elapsed_hours": elapsed,
                        "text_hash": None,
                        "vader_compound": c.get("compound"),
                        "vader_present": c.get("compound") is not None,
                        "model": "VADER",
                    }
                )
    return rows


def analyze_fixed_eligibility(
    *,
    cohort: str,
    labels: dict[str, str],
    posts: dict[str, dict],
    comments_by: dict[str, list[dict]],
    post_ts: dict[str, float],
) -> dict[str, Any]:
    """Event0→first→second author path on fixed ≥2 set; plus ≥1 first−post reconciliation."""

    def collect(min_author: int):
        out = {"TPS": [], "GDS": []}
        for rid, lab in labels.items():
            ps = posts.get(rid)
            if ps is None or ps.get("combined_compound") is None:
                continue
            if rid not in post_ts:
                continue
            authors = role_ordered(comments_by.get(rid, []), "author")
            # require non-negative elapsed for chronology validity
            valid = []
            for a in authors:
                el = (float(a["timestamp_epoch_seconds"]) - float(post_ts[rid])) / 3600.0
                if el < 0:
                    continue
                valid.append((el, a))
            if len(valid) < min_author:
                continue
            valid.sort(key=lambda x: (x[0], x[1].get("original_index") or 0))
            rec = {
                "reddit_id": rid,
                "label": lab,
                "post": float(ps["combined_compound"]),
                "first": float(valid[0][1]["compound"]),
                "first_elapsed_h": float(valid[0][0]),
            }
            if min_author >= 2:
                rec["second"] = float(valid[1][1]["compound"])
                rec["second_elapsed_h"] = float(valid[1][0])
            out[lab].append(rec)
        return out

    ge2 = collect(2)
    ge1 = collect(1)

    points = {}
    contrasts = {}
    for lab in ("TPS", "GDS"):
        rows = ge2[lab]
        for key in ("post", "first", "second"):
            vals = np.array([r[key] for r in rows], dtype=float)
            points[f"{lab}_{key}"] = bootstrap_mean_ci(
                vals, seed=stable_seed(SEED, cohort, lab, key), n_boot=N_BOOT
            )
            points[f"{lab}_{key}"]["point"] = key
            points[f"{lab}_{key}"]["label"] = lab
            points[f"{lab}_{key}"]["cohort"] = cohort
            points[f"{lab}_{key}"]["eligibility"] = "author_ge2_fixed"
        if rows:
            d1 = np.array([r["first"] - r["post"] for r in rows], dtype=float)
            d2 = np.array([r["second"] - r["first"] for r in rows], dtype=float)
            contrasts[f"{lab}_first_minus_post_ge2"] = bootstrap_paired_mean_ci(
                d1, seed=SEED + 11, n_boot=N_BOOT
            )
            contrasts[f"{lab}_second_minus_first_ge2"] = bootstrap_paired_mean_ci(
                d2, seed=SEED + 12, n_boot=N_BOOT
            )

    # RQ2-style GDS−TPS of first−post on ≥1 cohort
    def deltas(rows):
        return np.array([r["first"] - r["post"] for r in rows], dtype=float)

    rq2_ge1 = bootstrap_gds_minus_tps(
        deltas(ge1["GDS"]),
        deltas(ge1["TPS"]),
        seed=SEED + 21,
        n_boot=N_BOOT,
    )
    rq2_ge1["estimand"] = "GDS_minus_TPS_of_(first_author_minus_post)_on_author_ge1"
    rq2_ge1["cohort"] = cohort
    rq2_ge1["note"] = (
        "Reconciles with primary RQ2 when cohort=ai; ge2 connected path is a different estimand."
    )

    rq2_ge2 = bootstrap_gds_minus_tps(
        deltas(ge2["GDS"]),
        deltas(ge2["TPS"]),
        seed=SEED + 22,
        n_boot=N_BOOT,
    )
    rq2_ge2["estimand"] = "GDS_minus_TPS_of_(first_author_minus_post)_on_author_ge2_fixed"
    rq2_ge2["cohort"] = cohort

    # elapsed summaries for ge2
    elapsed = {}
    for lab in ("TPS", "GDS"):
        rows = ge2[lab]
        if not rows:
            elapsed[lab] = None
            continue
        e1 = np.array([r["first_elapsed_h"] for r in rows], dtype=float)
        e2 = np.array([r["second_elapsed_h"] for r in rows], dtype=float)
        elapsed[lab] = {
            "n": len(rows),
            "first_elapsed_h_median": float(np.median(e1)),
            "first_elapsed_h_mean": float(np.mean(e1)),
            "second_elapsed_h_median": float(np.median(e2)),
            "second_elapsed_h_mean": float(np.mean(e2)),
        }

    # plotting rows (machine-readable)
    plot_rows = []
    for lab in ("TPS", "GDS"):
        for key, xlab in (("post", "Event 0 (post)"), ("first", "First author reply"), ("second", "Second author reply")):
            st = points[f"{lab}_{key}"]
            plot_rows.append(
                {
                    "cohort": cohort,
                    "label": lab,
                    "point": key,
                    "x_label": xlab,
                    "n": st["n"],
                    "mean": st["mean"],
                    "median": st["median"],
                    "ci95_low": st["ci95_low"],
                    "ci95_high": st["ci95_high"],
                    "eligibility": "author_ge2_fixed",
                    "scale": "VADER compound [-1,1]",
                }
            )

    return {
        "n_ge2": {lab: len(ge2[lab]) for lab in ("TPS", "GDS")},
        "n_ge1": {lab: len(ge1[lab]) for lab in ("TPS", "GDS")},
        "points": points,
        "contrasts": contrasts,
        "rq2_style_ge1": rq2_ge1,
        "rq2_style_ge2": rq2_ge2,
        "elapsed_summary_ge2": elapsed,
        "plot_rows": plot_rows,
        "unit_ids_ge2": {
            lab: [r["reddit_id"] for r in ge2[lab]] for lab in ("TPS", "GDS")
        },
    }


def analyze_elapsed(
    *,
    cohort: str,
    labels: dict[str, str],
    posts: dict[str, dict],
    comments_by: dict[str, list[dict]],
    post_ts: dict[str, float],
    bins: list[tuple[float, float, str]],
) -> dict[str, Any]:
    """Equal-thread-weight role means per elapsed bin; missing stays missing."""

    # Coverage-only pass (counts before sentiment aggregation is used for merges)
    coverage = {
        lab: {role: {b[2]: {"threads": 0, "comments": 0} for b in bins} for role in ("author", "community")}
        for lab in ("TPS", "GDS")
    }
    # thread -> role -> bin -> list of compounds
    thread_bin: dict[str, dict[str, dict[str, list[float]]]] = {}
    event0 = {"TPS": [], "GDS": []}

    for rid, lab in labels.items():
        ps = posts.get(rid)
        pts = post_ts.get(rid)
        if ps is None or pts is None or ps.get("combined_compound") is None:
            continue
        event0[lab].append(float(ps["combined_compound"]))
        tb: dict[str, dict[str, list[float]]] = {"author": defaultdict(list), "community": defaultdict(list)}
        for role in ("author", "community"):
            for c in role_ordered(comments_by.get(rid, []), role):
                el = (float(c["timestamp_epoch_seconds"]) - float(pts)) / 3600.0
                if el < 0:
                    continue
                bl = bin_label(el, bins)
                if bl is None:
                    continue
                tb[role][bl].append(float(c["compound"]))
        thread_bin[rid] = tb
        for role in ("author", "community"):
            for bl, vals in tb[role].items():
                if vals:
                    coverage[lab][role][bl]["threads"] += 1
                    coverage[lab][role][bl]["comments"] += len(vals)

    # Merge rule: if any purpose×role cell < MIN across a bin, mark sparse (do not auto-merge after sentiment)
    # Predeclared bins kept; sparse cells reported and omitted from curves.
    sparse_cells = []
    for lab in ("TPS", "GDS"):
        for role in ("author", "community"):
            for _, _, bl in bins:
                n = coverage[lab][role][bl]["threads"]
                if 0 < n < MIN_THREADS_PER_CELL:
                    sparse_cells.append({"label": lab, "role": role, "bin": bl, "n_threads": n})
                elif n == 0:
                    sparse_cells.append({"label": lab, "role": role, "bin": bl, "n_threads": 0})

    plot_rows = []
    stats = {}
    # Event 0 (post) — one point per purpose, not a role reply
    for lab in ("TPS", "GDS"):
        st = bootstrap_mean_ci(
            np.array(event0[lab], dtype=float),
            seed=stable_seed(SEED, cohort, lab, "e0"),
            n_boot=N_BOOT,
        )
        stats[f"{lab}_event0"] = st
        plot_rows.append(
            {
                "cohort": cohort,
                "label": lab,
                "role": "post",
                "bin": "Event0",
                "bin_mid_hours": 0.0,
                "n_threads": st["n"],
                "n_comments": st["n"],
                "mean": st["mean"],
                "median": st["median"],
                "ci95_low": st["ci95_low"],
                "ci95_high": st["ci95_high"],
                "sparse": False,
                "scale": "VADER compound [-1,1]",
            }
        )

    bin_mids = {"0-6h": 3.0, "6-24h": 15.0, "1-3d": 48.0, ">3d": 168.0}

    for lab in ("TPS", "GDS"):
        ids = [rid for rid, l in labels.items() if l == lab and rid in thread_bin]
        for role in ("author", "community"):
            for _, _, bl in bins:
                # one thread mean per bin
                vals = []
                n_comments = 0
                for rid in ids:
                    xs = thread_bin[rid][role].get(bl) or []
                    if xs:
                        vals.append(float(np.mean(xs)))
                        n_comments += len(xs)
                sparse = len(vals) < MIN_THREADS_PER_CELL
                st = bootstrap_mean_ci(
                    np.array(vals, dtype=float),
                    seed=stable_seed(SEED, cohort, lab, role, bl),
                    n_boot=N_BOOT,
                )
                key = f"{lab}_{role}_{bl}"
                stats[key] = {**st, "n_comments": n_comments, "sparse": sparse}
                plot_rows.append(
                    {
                        "cohort": cohort,
                        "label": lab,
                        "role": role,
                        "bin": bl,
                        "bin_mid_hours": bin_mids[bl],
                        "n_threads": st["n"],
                        "n_comments": n_comments,
                        "mean": st["mean"],
                        "median": st["median"],
                        "ci95_low": st["ci95_low"],
                        "ci95_high": st["ci95_high"],
                        "sparse": sparse,
                        "scale": "VADER compound [-1,1]",
                    }
                )

    # Exploratory within-thread gap where both roles present in bin
    gap_rows = []
    gap_stats = {}
    for lab in ("TPS", "GDS"):
        ids = [rid for rid, l in labels.items() if l == lab and rid in thread_bin]
        for _, _, bl in bins:
            gaps = []
            for rid in ids:
                a = thread_bin[rid]["author"].get(bl) or []
                c = thread_bin[rid]["community"].get(bl) or []
                if a and c:
                    gaps.append(float(np.mean(a) - np.mean(c)))
            sparse = len(gaps) < MIN_THREADS_PER_CELL
            st = bootstrap_mean_ci(
                np.array(gaps, dtype=float),
                seed=stable_seed(SEED, cohort, lab, "gap", bl),
                n_boot=N_BOOT,
            )
            gap_stats[f"{lab}_{bl}"] = {**st, "sparse": sparse, "exploratory": True}
            gap_rows.append(
                {
                    "cohort": cohort,
                    "label": lab,
                    "bin": bl,
                    "bin_mid_hours": bin_mids[bl],
                    "n_threads_both_roles": st["n"],
                    "mean_author_minus_community": st["mean"],
                    "ci95_low": st["ci95_low"],
                    "ci95_high": st["ci95_high"],
                    "sparse": sparse,
                    "exploratory": True,
                }
            )

    return {
        "bins": [{"lo": a, "hi": b, "label": c} for a, b, c in bins],
        "min_threads_per_cell": MIN_THREADS_PER_CELL,
        "coverage": coverage,
        "sparse_cells": sparse_cells,
        "stats": stats,
        "plot_rows": plot_rows,
        "gap_exploratory": {"stats": gap_stats, "plot_rows": gap_rows},
        "n_event0": {lab: len(event0[lab]) for lab in ("TPS", "GDS")},
    }


def analyze_four_model() -> dict[str, Any]:
    rows = []
    for path, cohort in ((P.FOUR_MODEL_AI, "ai"), (P.FOUR_MODEL_FROZEN, "frozen")):
        for r in load_jsonl(path):
            rows.append(r)
    if len(rows) != P.EXPECTED["four_model_n"]:
        raise SystemExit(
            f"four-model matched n={len(rows)} != expected {P.EXPECTED['four_model_n']}"
        )

    models = [
        ("vader", "post_vader", "reply_vader", "VADER compound", "continuous"),
        ("textblob", "post_textblob", "reply_textblob", "TextBlob polarity", "continuous"),
        ("cardiff", "post_cardiff_sign", "reply_cardiff_sign", "Cardiff sign {-1,0,+1}", "sign"),
        ("senti4sd", "post_senti4sd_sign", "reply_senti4sd_sign", "Senti4SD sign {-1,0,+1}", "sign"),
    ]
    out_stats = {}
    plot_rows = []
    for cohort in ("ai", "frozen"):
        subset = [r for r in rows if (r.get("cohort") in {cohort, "ai_assisted_v2", "original_frozen"})]
        # normalize cohort filter
        if cohort == "ai":
            subset = [r for r in rows if r.get("cohort") == "ai_assisted_v2"]
        else:
            subset = [r for r in rows if r.get("cohort") == "original_frozen"]
        for lab in ("TPS", "GDS"):
            lab_rows = [r for r in subset if r.get("label") == lab]
            for mid, post_k, reply_k, scale, kind in models:
                posts = np.array([float(r[post_k]) for r in lab_rows if r.get(post_k) is not None])
                replies = np.array([float(r[reply_k]) for r in lab_rows if r.get(reply_k) is not None])
                # pairs only where both present
                pairs = [
                    (float(r[post_k]), float(r[reply_k]))
                    for r in lab_rows
                    if r.get(post_k) is not None and r.get(reply_k) is not None
                ]
                pv = np.array([a for a, _ in pairs], dtype=float)
                rv = np.array([b for _, b in pairs], dtype=float)
                st_p = bootstrap_mean_ci(pv, seed=stable_seed(SEED, cohort, lab, mid, "p"), n_boot=N_BOOT)
                st_r = bootstrap_mean_ci(rv, seed=stable_seed(SEED, cohort, lab, mid, "r"), n_boot=N_BOOT)
                d = rv - pv
                st_d = bootstrap_paired_mean_ci(d, seed=stable_seed(SEED, cohort, lab, mid, "d"), n_boot=N_BOOT)
                key = f"{cohort}_{lab}_{mid}"
                out_stats[key] = {
                    "n": len(pairs),
                    "post": st_p,
                    "first_author": st_r,
                    "first_minus_post": st_d,
                    "scale": scale,
                    "scale_kind": kind,
                }
                for point, st, xlab in (
                    ("post", st_p, "Event 0 (post)"),
                    ("first_author", st_r, "First author reply"),
                ):
                    plot_rows.append(
                        {
                            "cohort": cohort,
                            "label": lab,
                            "model": mid,
                            "scale": scale,
                            "scale_kind": kind,
                            "point": point,
                            "x_label": xlab,
                            "n": st["n"],
                            "mean": st["mean"],
                            "ci95_low": st["ci95_low"],
                            "ci95_high": st["ci95_high"],
                        }
                    )
    # GDS-TPS of delta for AI VADER (sanity vs RQ2)
    ai = [r for r in rows if r.get("cohort") == "ai_assisted_v2"]
    g = np.array([float(r["reply_vader"]) - float(r["post_vader"]) for r in ai if r["label"] == "GDS"])
    t = np.array([float(r["reply_vader"]) - float(r["post_vader"]) for r in ai if r["label"] == "TPS"])
    recon = bootstrap_gds_minus_tps(g, t, seed=SEED + 99, n_boot=N_BOOT)
    recon["note"] = "Four-model matched AI VADER first−post GDS−TPS; should approximate RQ2 on same 5942 pairs."
    return {
        "n_total": len(rows),
        "n_ai": sum(1 for r in rows if r["cohort"] == "ai_assisted_v2"),
        "n_frozen": sum(1 for r in rows if r["cohort"] == "original_frozen"),
        "stats": out_stats,
        "plot_rows": plot_rows,
        "ai_vader_rq2_style": recon,
        "constraint": "Two-event panels only; no multi-reply Cardiff/Senti4SD invented.",
    }


def sensitivity_tables(
    *,
    cohort: str,
    labels: dict[str, str],
    posts: dict[str, dict],
    comments_by: dict[str, list[dict]],
    post_ts: dict[str, float],
) -> dict[str, Any]:
    """Descriptive coverage / depth sensitivities (no unplanned hypothesis tests)."""
    rows = []
    for rid, lab in labels.items():
        ps = posts.get(rid)
        pts = post_ts.get(rid)
        authors = role_ordered(comments_by.get(rid, []), "author") if rid in comments_by else []
        comm = role_ordered(comments_by.get(rid, []), "community") if rid in comments_by else []
        neg_elapsed = 0
        if pts is not None:
            for c in authors + comm:
                el = (float(c["timestamp_epoch_seconds"]) - float(pts)) / 3600.0
                if el < 0:
                    neg_elapsed += 1
        rows.append(
            {
                "reddit_id": rid,
                "label": lab,
                "has_post_score": ps is not None and ps.get("combined_compound") is not None,
                "has_post_timestamp": pts is not None,
                "n_author_scored_timed": len(authors),
                "n_community_scored_timed": len(comm),
                "n_negative_elapsed": neg_elapsed,
                "count_gap": (ps or {}).get("count_gap"),
                "date_band": (ps or {}).get("date_band"),
                "body_len_band": (ps or {}).get("body_len_band"),
            }
        )
    summary = {
        "cohort": cohort,
        "n": len(rows),
        "n_missing_post_timestamp": sum(1 for r in rows if not r["has_post_timestamp"]),
        "n_missing_post_score": sum(1 for r in rows if not r["has_post_score"]),
        "n_any_negative_elapsed": sum(1 for r in rows if r["n_negative_elapsed"] > 0),
        "by_label": {},
    }
    for lab in ("TPS", "GDS"):
        sub = [r for r in rows if r["label"] == lab]
        summary["by_label"][lab] = {
            "n": len(sub),
            "author_ge1": sum(1 for r in sub if r["n_author_scored_timed"] >= 1),
            "author_ge2": sum(1 for r in sub if r["n_author_scored_timed"] >= 2),
            "community_ge1": sum(1 for r in sub if r["n_community_scored_timed"] >= 1),
            "missing_post_ts": sum(1 for r in sub if not r["has_post_timestamp"]),
        }
    # date_band / count_gap distributions among ge2
    ge2 = [r for r in rows if r["n_author_scored_timed"] >= 2]
    summary["ge2_date_band"] = {
        str(k): v for k, v in Counter(r.get("date_band") for r in ge2).items()
    }
    summary["ge2_count_gap"] = dict(Counter(str(r.get("count_gap")) for r in ge2))
    summary["ge2_body_len_band"] = {
        str(k): v for k, v in Counter(r.get("body_len_band") for r in ge2).items()
    }
    return summary


def write_docs(
    out: Path,
    *,
    gate: dict,
    trajectory: dict,
    four_model: dict,
    prior: dict,
    input_hashes: dict,
    gaps: list[str],
):
    (out / "FIGURE_CAPTIONS.md").write_text(
        """# Figure captions (candidate; awaiting Codex validation)

## Figure C — Fixed-eligibility original-author path (VADER)

Mean VADER compound score (native scale [−1, 1]) at Event 0 (original post title+body as one unit), first original-author reply, and second original-author reply, for threads with ≥2 timed author replies. The same thread set appears at all three points. Panels separate AI-assisted and frozen-score cohorts; lines separate TPS and strict GDS. Error bars are 95% whole-thread bootstrap intervals (10,000 resamples). This connected three-point path is not the primary RQ2 estimand; RQ2 uses the larger author≥1 first−post contrast.

## Figure B — Elapsed-time author and community means (VADER)

Thread-equal-weighted mean VADER compound by role within predeclared elapsed-time bins (0–6 h, 6–24 h, 1–3 d, >3 d) measured from the Reddit post `created_utc`. Event 0 marks the post score at elapsed = 0. Sparse cells (n_threads < 20) are shown hollow / noted and must not be over-interpreted. Missing role×bin observations remain missing (not imputed as neutral). Separate AI/frozen and TPS/GDS facets. Uncertainty: 95% whole-thread bootstrap.

## Figure E — Exploratory within-bin author−community gap (optional)

Within-thread author mean minus community mean in the same elapsed bin, requiring both roles observed in that bin. Exploratory only; not a causal resolution claim. If strict-GDS cells are sparse, eligibility is reported and curves are suppressed.

## Figure F — Four-model two-event sensitivity (existing matched pairs)

Event 0 vs first original-author reply for the existing 6,996 matched pairs (5,942 AI + 1,054 frozen) under VADER, TextBlob, Cardiff sign, and Senti4SD sign. Separate native-scale panels; no cross-model magnitude overlay. Multi-reply Cardiff/Senti4SD trajectories are unavailable without new scoring and are not produced.
""",
        encoding="utf-8",
    )

    (out / "LIMITATIONS.md").write_text(
        """# Limitations (trajectory package)

1. VADER multi-point curves use saved comment scores only; Cardiff/Senti4SD lack intermediate comment coverage.
2. Strict GDS sample sizes are small at deeper reply depth and in late elapsed bins; sparse cells are flagged (n < 20 threads).
3. Reply order ≠ conversational reply-to edge; chronology is wall-clock within role.
4. Fixed ≥2 author path conditions on reply depth and is not identical to RQ2’s ≥1 first-change estimand.
5. Missing replies are unobserved — not neutral sentiment and not interpolated.
6. AI-assisted purpose labels are the primary claim setting; frozen-score panels are comparative.
7. Elapsed time uses capture `created_utc` and saved comment timestamps; invalid/negative elapsed values are excluded and counted.
8. No causal interpretation of sentiment shifts as issue resolution.
9. Historical-only union rows lack broad timestamps and are excluded from the trajectory frame.
10. All trajectory results await independent Codex validation.
""",
        encoding="utf-8",
    )

    gaps_txt = "\n".join(f"- {g}" for g in gaps) if gaps else "- None reported at packaging time."
    (out / "RUN.md").write_text(
        f"""# RUN — {RUN_ID}

- UTC: {utc_now()}
- Seed: {SEED}
- Bootstrap resamples: {N_BOOT}
- Entry: `python -m full_resource_sentiment_v1.final_timestamp_trajectory_2026_11_03.run_all`
- Status: **AWAITING_INDEPENDENT_CODEX_VALIDATION**
- Manuscript writing: **not started** (by design)

## Input hashes
{json.dumps(input_hashes, indent=2)}

## Eligibility gate
ok={gate.get("ok")} failures={gate.get("failures")}

## Essential data gaps
{gaps_txt}
""",
        encoding="utf-8",
    )


def main() -> int:
    out = P.OUT
    out_r = P.OUT_RESTRICTED
    out.mkdir(parents=True, exist_ok=True)
    out_r.mkdir(parents=True, exist_ok=True)
    for sub in ("figures", "tables", "plot_data", "tests"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    gaps: list[str] = []

    input_hashes = {
        "union_jsonl": sha256_file(P.UNION_AI),
        "full_dataset_jsonl": sha256_file(P.FULL_DATASET),
        "post_scores": sha256_file(P.POST_SCORES),
        "thread_metrics": sha256_file(P.THREAD_METRICS),
        "comment_scores": sha256_file(P.COMMENT_SCORES),
        "four_model_ai": sha256_file(P.FOUR_MODEL_AI),
        "four_model_frozen": sha256_file(P.FOUR_MODEL_FROZEN),
        "protected_ids": sha256_file(P.PROTECTED_IDS),
        "expected_full_dataset_sha256": "99b575277fb3c9acc4a116ceb2ea9ec44efb2a0875ebdb088240cc2cf2d5aefd",
    }
    if input_hashes["full_dataset_jsonl"] != input_hashes["expected_full_dataset_sha256"]:
        raise SystemExit("full_dataset.jsonl hash mismatch vs sealed expected")

    print("[stage0] loading cohorts…", flush=True)
    cohorts = load_cohorts()
    if cohorts["n_historical_only"] != 520:
        gaps.append(
            f"historical-only count={cohorts['n_historical_only']} (expected 520); "
            "definition=union IDs absent from broad post_scores"
        )

    tm = load_thread_metrics()
    gate = expected_gate(cohorts, tm)
    write_json(out / "ELIGIBILITY_BY_COHORT.json", {**gate, "historical_only_n": cohorts["n_historical_only"]})
    if not gate["ok"]:
        write_json(out / "STAGE0_FAILURE.json", gate)
        print("STAGE0 GATE FAILURE:", gate["failures"], file=sys.stderr)
        return 2

    prot = protected_check(cohorts["union_ids"])
    write_json(out / "PROTECTED_OVERLAP_CHECK.json", prot)

    need = set(cohorts["frozen"]) | set(cohorts["ai"])
    print(f"[stage0] loading posts/comments/timestamps for {len(need)} threads…", flush=True)
    posts = load_post_scores(need)
    post_ts = load_post_created_utc(need)
    comments_by = load_comments(need)

    # text hashes from manifest when available (no full text export)
    text_hashes: dict[str, str] = {}
    if P.TEXT_MANIFEST.exists():
        for r in load_jsonl(P.TEXT_MANIFEST):
            rid = r.get("reddit_id")
            if rid in need:
                # combine title/body hashes if present
                th = r.get("title_sha256") or ""
                bh = r.get("body_sha256") or ""
                if th or bh:
                    text_hashes[rid] = sha256_text(th + "|" + bh)

    ts_cov = {
        "n_need": len(need),
        "n_post_scores": len(posts),
        "n_post_timestamps": len(post_ts),
        "n_threads_with_any_comment_scores": len(comments_by),
        "missing_post_timestamp_frozen": sum(1 for r in cohorts["frozen"] if r not in post_ts),
        "missing_post_timestamp_ai": sum(1 for r in cohorts["ai"] if r not in post_ts),
    }
    write_json(out / "TIMESTAMP_COVERAGE.json", ts_cov)
    if ts_cov["missing_post_timestamp_frozen"] or ts_cov["missing_post_timestamp_ai"]:
        gaps.append(
            f"Missing capture created_utc for "
            f"frozen={ts_cov['missing_post_timestamp_frozen']} ai={ts_cov['missing_post_timestamp_ai']} threads"
        )

    print("[stage0] writing event manifest (restricted; no usernames/full text)…", flush=True)
    event_rows = []
    event_rows.extend(
        build_event_rows(
            cohort="frozen",
            labels=cohorts["frozen"],
            posts=posts,
            post_ts=post_ts,
            comments_by=comments_by,
            text_hashes=text_hashes,
        )
    )
    event_rows.extend(
        build_event_rows(
            cohort="ai",
            labels=cohorts["ai"],
            posts=posts,
            post_ts=post_ts,
            comments_by=comments_by,
            text_hashes=text_hashes,
        )
    )
    n_manifest = write_jsonl(out_r / "event_manifest.jsonl", event_rows)
    write_json(
        out / "EVENT_SCHEMA.json",
        {
            "run_id": RUN_ID,
            "n_rows": n_manifest,
            "join_keys": ["capture_page", "reddit_id", "original_index"],
            "event0": "title+body as one VADER combined_compound unit",
            "fields": [
                "reddit_id",
                "cohort",
                "label",
                "event_kind",
                "event_index_within_role",
                "role",
                "capture_page",
                "original_index",
                "comment_key",
                "timestamp_epoch_seconds",
                "timestamp_valid",
                "elapsed_hours",
                "text_hash",
                "vader_compound",
                "vader_present",
                "model",
            ],
            "public_export": "no usernames or full text",
            "negative_elapsed_policy": "excluded from order/elapsed analyses; counted in sensitivity",
        },
    )

    print("[stage1] fixed-eligibility paths…", flush=True)
    fixed = {
        "ai": analyze_fixed_eligibility(
            cohort="ai",
            labels=cohorts["ai"],
            posts=posts,
            comments_by=comments_by,
            post_ts=post_ts,
        ),
        "frozen": analyze_fixed_eligibility(
            cohort="frozen",
            labels=cohorts["frozen"],
            posts=posts,
            comments_by=comments_by,
            post_ts=post_ts,
        ),
    }
    # Verify ge2 counts match expected author_ge2 when timestamps valid
    for cohort_name, exp_key in (("ai", "ai"), ("frozen", "frozen")):
        got = fixed[cohort_name]["n_ge2"]
        exp = P.EXPECTED[exp_key]["author_ge2"]
        for lab in ("TPS", "GDS"):
            if got[lab] != exp[lab]:
                # May differ if negative elapsed excluded
                gaps.append(
                    f"fixed ge2 {cohort_name} {lab}={got[lab]} vs thread_metrics author_ge2={exp[lab]} "
                    "(difference may reflect invalid/negative elapsed exclusions)"
                )

    print("[stage1] elapsed-time bins…", flush=True)
    elapsed = {
        "ai": analyze_elapsed(
            cohort="ai",
            labels=cohorts["ai"],
            posts=posts,
            comments_by=comments_by,
            post_ts=post_ts,
            bins=ELAPSED_BINS_HOURS,
        ),
        "frozen": analyze_elapsed(
            cohort="frozen",
            labels=cohorts["frozen"],
            posts=posts,
            comments_by=comments_by,
            post_ts=post_ts,
            bins=ELAPSED_BINS_HOURS,
        ),
    }

    print("[stage1] four-model two-event…", flush=True)
    four_model = analyze_four_model()

    print("[stage1] sensitivity coverage…", flush=True)
    sensitivity = {
        "ai": sensitivity_tables(
            cohort="ai",
            labels=cohorts["ai"],
            posts=posts,
            comments_by=comments_by,
            post_ts=post_ts,
        ),
        "frozen": sensitivity_tables(
            cohort="frozen",
            labels=cohorts["frozen"],
            posts=posts,
            comments_by=comments_by,
            post_ts=post_ts,
        ),
    }

    trajectory = {
        "run_id": RUN_ID,
        "utc": utc_now(),
        "seed": SEED,
        "n_boot": N_BOOT,
        "status": "AWAITING_INDEPENDENT_CODEX_VALIDATION",
        "fixed_eligibility_author_path": fixed,
        "elapsed_time": elapsed,
        "four_model_two_event": {
            k: v
            for k, v in four_model.items()
            if k != "plot_rows"
        },
        "sensitivity": sensitivity,
        "gaps": gaps,
    }
    # Strip bulky unit id lists from main JSON but keep in restricted
    slim = json.loads(json.dumps(trajectory))
    for cohort_name in ("ai", "frozen"):
        ids = slim["fixed_eligibility_author_path"][cohort_name].pop("unit_ids_ge2", None)
        write_json(out_r / f"fixed_ge2_ids_{cohort_name}.json", ids)
        slim["fixed_eligibility_author_path"][cohort_name].pop("plot_rows", None)
        slim["elapsed_time"][cohort_name].pop("plot_rows", None)
        if "gap_exploratory" in slim["elapsed_time"][cohort_name]:
            slim["elapsed_time"][cohort_name]["gap_exploratory"].pop("plot_rows", None)
    write_json(out / "TRAJECTORY_RESULTS.json", slim)

    # plot data tables
    import csv

    def write_csv(path: Path, rows: list[dict]):
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    fixed_plot = fixed["ai"]["plot_rows"] + fixed["frozen"]["plot_rows"]
    elapsed_plot = elapsed["ai"]["plot_rows"] + elapsed["frozen"]["plot_rows"]
    gap_plot = (
        elapsed["ai"]["gap_exploratory"]["plot_rows"]
        + elapsed["frozen"]["gap_exploratory"]["plot_rows"]
    )
    write_csv(out / "plot_data" / "fixed_eligibility_author_path.csv", fixed_plot)
    write_csv(out / "plot_data" / "elapsed_time_role_means.csv", elapsed_plot)
    write_csv(out / "plot_data" / "exploratory_author_community_gap.csv", gap_plot)
    write_csv(out / "plot_data" / "four_model_two_event.csv", four_model["plot_rows"])
    write_json(out / "tables" / "FIXED_ELIGIBILITY_AUTHOR_PATH.json", fixed_plot)
    write_json(out / "tables" / "ELAPSED_TIME_ROLE_MEANS.json", elapsed_plot)
    write_json(out / "tables" / "EXPLORATORY_GAP.json", gap_plot)
    write_json(out / "tables" / "FOUR_MODEL_TWO_EVENT.json", four_model["plot_rows"])
    write_json(out / "tables" / "SENSITIVITY_COVERAGE.json", sensitivity)
    write_json(
        out / "tables" / "ELIGIBILITY_COUNTS.json",
        gate["eligibility"],
    )
    _write_eligibility_markdown(
        out / "tables" / "COUNT_ELIGIBILITY.md",
        gate=gate,
        fixed=fixed,
        elapsed=elapsed,
        ts_cov=ts_cov,
        historical_n=cohorts["n_historical_only"],
    )

    print("[stage2] prior-experiment trace…", flush=True)
    prior = run_prior_trace(out)
    write_json(out / "PRIOR_EXPERIMENT_TRACE.json", prior)
    (out / "PRIOR_EXPERIMENT_TRACE.md").write_text(prior["markdown"], encoding="utf-8")

    write_docs(
        out,
        gate=gate,
        trajectory=slim,
        four_model=four_model,
        prior=prior,
        input_hashes=input_hashes,
        gaps=gaps,
    )

    print("[figures] rendering…", flush=True)
    fig_info = make_all_figures(
        out,
        fixed_plot=fixed_plot,
        elapsed_plot=elapsed_plot,
        gap_plot=gap_plot,
        four_plot=four_model["plot_rows"],
    )
    write_json(out / "figures" / "FIGURE_INDEX.json", fig_info)

    # tests
    tests = run_tests(fixed, elapsed, four_model, gate, cohorts)
    write_json(out / "tests" / "TEST_RESULTS.json", tests)

    write_methodology(out)

    write_codex_packet(
        out,
        input_hashes=input_hashes,
        gate=gate,
        fixed=fixed,
        elapsed=elapsed,
        four_model=four_model,
        gaps=gaps,
        tests=tests,
        fig_info=fig_info,
    )

    # hashes last so packet/methodology are included
    file_hashes = {}
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name != "OUTPUT_HASHES.json":
            file_hashes[str(p.relative_to(out))] = sha256_file(p)
    write_json(
        out / "OUTPUT_HASHES.json",
        {"utc": utc_now(), "run_id": RUN_ID, "files": file_hashes, "n_files": len(file_hashes)},
    )

    print(f"DONE → {out}", flush=True)
    print("GAPS:", gaps or "none", flush=True)
    return 0 if tests.get("ok") else 1


def _write_eligibility_markdown(
    path: Path,
    *,
    gate: dict,
    fixed: dict,
    elapsed: dict,
    ts_cov: dict,
    historical_n: int,
) -> None:
    el = gate["eligibility"]
    lines = [
        "# Count / eligibility table (mandatory companion)",
        "",
        f"Historical-only union rows outside broad timestamp frame: **{historical_n}**.",
        "",
        f"Post timestamp coverage among analysis IDs: {ts_cov['n_post_timestamps']}/{ts_cov['n_need']}.",
        f"Threads with any saved comment scores: {ts_cov['n_threads_with_any_comment_scores']}/{ts_cov['n_need']}.",
        "",
        "## Purpose and reply-depth (from thread_metrics; Stage-0 gate)",
        "",
        "| Cohort | TPS | GDS | author≥1 TPS/GDS | author≥2 TPS/GDS | both roles TPS/GDS |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("ai", "frozen"):
        c = el[name]
        lines.append(
            f"| {name} | {c['TPS']} | {c['GDS']} | "
            f"{c['author_ge1']['TPS']}/{c['author_ge1']['GDS']} | "
            f"{c['author_ge2']['TPS']}/{c['author_ge2']['GDS']} | "
            f"{c['both_roles']['TPS']}/{c['both_roles']['GDS']} |"
        )
    lines += [
        "",
        "## Fixed-eligibility path n (same threads at Event0/first/second)",
        "",
        "| Cohort | TPS | GDS |",
        "|---|---:|---:|",
        f"| ai | {fixed['ai']['n_ge2']['TPS']} | {fixed['ai']['n_ge2']['GDS']} |",
        f"| frozen | {fixed['frozen']['n_ge2']['TPS']} | {fixed['frozen']['n_ge2']['GDS']} |",
        "",
        "## Elapsed-time sparse cells (n_threads < 20)",
        "",
    ]
    for cohort in ("ai", "frozen"):
        sparse = elapsed[cohort].get("sparse_cells") or []
        lines.append(f"### {cohort}")
        if not sparse:
            lines.append("- none")
        else:
            for s in sparse:
                lines.append(
                    f"- {s['label']} · {s['role']} · {s['bin']}: n_threads={s['n_threads']}"
                )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_methodology(out: Path) -> None:
    (out / "METHODOLOGY.md").write_text(
        f"""# Methodology notes — {RUN_ID}

## Scope
Bounded VADER timestamp/order trajectory analysis on **existing** broad-capture scores and Reddit timestamps. No new model scoring. No manuscript drafting.

## Event construction
- **Event 0:** original post `title + body` as one text unit; score = saved `combined_compound` from `post_scores.jsonl`.
- **Author vs community:** `comment_scores.jsonl` `role` field (`author` / `community`); unresolved roles excluded from role-specific curves.
- **Join keys:** `capture_page + reddit_id + original_index` (never row position alone).
- **Timestamps:** post `created_utc` from capture pages; comment `timestamp_epoch_seconds` from saved comment scores. Elapsed hours = (comment − post) / 3600. Negative elapsed excluded from analyses and counted in sensitivity.
- **Public artifacts** omit usernames and full text; restricted `event_manifest.jsonl` carries hashes/keys only.

## Cohorts
- **Frozen-score comparable:** `full_dataset.jsonl` with `sentiment_definition = frozen_broad_release_train_package_scores`, relevant TPS/GDS.
- **AI-assisted:** corrected union `record_origin = ai_assisted_screen_candidate_v2`, TPS/GDS.
- **Historical-only (520):** union IDs absent from broad `post_scores` — outside the timestamp frame.

## Estimands
1. **Fixed-eligibility author path:** threads with ≥2 timed scored author replies; same IDs at Event0, first, second. Means/medians + 95% whole-thread bootstrap CIs; paired first−post and second−first. Separately, author≥1 first−post GDS−TPS reconciles with primary RQ2 (not identical to the ≥2 path).
2. **Elapsed-time role means:** predeclared bins 0–6h, 6–24h, 1–3d, >3d chosen from coverage before interpreting sentiment. Each thread contributes at most one equal-weight mean per role×bin. Sparse if n_threads < {MIN_THREADS_PER_CELL}. Missing stays missing.
3. **Exploratory gap:** author mean − community mean within the same thread and bin (both required). Not causal.
4. **Four-model two-event:** reuse existing 6,996 matched pairs only; native scales; no invented intermediate Cardiff/Senti4SD scores.

## Uncertainty
Seed `{SEED}`, `{N_BOOT}` whole-thread bootstrap resamples (stratified by purpose for GDS−TPS contrasts). Process-independent subseeds via SHA-256.

## Non-interpretation
Sentiment shifts are not treated as issue resolution or causal effects of community interaction.
""",
        encoding="utf-8",
    )


def run_tests(fixed, elapsed, four_model, gate, cohorts) -> dict:
    checks = []

    def check(name, cond, detail=""):
        checks.append({"name": name, "ok": bool(cond), "detail": detail})

    check("stage0_gate", gate["ok"], str(gate.get("failures")))
    check("four_model_n", four_model["n_total"] == 6996, str(four_model["n_total"]))
    check("union_n", cohorts["union_n"] == 18478, str(cohorts["union_n"]))

    for cohort in ("ai", "frozen"):
        n = fixed[cohort]["n_ge2"]
        # same n at all three points implied by construction
        for lab in ("TPS", "GDS"):
            rows = [r for r in fixed[cohort]["plot_rows"] if r["label"] == lab]
            ns = {r["point"]: r["n"] for r in rows}
            check(
                f"fixed_same_n_{cohort}_{lab}",
                ns.get("post") == ns.get("first") == ns.get("second") == n[lab],
                str(ns),
            )

    # no NaN fill: sparse elapsed cells may have n=0 mean None
    for cohort in ("ai", "frozen"):
        for r in elapsed[cohort]["plot_rows"]:
            if r["n_threads"] == 0:
                check(
                    f"no_impute_{cohort}_{r['label']}_{r['role']}_{r['bin']}",
                    r["mean"] is None,
                    "mean should be null when n=0",
                )

    # AI RQ2-style ge1 should be near canonical mean -0.1147
    rq = fixed["ai"]["rq2_style_ge1"]
    if rq["mean"] is not None:
        check(
            "ai_rq2_ge1_near_canonical",
            abs(rq["mean"] - (-0.114708820115)) < 1e-6,
            str(rq["mean"]),
        )
        check(
            "ai_rq2_ge1_ns",
            rq["n_gds"] == 610 and rq["n_tps"] == 5332,
            f"{rq['n_gds']}/{rq['n_tps']}",
        )

    ok = all(c["ok"] for c in checks)
    return {"ok": ok, "n_checks": len(checks), "checks": checks, "utc": utc_now()}


def write_codex_packet(
    out: Path,
    *,
    input_hashes,
    gate,
    fixed,
    elapsed,
    four_model,
    gaps,
    tests,
    fig_info,
):
    ai_rq = fixed["ai"]["rq2_style_ge1"]
    lines = [
        f"# CODEX REVIEW PACKET — {RUN_ID}",
        "",
        "**Status:** AWAITING_INDEPENDENT_CODEX_VALIDATION",
        "",
        "Do not treat these trajectory figures as manuscript-ready until Codex verifies them.",
        "",
        "## What was produced",
        "",
        f"- Output tree: `{out}`",
        f"- Restricted event manifest: `{P.OUT_RESTRICTED}/event_manifest.jsonl`",
        "- Primary figures: fixed-eligibility author path (C), elapsed-time author/community (B)",
        "- Secondary: exploratory gap (E) if eligibility allows; four-model two-event (F) from existing 6,996 pairs",
        "",
        "## Exact artifacts Codex must independently verify",
        "",
        "1. `ELIGIBILITY_BY_COHORT.json` — frozen 3028/231 and AI 11054/999 with author≥1/≥2/both counts",
        "2. `EVENT_SCHEMA.json` + restricted `event_manifest.jsonl` — Event 0 = title+body; roles separated; joins by page+id+index",
        "3. `TRAJECTORY_RESULTS.json` — fixed path means/CIs; elapsed bin means/CIs; no imputation",
        "4. `plot_data/fixed_eligibility_author_path.csv` ↔ `figures/fig_C_fixed_eligibility_author_path.png`",
        "5. `plot_data/elapsed_time_role_means.csv` ↔ `figures/fig_B_elapsed_time_author_community.png`",
        "6. `plot_data/four_model_two_event.csv` ↔ `figures/fig_F_four_model_two_event.png`",
        "7. `tables/ELIGIBILITY_COUNTS.json`, `tables/COUNT_ELIGIBILITY.md`, sensitivity coverage tables",
        "8. AI author≥1 first−post GDS−TPS reconciliation vs canonical RQ2 mean −0.114708820115 "
        f"(observed mean={ai_rq.get('mean')}, n_gds={ai_rq.get('n_gds')}, n_tps={ai_rq.get('n_tps')})",
        "9. `PRIOR_EXPERIMENT_TRACE.md` — historical/A/B reuse checks without retraining",
        "10. `METHODOLOGY.md`, `OUTPUT_HASHES.json`, `tests/TEST_RESULTS.json`, `LIMITATIONS.md`, `FIGURE_CAPTIONS.md`",
        "11. `plot_data/exploratory_author_community_gap.csv` ↔ `figures/fig_E_exploratory_author_community_gap.png`",
        "12. Confirm no sealed prior trees, PDFs, labels, or datasets were modified",
        "13. Confirm sparse cells / gaps were reported rather than silently filled",
        "14. Confirm four-model figure uses only existing 6,996 two-event pairs (no invented later scores)",
        "",
        "## Stage-0 gate",
        "",
        f"- ok: `{gate.get('ok')}`",
        f"- failures: `{gate.get('failures')}`",
        "",
        "## Input hashes",
        "",
        "```json",
        json.dumps(input_hashes, indent=2),
        "```",
        "",
        "## Key trajectory n (fixed ge2)",
        "",
        f"- AI TPS/GDS: {fixed['ai']['n_ge2']}",
        f"- Frozen TPS/GDS: {fixed['frozen']['n_ge2']}",
        "",
        "## Essential gaps reported",
        "",
    ]
    if gaps:
        lines.extend(f"- {g}" for g in gaps)
    else:
        lines.append("- None")
    lines += [
        "",
        "## Automated tests",
        "",
        f"- ok: `{tests.get('ok')}` ({tests.get('n_checks')} checks)",
        "",
        "## Figures",
        "",
        "```json",
        json.dumps(fig_info, indent=2),
        "```",
        "",
        "## Non-goals confirmed",
        "",
        "- No manuscript writing",
        "- No new sentiment model scoring",
        "- No interpolation of missing replies as neutral",
        "- No change to earlier manuscripts or sealed results",
        "",
    ]
    (out / "CODEX_REVIEW_PACKET.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
