"""Workstream I cadence/role robustness: I-C, I-D, I-E."""

from __future__ import annotations

from typing import Any

from .inference import bootstrap_spearman
from .robustness_i_common import (
    EXPECTED_N,
    I_N_BOOT,
    I_SEED,
    date_stratum,
    describe_stratum,
    reading_aid,
    timed_comments,
    timed_reply_count_bin,
)


def _copy_g_primary(block: dict[str, Any]) -> dict[str, Any]:
    keys = ("n", "mean", "median", "mean_ci95", "median_ci95", "std", "spearman", "spearman_ci95")
    return {key: block.get(key) for key in keys if key in block}


def _role_cell(n_author: int, n_community: int) -> str:
    a = "1" if n_author == 1 else "ge2"
    c = "1" if n_community == 1 else "ge2"
    return f"author_{a}_community_{c}"


def analysis_i_c(threads: dict[str, dict[str, Any]], g_analysis: dict[str, Any]) -> dict[str, Any]:
    eligible = []
    for rid, thread in threads.items():
        if not thread.get("both_roles_eligible"):
            continue
        if thread.get("author_mean") is None or thread.get("community_mean") is None:
            continue
        n_author = int(thread["n_author_replies_nonempty"])
        n_community = int(thread["n_community_replies_nonempty"])
        ratio = n_author / n_community
        eligible.append(
            {
                "reddit_id": rid,
                "diff": float(thread["community_mean"] - thread["author_mean"]),
                "count_gap": bool(thread.get("count_gap")),
                "date_stratum": date_stratum(thread.get("created_month")),
                "n_author": n_author,
                "n_community": n_community,
                "role_cell": _role_cell(n_author, n_community),
                "balanced": 0.5 <= ratio <= 2.0,
            }
        )
    if len(eligible) != EXPECTED_N["C"]:
        raise ValueError(f"I-C n {len(eligible)} != {EXPECTED_N['C']}")
    gap_strata = {
        "count_gap": describe_stratum([row["diff"] for row in eligible if row["count_gap"]]),
        "no_gap": describe_stratum([row["diff"] for row in eligible if not row["count_gap"]]),
        "older": describe_stratum([row["diff"] for row in eligible if row["date_stratum"] == "older"]),
        "recent_oct2026": describe_stratum([row["diff"] for row in eligible if row["date_stratum"] == "recent_oct2026"]),
    }
    cells = {}
    for name in ("author_1_community_1", "author_1_community_ge2", "author_ge2_community_1", "author_ge2_community_ge2"):
        cells[name] = describe_stratum([row["diff"] for row in eligible if row["role_cell"] == name])
        cells[name]["retained_in_main_estimate"] = True
    balanced = describe_stratum([row["diff"] for row in eligible if row["balanced"]])
    reference = g_analysis["primary_community_minus_author"]
    return {
        "analysis_id": "I-C",
        "tests_g_analysis": "C",
        "seed": I_SEED,
        "n_bootstrap": I_N_BOOT,
        "eligible_n": len(eligible),
        "n_count_gap_retained": sum(1 for row in eligible if row["count_gap"]),
        "g_reference": _copy_g_primary(reference),
        "overall": describe_stratum([row["diff"] for row in eligible]),
        "count_gap_and_date": gap_strata,
        "role_count_cells": cells,
        "balanced_role_ratio_0.5_to_2": balanced,
        "pooling_note": "Thread is the unit; comments are not pooled as independent observations. One-reply-per-role threads remain in the main estimate.",
        "reading_aid": reading_aid(
            reference=reference.get("mean"),
            diagnostic=None,
            strata=gap_strata,
            estimand_note="Same community_mean - author_mean estimand as G C.",
        ),
        "count_gap_exclusion": False,
    }


def analysis_i_d(
    threads: dict[str, dict[str, Any]],
    comments_by_id: dict[str, list[dict[str, Any]]],
    g_analysis: dict[str, Any],
) -> dict[str, Any]:
    eligible = []
    n_unresolved_pair = 0
    for rid, thread in threads.items():
        if not thread.get("order_eligible") or thread.get("last_minus_first") is None:
            continue
        timed = timed_comments(comments_by_id.get(rid) or [])
        if len(timed) < 2:
            raise ValueError(f"D-eligible thread has <2 timed comments: {rid}")
        first_role = timed[0].get("role") or "unresolved"
        last_role = timed[-1].get("role") or "unresolved"
        if first_role == "unresolved" or last_role == "unresolved":
            n_unresolved_pair += 1
        diff = float(thread["last_minus_first"])
        eligible.append(
            {
                "reddit_id": rid,
                "diff": diff,
                "count_gap": bool(thread.get("count_gap")),
                "date_stratum": date_stratum(thread.get("created_month")),
                "n_timed": int(thread["n_timed_nonempty"]),
                "timed_bin": timed_reply_count_bin(int(thread["n_timed_nonempty"])),
                "role_pair": f"{first_role}->{last_role}",
                "abs_ge_010": abs(diff) >= 0.10,
                "abs_ge_030": abs(diff) >= 0.30,
            }
        )
    if len(eligible) != EXPECTED_N["D"]:
        raise ValueError(f"I-D n {len(eligible)} != {EXPECTED_N['D']}")
    overall = describe_stratum([row["diff"] for row in eligible])
    overall["share_abs_ge_0.10"] = sum(row["abs_ge_010"] for row in eligible) / len(eligible)
    overall["share_abs_ge_0.30"] = sum(row["abs_ge_030"] for row in eligible) / len(eligible)
    overall["magnitude_note"] = "0.10 and 0.30 are descriptive markers, not universal importance thresholds."
    gap_strata = {
        "count_gap": describe_stratum([row["diff"] for row in eligible if row["count_gap"]]),
        "no_gap": describe_stratum([row["diff"] for row in eligible if not row["count_gap"]]),
        "older": describe_stratum([row["diff"] for row in eligible if row["date_stratum"] == "older"]),
        "recent_oct2026": describe_stratum([row["diff"] for row in eligible if row["date_stratum"] == "recent_oct2026"]),
    }
    count_bins = {
        name: describe_stratum([row["diff"] for row in eligible if row["timed_bin"] == name])
        for name in ("2", "3", "4-7", "ge_8")
    }
    pair_names = sorted({row["role_pair"] for row in eligible})
    role_pairs = {name: describe_stratum([row["diff"] for row in eligible if row["role_pair"] == name]) for name in pair_names}
    reference = g_analysis["primary_last_minus_first"]
    return {
        "analysis_id": "I-D",
        "tests_g_analysis": "D",
        "seed": I_SEED,
        "n_bootstrap": I_N_BOOT,
        "eligible_n": len(eligible),
        "n_count_gap_retained": sum(1 for row in eligible if row["count_gap"]),
        "n_role_pair_with_unresolved": n_unresolved_pair,
        "g_reference": _copy_g_primary(reference),
        "overall": overall,
        "count_gap_and_date": gap_strata,
        "timed_reply_count_bins": count_bins,
        "first_last_role_pairs": role_pairs,
        "unresolved_rule": "Roles that cannot be established stay unresolved; they are never silently assigned.",
        "de_overlap_note": "D and E share almost the same threads and first/last outcome; they are not independent confirmations.",
        "reading_aid": reading_aid(
            reference=reference.get("mean"),
            diagnostic=None,
            strata=gap_strata,
            estimand_note="Saved first-to-last reply difference only; missing replies are not inferred from count gaps.",
        ),
        "count_gap_exclusion": False,
    }


def analysis_i_e(
    threads: dict[str, dict[str, Any]],
    comments_by_id: dict[str, list[dict[str, Any]]],
    g_analysis: dict[str, Any],
) -> dict[str, Any]:
    eligible = []
    n_negative_latency = 0
    coverage_num = 0
    coverage_den = 0
    per_thread_coverage = []
    malformed_or_missing = 0
    for comments in comments_by_id.values():
        n_nonempty = len(comments)
        n_timed = sum(1 for row in comments if row.get("timestamp_valid"))
        coverage_num += n_timed
        coverage_den += n_nonempty
        if n_nonempty:
            per_thread_coverage.append(n_timed / n_nonempty)
            malformed_or_missing += n_nonempty - n_timed
    for rid, thread in threads.items():
        if thread.get("post_to_first_reply_hours") is not None and float(thread["post_to_first_reply_hours"]) < 0:
            n_negative_latency += 1
        if not thread.get("elapsed_eligible") or thread.get("elapsed_hours") is None or thread.get("last_minus_first") is None:
            continue
        hours = float(thread["elapsed_hours"])
        if hours < 1:
            band = "lt_1h"
        elif hours <= 24:
            band = "from_1_to_24h"
        else:
            band = "gt_24h"
        eligible.append(
            {
                "reddit_id": rid,
                "diff": float(thread["last_minus_first"]),
                "hours": hours,
                "count_gap": bool(thread.get("count_gap")),
                "date_stratum": date_stratum(thread.get("created_month")),
                "n_timed": int(thread["n_timed_nonempty"]),
                "timed_bin": timed_reply_count_bin(int(thread["n_timed_nonempty"])),
                "duration_band": band,
            }
        )
    if len(eligible) != EXPECTED_N["E"]:
        raise ValueError(f"I-E n {len(eligible)} != {EXPECTED_N['E']}")
    total_coverage = None if coverage_den == 0 else coverage_num / coverage_den
    coverage_effectively_complete = total_coverage is not None and total_coverage >= 0.999
    overall = bootstrap_spearman(
        [row["hours"] for row in eligible],
        [row["diff"] for row in eligible],
        n_boot=I_N_BOOT,
        seed=I_SEED,
    )

    def spearman_stratum(group: list[dict[str, Any]]) -> dict[str, Any]:
        if len(group) < 30:
            return {"n": len(group), "spearman": None, "spearman_ci95": None, "interval_suppressed": True, "reason": "n<30"}
        out = bootstrap_spearman(
            [row["hours"] for row in group],
            [row["diff"] for row in group],
            n_boot=I_N_BOOT,
            seed=I_SEED,
        )
        out["interval_suppressed"] = False
        return out

    gap_strata = {
        "count_gap": spearman_stratum([row for row in eligible if row["count_gap"]]),
        "no_gap": spearman_stratum([row for row in eligible if not row["count_gap"]]),
        "older": spearman_stratum([row for row in eligible if row["date_stratum"] == "older"]),
        "recent_oct2026": spearman_stratum([row for row in eligible if row["date_stratum"] == "recent_oct2026"]),
    }
    count_bins = {
        name: spearman_stratum([row for row in eligible if row["timed_bin"] == name])
        for name in ("2", "3", "4-7", "ge_8")
    }
    duration = {
        name: describe_stratum([row["diff"] for row in eligible if row["duration_band"] == name])
        for name in ("lt_1h", "from_1_to_24h", "gt_24h")
    }
    n_d = sum(1 for row in threads.values() if row.get("order_eligible") and row.get("last_minus_first") is not None)
    reference = g_analysis["spearman_elapsed_hours_vs_last_minus_first"]
    e_mean = describe_stratum([row["diff"] for row in eligible])
    return {
        "analysis_id": "I-E",
        "tests_g_analysis": "E",
        "seed": I_SEED,
        "n_bootstrap": I_N_BOOT,
        "eligible_n": len(eligible),
        "n_count_gap_retained": sum(1 for row in eligible if row["count_gap"]),
        "g_reference": _copy_g_primary(reference),
        "overall_spearman": overall,
        "last_minus_first_on_e_eligible": e_mean,
        "count_gap_and_date": gap_strata,
        "timed_reply_count_bins": count_bins,
        "duration_bands": duration,
        "timestamp_coverage": {
            "valid_timestamped_nonempty_comments": coverage_num,
            "all_nonempty_saved_comments": coverage_den,
            "total_coverage": total_coverage,
            "mean_per_thread_coverage": None if not per_thread_coverage else sum(per_thread_coverage) / len(per_thread_coverage),
            "n_malformed_or_missing_timestamps": malformed_or_missing,
            "effectively_complete": coverage_effectively_complete,
            "subgroup_invented": False,
            "note": (
                "No meaningful timestamp-coverage subgroup exists; coverage is effectively complete."
                if coverage_effectively_complete
                else "Timestamp coverage is incomplete; report coverage rather than inventing a completeness subgroup."
            ),
        },
        "n_negative_post_to_first_latency_flagged_not_dropped": n_negative_latency,
        "d_eligible_n": n_d,
        "n_d_minus_n_e": n_d - len(eligible),
        "de_overlap_note": (
            f"D n={n_d} and E n={len(eligible)} differ by {n_d - len(eligible)} threads. "
            "E's first/last mean largely repeats D and is not an independent confirmation."
        ),
        "reading_aid": reading_aid(
            reference=reference.get("spearman"),
            diagnostic=overall.get("spearman"),
            strata={
                name: {"n": block.get("n"), "mean": block.get("spearman")}
                for name, block in gap_strata.items()
            },
            estimand_note="E is elapsed-hours vs first/last difference; distinct from D's order estimand but almost the same threads.",
        ),
        "count_gap_exclusion": False,
    }
