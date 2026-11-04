"""Workstream I length robustness: I-A, I-B, I-F."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .inference import percentile_ci, stratified_mean_diff
from .robustness_i_common import (
    EXPECTED_N,
    I_N_BOOT,
    I_SEED,
    assign_cut_bin,
    author_reply_count_bin,
    date_stratum,
    describe_stratum,
    first_author_comment,
    pooled_character_bins,
    ratio_bin_ia,
    ratio_bin_ib,
    reading_aid,
    stratified_ols_s2_ci,
)


def bin_support_ok(n_troubleshooting: int, n_other: int) -> bool:
    return n_troubleshooting >= 30 and n_other >= 10


def _copy_g_primary(block: dict[str, Any]) -> dict[str, Any]:
    keys = ("n", "mean", "median", "mean_ci95", "median_ci95", "std", "difference_name", "mean_diff", "mean_diff_ci95")
    return {key: block.get(key) for key in keys if key in block}


def analysis_i_a(
    posts: dict[str, dict[str, Any]],
    threads: dict[str, dict[str, Any]],
    measures: dict[str, dict[str, Any]],
    g_analysis: dict[str, Any],
) -> dict[str, Any]:
    paired = []
    for rid, post in posts.items():
        if post.get("title_compound") is None or post.get("body_compound") is None:
            continue
        measure = measures[rid]
        thread = threads[rid]
        if measure["title_words"] <= 0:
            raise ValueError(f"empty title word count for nonempty title {rid}")
        ratio = measure["body_words"] / measure["title_words"]
        k = measure.get("k_title_body")
        paired.append(
            {
                "reddit_id": rid,
                "body_minus_title": float(post["body_minus_title"]),
                "title_words": measure["title_words"],
                "body_words": measure["body_words"],
                "ratio": ratio,
                "ratio_bin": ratio_bin_ia(ratio),
                "count_gap": bool(thread.get("count_gap")),
                "date_stratum": date_stratum(thread.get("created_month")),
                "k": k,
                "title_prefix": measure.get("title_prefix_compound"),
                "body_prefix": measure.get("body_prefix_compound"),
            }
        )
    if len(paired) != EXPECTED_N["A"]:
        raise ValueError(f"I-A n {len(paired)} != {EXPECTED_N['A']}")
    n_count_gap = sum(1 for row in paired if row["count_gap"])
    prefix_eligible = [
        row
        for row in paired
        if row["k"] is not None and row["k"] >= 3 and row["title_prefix"] is not None and row["body_prefix"] is not None
    ]
    n_fail_k = len(paired) - len(prefix_eligible)
    ratio_strata = {}
    for name in ("le_1", "(1,2]", "(2,5]", "gt_5"):
        group = [row for row in paired if row["ratio_bin"] == name]
        values = [row["body_minus_title"] for row in group]
        block = describe_stratum(values)
        block["median_title_words"] = None if not group else float(np.median([row["title_words"] for row in group]))
        block["median_body_words"] = None if not group else float(np.median([row["body_words"] for row in group]))
        block["sparse_n_lt_100"] = len(group) < 100
        ratio_strata[name] = block
    gap_strata = {
        "count_gap": describe_stratum([row["body_minus_title"] for row in paired if row["count_gap"]]),
        "no_gap": describe_stratum([row["body_minus_title"] for row in paired if not row["count_gap"]]),
        "older": describe_stratum([row["body_minus_title"] for row in paired if row["date_stratum"] == "older"]),
        "recent_oct2026": describe_stratum([row["body_minus_title"] for row in paired if row["date_stratum"] == "recent_oct2026"]),
    }
    full_on_prefix = describe_stratum([row["body_minus_title"] for row in prefix_eligible])
    full_on_prefix["difference_name"] = "full_body_minus_title_on_prefix_eligible_subset"
    prefix_diff = describe_stratum([row["body_prefix"] - row["title_prefix"] for row in prefix_eligible])
    prefix_diff["difference_name"] = "body_prefix_minus_title_prefix"
    prefix_diff["estimand"] = "truncated_text_first_k_words; not a correction of full-body sentiment"
    reference = g_analysis["primary_body_minus_title"]
    return {
        "analysis_id": "I-A",
        "tests_g_analysis": "A",
        "seed": I_SEED,
        "n_bootstrap": I_N_BOOT,
        "eligible_n": len(paired),
        "n_count_gap_retained": n_count_gap,
        "g_reference": _copy_g_primary(reference),
        "word_ratio_bins": ratio_strata,
        "count_gap_and_date": gap_strata,
        "prefix_diagnostic": {
            "k_rule": "k=min(title_words, body_words); require k>=3",
            "n_k_ge_3": len(prefix_eligible),
            "n_fail_k_lt_3": n_fail_k,
            "full_text_on_same_subset": full_on_prefix,
            "prefix": prefix_diff,
        },
        "reading_aid": reading_aid(
            reference=reference.get("mean"),
            diagnostic=prefix_diff.get("mean"),
            strata=gap_strata,
            estimand_note="Prefix is a different truncated-text estimand; persistence is not purely semantic proof.",
        ),
        "count_gap_exclusion": False,
    }


def analysis_i_b(
    threads: dict[str, dict[str, Any]],
    comments_by_id: dict[str, list[dict[str, Any]]],
    measures: dict[str, dict[str, Any]],
    g_analysis: dict[str, Any],
) -> dict[str, Any]:
    eligible = []
    n_zero_post_words = 0
    for rid, thread in threads.items():
        if not thread.get("author_reply_eligible") or thread.get("author_mean") is None or thread.get("combined_compound") is None:
            continue
        measure = measures[rid]
        first = first_author_comment(comments_by_id.get(rid) or [])
        if first is None:
            raise ValueError(f"B-eligible thread missing author comment scores: {rid}")
        post_words = measure["original_post_words"]
        mean_reply_words = measure["mean_author_reply_words"]
        if post_words <= 0 or mean_reply_words is None:
            n_zero_post_words += 1
            ratio = None
            ratio_bin = "undefined_zero_denominator"
        else:
            ratio = float(mean_reply_words) / float(post_words)
            ratio_bin = ratio_bin_ib(ratio)
        eligible.append(
            {
                "reddit_id": rid,
                "author_mean_minus_combined": float(thread["author_mean"] - thread["combined_compound"]),
                "combined_compound": float(thread["combined_compound"]),
                "first_reply_compound": float(first["compound"]),
                "n_author": int(thread["n_author_replies_nonempty"]),
                "reply_count_bin": author_reply_count_bin(int(thread["n_author_replies_nonempty"])),
                "ratio": ratio,
                "ratio_bin": ratio_bin,
                "count_gap": bool(thread.get("count_gap")),
                "date_stratum": date_stratum(thread.get("created_month")),
                "k": measure.get("k_post_first_reply"),
                "post_prefix": measure.get("post_prefix_compound"),
                "reply_prefix": measure.get("first_reply_prefix_compound"),
            }
        )
    if len(eligible) != EXPECTED_N["B"]:
        raise ValueError(f"I-B n {len(eligible)} != {EXPECTED_N['B']}")
    prefix_eligible = [
        row
        for row in eligible
        if row["k"] is not None and row["k"] >= 3 and row["post_prefix"] is not None and row["reply_prefix"] is not None
    ]
    ratio_order = ("le_0.25", "(0.25,0.5]", "(0.5,1]", "gt_1")
    ratio_strata = {
        name: describe_stratum([row["author_mean_minus_combined"] for row in eligible if row["ratio_bin"] == name])
        for name in ratio_order
    }
    if any(row["ratio_bin"] == "undefined_zero_denominator" for row in eligible):
        ratio_strata["undefined_zero_denominator"] = describe_stratum(
            [row["author_mean_minus_combined"] for row in eligible if row["ratio_bin"] == "undefined_zero_denominator"]
        )
    count_strata = {
        name: describe_stratum([row["author_mean_minus_combined"] for row in eligible if row["reply_count_bin"] == name])
        for name in ("1", "2-3", "ge_4")
    }
    gap_strata = {
        "count_gap": describe_stratum([row["author_mean_minus_combined"] for row in eligible if row["count_gap"]]),
        "no_gap": describe_stratum([row["author_mean_minus_combined"] for row in eligible if not row["count_gap"]]),
        "older": describe_stratum([row["author_mean_minus_combined"] for row in eligible if row["date_stratum"] == "older"]),
        "recent_oct2026": describe_stratum(
            [row["author_mean_minus_combined"] for row in eligible if row["date_stratum"] == "recent_oct2026"]
        ),
    }
    first_reply_full = describe_stratum(
        [row["first_reply_compound"] - row["combined_compound"] for row in prefix_eligible]
    )
    first_reply_full["difference_name"] = "first_author_reply_minus_combined_post_on_prefix_eligible_subset"
    first_reply_full["estimand"] = "first chronological author reply vs original combined post; not mean-of-all-replies"
    prefix_diff = describe_stratum([row["reply_prefix"] - row["post_prefix"] for row in prefix_eligible])
    prefix_diff["difference_name"] = "first_reply_prefix_minus_post_prefix"
    prefix_diff["estimand"] = "truncated_text_first_k_words of original-post vs first author reply"
    reference = g_analysis["primary_author_mean_minus_combined"]
    return {
        "analysis_id": "I-B",
        "tests_g_analysis": "B",
        "seed": I_SEED,
        "n_bootstrap": I_N_BOOT,
        "eligible_n": len(eligible),
        "n_count_gap_retained": sum(1 for row in eligible if row["count_gap"]),
        "n_undefined_word_ratio": n_zero_post_words,
        "g_reference": _copy_g_primary(reference),
        "reply_over_original_word_ratio_bins": ratio_strata,
        "author_reply_count_bins": count_strata,
        "count_gap_and_date": gap_strata,
        "prefix_diagnostic": {
            "k_rule": "k=min(original_post_words, first_author_reply_words); require k>=3",
            "n_k_ge_3": len(prefix_eligible),
            "n_fail_k_lt_3": len(eligible) - len(prefix_eligible),
            "full_first_reply_on_same_subset": first_reply_full,
            "prefix": prefix_diff,
            "not_equivalent_to": "mean of all observed author replies",
        },
        "reading_aid": reading_aid(
            reference=reference.get("mean"),
            diagnostic=prefix_diff.get("mean"),
            strata=gap_strata,
            estimand_note="Prefix compares truncated original-post and first reply only.",
        ),
        "count_gap_exclusion": False,
    }


def analysis_i_f(
    threads: dict[str, dict[str, Any]],
    measures: dict[str, dict[str, Any]],
    g_analysis: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    for rid, thread in threads.items():
        if thread.get("f_eligible") is not True:
            continue
        measure = measures[rid]
        chars = int(measure["original_post_chars"])
        rows.append(
            {
                "reddit_id": rid,
                "s2_target": int(thread["s2_target"]),
                "combined_compound": float(thread["combined_compound"]),
                "title_compound": None if thread.get("title_compound") is None else float(thread["title_compound"]),
                "original_post_chars": chars,
                "log1p_chars": float(math.log1p(chars)),
                "body_present": 1 if thread.get("body_nonempty") else 0,
                "october": 1 if thread.get("created_month") == "2026-10" else 0,
                "count_gap": 1 if thread.get("count_gap") else 0,
                "source_is_reddit": 1 if thread.get("source_is_reddit") else 0,
            }
        )
    if len(rows) != EXPECTED_N["F"]:
        raise ValueError(f"I-F n {len(rows)} != {EXPECTED_N['F']}")
    tps = [row for row in rows if row["s2_target"] == 1]
    other = [row for row in rows if row["s2_target"] == 0]
    if len(tps) != 1628 or len(other) != 236:
        raise ValueError("I-F group split drifted")
    cuts = pooled_character_bins([row["original_post_chars"] for row in rows], n_bins=5)
    for row in rows:
        row["bin"] = assign_cut_bin(row["original_post_chars"], cuts)
    n_bins = len(cuts) - 1
    bin_rows = []
    support_ok = n_bins >= 1
    for bin_id in range(n_bins):
        group = [row for row in rows if row["bin"] == bin_id]
        tps_b = [row for row in group if row["s2_target"] == 1]
        other_b = [row for row in group if row["s2_target"] == 0]
        ok = bin_support_ok(len(tps_b), len(other_b))
        if not ok:
            support_ok = False
        mean_tps = None if not tps_b else float(sum(r["combined_compound"] for r in tps_b) / len(tps_b))
        mean_other = None if not other_b else float(sum(r["combined_compound"] for r in other_b) / len(other_b))
        mean_diff = None if mean_tps is None or mean_other is None else mean_tps - mean_other
        bin_rows.append(
            {
                "bin_id": bin_id,
                "char_lower": cuts[bin_id],
                "char_upper": cuts[bin_id + 1],
                "n": len(group),
                "n_troubleshooting": len(tps_b),
                "n_other": len(other_b),
                "weight": len(group) / len(rows),
                "mean_troubleshooting": mean_tps,
                "mean_other": mean_other,
                "mean_diff": mean_diff,
                "support_ok": ok,
                "min_support_rule": ">=30 troubleshooting and >=10 other",
            }
        )
    unadjusted = stratified_mean_diff(
        [row["combined_compound"] for row in tps],
        [row["combined_compound"] for row in other],
        n_boot=I_N_BOOT,
        seed=I_SEED,
        difference_name="troubleshooting_minus_other_in_scope",
    )
    tps_title = [row["title_compound"] for row in tps if row["title_compound"] is not None]
    other_title = [row["title_compound"] for row in other if row["title_compound"] is not None]
    title_only = stratified_mean_diff(
        tps_title,
        other_title,
        n_boot=I_N_BOOT,
        seed=I_SEED,
        difference_name="title_only_troubleshooting_minus_other",
    )
    title_only["n_missing_title_troubleshooting"] = len(tps) - len(tps_title)
    title_only["n_missing_title_other"] = len(other) - len(other_title)

    def _group_diff(label: str, predicate) -> dict[str, Any]:
        t_vals = [row["combined_compound"] for row in tps if predicate(row)]
        o_vals = [row["combined_compound"] for row in other if predicate(row)]
        block = stratified_mean_diff(
            t_vals,
            o_vals,
            n_boot=I_N_BOOT,
            seed=I_SEED,
            difference_name="troubleshooting_minus_other_in_scope",
        )
        block["stratum"] = label
        return block

    gap_strata = {
        "count_gap": _group_diff("count_gap", lambda row: row["count_gap"] == 1),
        "no_gap": _group_diff("no_gap", lambda row: row["count_gap"] == 0),
        "older": _group_diff("older", lambda row: row["october"] == 0),
        "recent_oct2026": _group_diff("recent_oct2026", lambda row: row["october"] == 1),
    }
    adjusted = None
    if support_ok:
        weights = [block["weight"] for block in bin_rows]
        point = float(sum((block["mean_diff"] or 0.0) * block["weight"] for block in bin_rows))
        rng = np.random.default_rng(I_SEED)
        samples: list[float] = []
        for _ in range(I_N_BOOT):
            t_boot = [tps[i] for i in rng.integers(0, len(tps), size=len(tps))]
            o_boot = [other[i] for i in rng.integers(0, len(other), size=len(other))]
            ok_rep = True
            diffs = []
            for bin_id in range(n_bins):
                t_vals = [row["combined_compound"] for row in t_boot if row["bin"] == bin_id]
                o_vals = [row["combined_compound"] for row in o_boot if row["bin"] == bin_id]
                if not t_vals or not o_vals:
                    ok_rep = False
                    break
                diffs.append(sum(t_vals) / len(t_vals) - sum(o_vals) / len(o_vals))
            if ok_rep:
                samples.append(float(sum(w * d for w, d in zip(weights, diffs))))
        adjusted = {
            "mean_diff": point,
            "mean_diff_ci95": percentile_ci(samples) if samples else None,
            "weights": weights,
            "n_bootstrap": I_N_BOOT,
            "seed": I_SEED,
            "bootstrap_unit": "thread_stratified",
            "n_defined_resamples": len(samples),
            "note": "Bin membership and weights are fixed from the 1,864 reviewed posts.",
        }
    ols = stratified_ols_s2_ci(rows)
    g_ref = g_analysis["primary_combined_post"]["comparison"]
    payload = {
        "analysis_id": "I-F",
        "tests_g_analysis": "F",
        "seed": I_SEED,
        "n_bootstrap": I_N_BOOT,
        "eligible_n": len(rows),
        "n_troubleshooting": len(tps),
        "n_other_in_scope": len(other),
        "n_count_gap_retained": sum(row["count_gap"] for row in rows),
        "g_reference": {
            "n_a": g_ref.get("n_a"),
            "n_b": g_ref.get("n_b"),
            "mean_diff": g_ref.get("mean_diff"),
            "mean_diff_ci95": g_ref.get("mean_diff_ci95"),
            "cliffs_delta": g_ref.get("cliffs_delta"),
        },
        "unadjusted_combined_post": unadjusted,
        "length_bins": {
            "cuts": cuts,
            "n_bins": n_bins,
            "support_ok": support_ok,
            "pooled_adjusted_estimate_reported": bool(adjusted),
            "support_failure": not support_ok,
            "rows": bin_rows,
        },
        "pooled_length_adjusted": adjusted,
        "ols_s2_target": ols,
        "title_only_crosscheck": title_only,
        "count_gap_and_date": gap_strata,
        "body_length_imbalance": {
            "mean_chars_troubleshooting": float(sum(r["original_post_chars"] for r in tps) / len(tps)),
            "mean_chars_other": float(sum(r["original_post_chars"] for r in other) / len(other)),
        },
        "missing_covariates": 0,
        "reading_aid": reading_aid(
            reference=g_ref.get("mean_diff"),
            diagnostic=None if adjusted is None else adjusted.get("mean_diff"),
            strata={
                name: {
                    "n": int((block.get("n_a") or 0) + (block.get("n_b") or 0)),
                    "mean": block.get("mean_diff"),
                }
                for name, block in gap_strata.items()
            },
            estimand_note="Length-adjusted F is an association, not a causal group effect or prevalence estimate.",
        ),
        "count_gap_exclusion": False,
        "selection_note": "F remains the selected 1,864 reviewed in-scope posts, not the 37,596 candidate frame.",
    }
    return payload, bin_rows
