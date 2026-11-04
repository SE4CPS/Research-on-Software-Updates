"""Predeclared A–F thread-level summaries. Exploratory; no prevalence or causal claims."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Iterable

from .inference import (
    bootstrap_mean_median_ci,
    bootstrap_spearman,
    paired_difference_summary,
    stratified_mean_diff,
    summarize_numeric,
)
from .paths import BOOTSTRAP_SEED, N_BOOTSTRAP
from .scoring import date_band

COMMON_NOTES = [
    "Exploratory descriptive comparisons; not a significance hunt.",
    "Thread is the resampling unit; each eligible thread has equal weight.",
    "Missing replies are null, not VADER 0.0.",
    "Count-gap is a sensitivity stratum, not an exclusion and not proof of completeness.",
    "The broad frame is a dated ReleaseTrain candidate snapshot, not verified update prevalence.",
]


def _ids_hash(ids: Iterable[str]) -> str:
    payload = "\n".join(sorted(str(i) for i in ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _values(rows: list[dict[str, Any]], key: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is not None:
            out.append(float(value))
    return out


def _filter(rows: list[dict[str, Any]], **flags: Any) -> list[dict[str, Any]]:
    selected = rows
    for key, expected in flags.items():
        selected = [row for row in selected if row.get(key) == expected]
    return selected


def _stratum_block(rows: list[dict[str, Any]], key: str, value_key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key))].append(row)
    return {
        str(name): bootstrap_mean_median_ci(_values(group, value_key))
        for name, group in sorted(grouped.items())
    }


def _header(analysis_id: str, question: str, rows: list[dict[str, Any]], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "analysis_id": analysis_id,
        "question": question,
        "capture_id": "2026-10-22-main",
        "package_version": "0.2.1",
        "seed": BOOTSTRAP_SEED,
        "n_bootstrap": N_BOOTSTRAP,
        "bootstrap_unit": "thread",
        "eligible_n": len(rows),
        "eligible_ids_sha256": _ids_hash(row["reddit_id"] for row in rows),
        "notes": list(COMMON_NOTES),
    }
    if extra:
        payload.update(extra)
    return payload


def analysis_a(threads: list[dict[str, Any]]) -> dict[str, Any]:
    titles = [row for row in threads if row.get("title_compound") is not None]
    paired = [row for row in titles if row.get("body_compound") is not None]
    combined = [row for row in threads if row.get("combined_compound") is not None]
    result = _header(
        "A",
        "Do title and body express different sentiment?",
        titles,
        {
            "paired_n": len(paired),
            "exclusions": {
                "empty_body_unpaired": len(titles) - len(paired),
                "reason": "Empty body is null, not zero, and is outside the paired comparison.",
            },
        },
    )
    result["title"] = bootstrap_mean_median_ci(_values(titles, "title_compound"))
    result["body"] = bootstrap_mean_median_ci(_values(paired, "body_compound"))
    result["combined"] = bootstrap_mean_median_ci(_values(combined, "combined_compound"))
    result["primary_body_minus_title"] = paired_difference_summary(
        _values(paired, "title_compound"),
        _values(paired, "body_compound"),
        difference_name="body_minus_title",
    )
    result["sensitivities"] = {
        "count_gap": _stratum_block(paired, "count_gap", "body_minus_title"),
        "date_band": _stratum_block(paired, "date_band", "body_minus_title"),
        "source_is_reddit": _stratum_block(paired, "source_is_reddit", "body_minus_title"),
        "title_len_band": _stratum_block(paired, "title_len_band", "body_minus_title"),
        "body_len_band": _stratum_block(paired, "body_len_band", "body_minus_title"),
    }
    return result


def analysis_b(threads: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in threads if row.get("author_reply_eligible") and row.get("combined_compound") is not None and row.get("author_mean") is not None]
    result = _header(
        "B",
        "Does the original author's expressed sentiment differ from their observed later replies?",
        eligible,
        {
            "exclusions": {
                "no_author_reply": sum(1 for row in threads if not row.get("author_reply_eligible")),
                "reason": "No observed author reply is null and outside this paired comparison, never neutral.",
            },
            "interpretation_limit": "Descriptive change only; not evidence that discussion caused sentiment to change.",
        },
    )
    result["author_reply_count"] = summarize_numeric(
        [float(row["n_author_replies_nonempty"]) for row in eligible]
    )
    result["primary_author_mean_minus_combined"] = paired_difference_summary(
        _values(eligible, "combined_compound"),
        _values(eligible, "author_mean"),
        difference_name="author_reply_mean_minus_combined_post",
    )
    result["sensitivities"] = {
        "author_replies_ge2": paired_difference_summary(
            _values([row for row in eligible if row["n_author_replies_nonempty"] >= 2], "combined_compound"),
            _values([row for row in eligible if row["n_author_replies_nonempty"] >= 2], "author_mean"),
            difference_name="author_reply_mean_minus_combined_post",
        ),
        "author_replies_ge4": paired_difference_summary(
            _values([row for row in eligible if row["n_author_replies_nonempty"] >= 4], "combined_compound"),
            _values([row for row in eligible if row["n_author_replies_nonempty"] >= 4], "author_mean"),
            difference_name="author_reply_mean_minus_combined_post",
        ),
        "count_gap": {
            str(flag): paired_difference_summary(
                _values(_filter(eligible, count_gap=flag), "combined_compound"),
                _values(_filter(eligible, count_gap=flag), "author_mean"),
                difference_name="author_reply_mean_minus_combined_post",
            )
            for flag in (True, False)
        },
    }
    return result


def analysis_c(threads: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [
        row
        for row in threads
        if row.get("both_roles_eligible") and row.get("author_mean") is not None and row.get("community_mean") is not None
    ]
    result = _header(
        "C",
        "Within the same saved conversation, how do author and community sentiments differ?",
        eligible,
        {
            "role_note": (
                "Community mean uses nonempty non-author replies, matching the frozen "
                "denominator (blank-name unresolved comments are non-author). "
                "comment_scores still label those rows unresolved."
            ),
            "pooling_note": "Do not pool comments as independent observations; each thread has equal weight.",
        },
    )
    result["author_reply_count"] = summarize_numeric(
        [float(row["n_author_replies_nonempty"]) for row in eligible]
    )
    result["community_reply_count"] = summarize_numeric(
        [float(row["n_community_replies_nonempty"]) for row in eligible]
    )
    result["primary_community_minus_author"] = paired_difference_summary(
        _values(eligible, "author_mean"),
        _values(eligible, "community_mean"),
        difference_name="community_mean_minus_author_mean",
    )
    ge2 = [
        row
        for row in eligible
        if row["n_author_replies_nonempty"] >= 2 and row["n_community_replies_nonempty"] >= 2
    ]
    result["sensitivities"] = {
        "at_least_two_replies_per_role": paired_difference_summary(
            _values(ge2, "author_mean"),
            _values(ge2, "community_mean"),
            difference_name="community_mean_minus_author_mean",
        ),
        "count_gap": {
            str(flag): paired_difference_summary(
                _values(_filter(eligible, count_gap=flag), "author_mean"),
                _values(_filter(eligible, count_gap=flag), "community_mean"),
                difference_name="community_mean_minus_author_mean",
            )
            for flag in (True, False)
        },
    }
    return result


def _position_bins(threads: list[dict[str, Any]]) -> dict[str, Any]:
    abs_bins: dict[str, list[float]] = defaultdict(list)
    quartile_bins: dict[str, list[float]] = defaultdict(list)
    for row in threads:
        compounds = row.get("timed_compounds") or []
        n = len(compounds)
        if n < 2:
            continue
        by_abs: dict[str, list[float]] = defaultdict(list)
        by_q: dict[str, list[float]] = defaultdict(list)
        for i, compound in enumerate(compounds):
            if i == 0:
                abs_name = "position_1"
            elif i == 1:
                abs_name = "position_2"
            elif i == 2:
                abs_name = "position_3"
            else:
                abs_name = "position_4plus"
            frac = i / (n - 1)
            if frac < 0.25:
                q_name = "q1_start"
            elif frac < 0.5:
                q_name = "q2"
            elif frac < 0.75:
                q_name = "q3"
            else:
                q_name = "q4_end"
            by_abs[abs_name].append(float(compound))
            by_q[q_name].append(float(compound))
        for name, values in by_abs.items():
            abs_bins[name].append(sum(values) / len(values))
        for name, values in by_q.items():
            quartile_bins[name].append(sum(values) / len(values))
    return {
        "absolute_position_thread_equal_weight": {
            name: bootstrap_mean_median_ci(values) for name, values in sorted(abs_bins.items())
        },
        "normalized_quartile_thread_equal_weight": {
            name: bootstrap_mean_median_ci(values) for name, values in sorted(quartile_bins.items())
        },
        "binning_note": "Within-thread means in each bin are averaged across threads so long threads are not overweighted.",
    }


def analysis_d(threads: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in threads if row.get("order_eligible") and row.get("last_minus_first") is not None]
    ge3 = [row for row in eligible if row.get("n_timed_nonempty", 0) >= 3 and row.get("order_slope_normalized") is not None]
    ge4 = [row for row in eligible if row.get("n_timed_nonempty", 0) >= 4]
    result = _header(
        "D",
        "How does observed reply sentiment vary with reply order?",
        eligible,
        {
            "interpretation_limit": "Sequence is not causal change; absent comments are not imputed.",
            "n_slope_ge3": len(ge3),
            "n_ge4_saved_timed_replies": len(ge4),
        },
    )
    result["primary_last_minus_first"] = bootstrap_mean_median_ci(_values(eligible, "last_minus_first"))
    result["secondary_normalized_slope_ge3"] = bootstrap_mean_median_ci(_values(ge3, "order_slope_normalized"))
    result["role_composition"] = {
        "n_author_replies": summarize_numeric([float(row["n_author_replies_nonempty"]) for row in eligible]),
        "n_community_replies": summarize_numeric([float(row["n_community_replies_nonempty"]) for row in eligible]),
        "n_unresolved_replies": summarize_numeric([float(row["n_unresolved_replies_nonempty"]) for row in eligible]),
        "n_timed": summarize_numeric([float(row["n_timed_nonempty"]) for row in eligible]),
    }
    result["package_role_shifts"] = {
        "author_shift_n_ge4": bootstrap_mean_median_ci(
            _values([row for row in eligible if row.get("author_shift") is not None], "author_shift")
        ),
        "community_shift_n_ge4": bootstrap_mean_median_ci(
            _values([row for row in eligible if row.get("community_shift") is not None], "community_shift")
        ),
        "rule": "Package early/late helper is null unless >=4 nonempty comments of that role.",
    }
    result["position_bins"] = _position_bins(eligible)
    result["sensitivities"] = {
        "count_gap": _stratum_block(eligible, "count_gap", "last_minus_first"),
        "timed_replies_ge4": bootstrap_mean_median_ci(_values(ge4, "last_minus_first")),
        "slope_ge4": bootstrap_mean_median_ci(_values(ge4, "order_slope_normalized")),
    }
    return result


def analysis_e(threads: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [
        row
        for row in threads
        if row.get("elapsed_eligible") and row.get("elapsed_hours") is not None and row.get("last_minus_first") is not None
    ]
    ge4 = [row for row in eligible if row.get("n_timed_nonempty", 0) >= 4]
    latency = [
        row
        for row in threads
        if row.get("post_to_first_reply_eligible") and row.get("post_to_first_reply_hours") is not None
    ]
    result = _header(
        "E",
        "Is observed sentiment change associated with elapsed time between replies?",
        eligible,
        {
            "interpretation_limit": "Elapsed time is distinct from reply order. No causal claim about time or resolution.",
            "n_equal_time_order_but_not_elapsed": sum(
                1 for row in threads if row.get("order_eligible") and not row.get("elapsed_eligible")
            ),
            "n_post_to_first_reply": len(latency),
            "n_negative_post_to_first": sum(
                1 for row in latency if float(row["post_to_first_reply_hours"]) < 0
            ),
            "n_elapsed_gt_30d": sum(1 for row in eligible if float(row["elapsed_hours"]) > 24 * 30),
        },
    )
    result["elapsed_hours"] = bootstrap_mean_median_ci(_values(eligible, "elapsed_hours"))
    result["primary_last_minus_first"] = bootstrap_mean_median_ci(_values(eligible, "last_minus_first"))
    result["duration_bands"] = {
        band: {
            "last_minus_first": bootstrap_mean_median_ci(_values(group, "last_minus_first")),
            "elapsed_hours": bootstrap_mean_median_ci(_values(group, "elapsed_hours")),
        }
        for band, group in (
            ("lt_1h", _filter(eligible, elapsed_band="lt_1h")),
            ("from_1_to_24h", _filter(eligible, elapsed_band="from_1_to_24h")),
            ("gt_24h", _filter(eligible, elapsed_band="gt_24h")),
        )
    }
    result["spearman_elapsed_hours_vs_last_minus_first"] = bootstrap_spearman(
        _values(eligible, "elapsed_hours"),
        _values(eligible, "last_minus_first"),
    )
    result["secondary_post_to_first_reply_hours"] = bootstrap_mean_median_ci(
        _values(latency, "post_to_first_reply_hours")
    )
    result["sensitivities"] = {
        "count_gap": _stratum_block(eligible, "count_gap", "last_minus_first"),
        "timed_replies_ge4": bootstrap_mean_median_ci(_values(ge4, "last_minus_first")),
        "date_band": _stratum_block(eligible, "date_band", "last_minus_first"),
    }
    return result


def _f_groups(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tps = [row for row in rows if row.get("s2_target") == 1]
    other = [row for row in rows if row.get("s2_target") == 0]
    return tps, other


def _f_compare(rows: list[dict[str, Any]], value_key: str) -> dict[str, Any]:
    usable = [row for row in rows if row.get(value_key) is not None]
    tps, other = _f_groups(usable)
    return {
        "n_total": len(usable),
        "n_troubleshooting": len(tps),
        "n_other_in_scope": len(other),
        "comparison": stratified_mean_diff(
            _values(tps, value_key),
            _values(other, value_key),
            difference_name="troubleshooting_minus_other_in_scope",
        ),
    }


def analysis_f(threads: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in threads if row.get("f_eligible") is True]
    reserved = [row for row in eligible if row.get("reserved_membership") or row.get("fresh_membership")]
    if reserved:
        raise ValueError("protected membership entered analysis F")
    tps, other = _f_groups(eligible)
    result = _header(
        "F",
        "Within the existing human-reviewed development overlap, is sentiment associated with discussion purpose?",
        eligible,
        {
            "n_troubleshooting": len(tps),
            "n_other_in_scope": len(other),
            "bootstrap_unit": "thread_stratified",
            "interpretation_limit": (
                "Exploratory association only; not a classifier test and not a prevalence estimate. "
                "The reviewed overlap is nonrandom and time-skewed."
            ),
            "protected_note": "Reserved/fresh outcomes were not read; F contains development in-scope IDs only.",
        },
    )
    result["primary_combined_post"] = _f_compare(eligible, "combined_compound")
    result["secondary"] = {
        "title": _f_compare(eligible, "title_compound"),
        "body": _f_compare(eligible, "body_compound"),
        "observed_comment_mean": _f_compare(
            [row for row in eligible if row.get("n_nonempty_comments", 0) >= 1],
            "observed_comment_mean",
        ),
        "author_reply_mean": _f_compare(
            [row for row in eligible if row.get("author_reply_eligible")],
            "author_mean",
        ),
        "paired_roles_community_minus_author": _f_compare(
            [row for row in eligible if row.get("both_roles_eligible")],
            "community_minus_author",
        ),
        "cadence_last_minus_first": _f_compare(
            [row for row in eligible if row.get("order_eligible")],
            "last_minus_first",
        ),
    }
    result["sensitivities"] = {
        "count_gap": {
            str(flag): _f_compare(_filter(eligible, count_gap=flag), "combined_compound")
            for flag in (True, False)
        },
        "date_band": {
            band: _f_compare(_filter(eligible, date_band=band), "combined_compound")
            for band in ("oct2026", "older", "missing")
        },
    }
    return result


def run_all_analyses(threads: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    for row in threads:
        row["date_band"] = date_band(row.get("created_month"))
        if row.get("author_mean") is not None and row.get("community_mean") is not None:
            row["community_minus_author"] = float(row["community_mean"] - row["author_mean"])
        else:
            row["community_minus_author"] = None
        n_nonempty = row.get("n_nonempty_comments") or 0
        if n_nonempty and row.get("author_mean") is None and row.get("community_mean") is None:
            row["observed_comment_mean"] = None
        elif n_nonempty:
            parts = []
            if row.get("author_mean") is not None:
                parts.extend([row["author_mean"]] * int(row["n_author_replies_nonempty"]))
            if row.get("community_mean") is not None:
                parts.extend([row["community_mean"]] * int(row["n_community_replies_nonempty"]))
            row["observed_comment_mean"] = (sum(parts) / len(parts)) if parts else None
        else:
            row["observed_comment_mean"] = None
    return {
        "A": analysis_a(threads),
        "B": analysis_b(threads),
        "C": analysis_c(threads),
        "D": analysis_d(threads),
        "E": analysis_e(threads),
        "F": analysis_f(threads),
    }
