"""Compare original C1-prime, package-native, and harmonized measures on 542 IDs."""

from __future__ import annotations

from collections import Counter
from typing import Any

from releasetrain_sentiment import score_post, trajectory_summary
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .adapter import package_eligibility, to_package_post
from .harmonized import discrepancy_reasons, harmonized_measures
from .io_utils import nonempty_reddit_id, write_json, write_jsonl
from .original_semantics import analyze_post_trajectories_original, compute_metrics_row_original
from .paths import RESULTS_DIR

FLOAT_FIELDS = (
    "title_compound",
    "author_mean_sentiment",
    "community_mean_sentiment",
    "sentiment_divergence",
    "author_trend",
    "community_trend",
    "author_volatility",
    "community_volatility",
    "author_early_late_shift",
    "community_early_late_shift",
)
INT_FIELDS = (
    "corrected_label",
    "n_author_replies",
    "n_community_comments",
    "n_comments_total",
)
BOOL_FIELDS = ("author_more_negative", "trajectory_eligible")
SAVED_COMPARE_FIELDS = FLOAT_FIELDS + INT_FIELDS + BOOL_FIELDS
TOLERANCE = 0.0005


def _as_bool(value: Any) -> bool:
    if value is True or value is False:
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"unusable boolean {value!r}")


def parse_saved_row(row: dict[str, Any]) -> dict[str, Any]:
    parsed: dict[str, Any] = {
        "reddit_id": nonempty_reddit_id(row),
        "subreddit": row.get("subreddit") or "",
        "url": row.get("url") or "",
        "trajectory_notes": row.get("trajectory_notes") or "",
    }
    for field in FLOAT_FIELDS:
        parsed[field] = float(row[field])
    for field in INT_FIELDS:
        parsed[field] = int(row[field])
    for field in BOOL_FIELDS:
        parsed[field] = _as_bool(row[field])
    return parsed


def compare_original_to_saved(reconstructed: dict[str, Any], saved: dict[str, Any]) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    max_abs = 0.0
    for field in FLOAT_FIELDS:
        diff = abs(float(reconstructed[field]) - float(saved[field]))
        max_abs = max(max_abs, diff)
        if diff > TOLERANCE:
            mismatches.append(
                {
                    "field": field,
                    "reconstructed": reconstructed[field],
                    "saved": saved[field],
                    "abs_diff": diff,
                }
            )
    for field in INT_FIELDS + BOOL_FIELDS:
        if reconstructed[field] != saved[field]:
            mismatches.append(
                {
                    "field": field,
                    "reconstructed": reconstructed[field],
                    "saved": saved[field],
                    "abs_diff": None,
                }
            )
    return {
        "n_mismatches": len(mismatches),
        "mismatches_gt_0.0005": mismatches,
        "max_abs_float_diff": max_abs,
        "within_tolerance": len(mismatches) == 0,
    }


def _original_string_indices(post: dict[str, Any]) -> list[int]:
    raw = post.get("comments") or []
    if not isinstance(raw, list):
        return []
    indexed = [(i, c) for i, c in enumerate(raw) if isinstance(c, dict)]
    indexed.sort(key=lambda pair: str(pair[1].get("created_utc") or ""))
    return [i for i, _ in indexed]


def compare_one(
    reddit_id: str,
    label_row: dict[str, Any],
    post: dict[str, Any],
    saved_row: dict[str, Any],
    analyzer: SentimentIntensityAnalyzer,
) -> dict[str, Any]:
    analysis = analyze_post_trajectories_original(post, analyzer)
    original_row = compute_metrics_row_original(
        reddit_id=reddit_id,
        subreddit=str(label_row.get("subreddit") or post.get("subreddit") or ""),
        corrected_label=int(label_row["corrected_label"]),
        url=str(label_row.get("url") or post.get("url") or ""),
        analysis=analysis,
    )
    saved = parse_saved_row(saved_row)
    original_vs_saved = compare_original_to_saved(original_row, saved)

    package_post = to_package_post(post, include_empty_comment_bodies=True)
    scored = score_post(package_post)
    summary = trajectory_summary(scored)
    elig = package_eligibility(package_post)
    post_item = scored[0] if scored else None
    harmonized = harmonized_measures(analyzer, post)
    string_indices = _original_string_indices(post)
    parsed_indices = package_post["adapter_meta"]["chronological_original_indices"]
    reasons = discrepancy_reasons(
        original_row=original_row,
        package_summary=summary,
        package_elig=elig,
        harmonized=harmonized,
        package_post_meta=package_post["adapter_meta"],
        original_string_indices=string_indices,
        parsed_indices=parsed_indices,
    )
    return {
        "reddit_id": reddit_id,
        "historical_corrected_label": int(label_row["corrected_label"]),
        "denominator": "historical_542_corrected_labels",
        "source_snapshot": "march_raw_tps_then_gds_gds_overwrites",
        "original_reconstructed": {k: original_row[k] for k in SAVED_COMPARE_FIELDS + ("trajectory_notes", "title_fallback", "used_body_as_first_author_point", "community_mean_is_observed", "author_early_late_shift_eligible", "community_early_late_shift_eligible")},
        "saved_c1_prime": {k: saved[k] for k in SAVED_COMPARE_FIELDS},
        "original_vs_saved": original_vs_saved,
        "package_native": {
            "n_scored_items": len(scored),
            "post_compound": None if post_item is None else post_item.get("compound"),
            "post_label": None if post_item is None else post_item.get("label"),
            "trajectory_summary": dict(summary),
            **elig,
            "body_field": package_post["adapter_meta"]["body_field"],
            "comments_status": package_post["adapter_meta"]["comments_status"],
        },
        "harmonized": {
            "body_field": harmonized["body_field"],
            "title_nonempty": harmonized["title_nonempty"],
            "body_nonempty": harmonized["body_nonempty"],
            "title_compound": harmonized["title_compound"],
            "body_compound": harmonized["body_compound"],
            "package_post_concat_compound": harmonized["package_post_concat_compound"],
            "n_author_comments_nonempty": harmonized["n_author_comments_nonempty"],
            "n_community_comments_nonempty": harmonized["n_community_comments_nonempty"],
            "author_comment_mean": harmonized["author_comment_mean"],
            "community_comment_mean": harmonized["community_comment_mean"],
            "author_comment_mean_is_observed": harmonized["author_comment_mean_is_observed"],
            "community_comment_mean_is_observed": harmonized["community_comment_mean_is_observed"],
            "author_shift_eligible": harmonized["author_shift_eligible"],
            "community_shift_eligible": harmonized["community_shift_eligible"],
            "author_shift_nonempty": harmonized["author_shift_nonempty"],
            "community_shift_nonempty": harmonized["community_shift_nonempty"],
            "original_like_author_track_mean": harmonized["original_like_author_track_mean"],
            "original_like_title_fallback": harmonized["original_like_title_fallback"],
            "package_like_author_avg": harmonized["package_like_author_avg"],
            "package_like_community_avg": harmonized["package_like_community_avg"],
            "package_like_author_avg_is_observed": harmonized["package_like_author_avg_is_observed"],
            "package_like_community_avg_is_observed": harmonized["package_like_community_avg_is_observed"],
        },
        "discrepancy_reasons": reasons,
        "comment_order": {
            "original_string_sort_indices": string_indices,
            "parsed_timestamp_indices": parsed_indices,
            "orders_equal": string_indices == parsed_indices,
        },
    }


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    n_tps = sum(1 for r in rows if r["historical_corrected_label"] == 1)
    n_gds = sum(1 for r in rows if r["historical_corrected_label"] == 0)
    original_mismatches = [r for r in rows if not r["original_vs_saved"]["within_tolerance"]]
    title_fallback = sum(1 for r in rows if r["original_reconstructed"].get("title_fallback"))
    used_body = sum(1 for r in rows if r["original_reconstructed"].get("used_body_as_first_author_point"))
    pkg_author_unobs = sum(1 for r in rows if not r["package_native"]["author_avg_is_observed"])
    pkg_comm_unobs = sum(1 for r in rows if not r["package_native"]["community_avg_is_observed"])
    pkg_author_zero = sum(1 for r in rows if r["package_native"]["trajectory_summary"]["author_avg"] == 0.0)
    harm_author_obs = sum(1 for r in rows if r["harmonized"]["author_comment_mean_is_observed"])
    harm_comm_obs = sum(1 for r in rows if r["harmonized"]["community_comment_mean_is_observed"])
    author_shift_elig = sum(1 for r in rows if r["harmonized"]["author_shift_eligible"])
    comm_shift_elig = sum(1 for r in rows if r["harmonized"]["community_shift_eligible"])
    order_diff = sum(1 for r in rows if not r["comment_order"]["orders_equal"])
    reason_counts = Counter(reason for r in rows for reason in r["discrepancy_reasons"])

    orig_author = [r["original_reconstructed"]["author_mean_sentiment"] for r in rows]
    pkg_author = [r["package_native"]["trajectory_summary"]["author_avg"] for r in rows]
    abs_orig_pkg_author = [abs(a - b) for a, b in zip(orig_author, pkg_author)]
    abs_orig_harm_author = [
        abs(r["original_reconstructed"]["author_mean_sentiment"] - r["harmonized"]["original_like_author_track_mean"])
        for r in rows
    ]
    abs_pkg_harm_author = [
        abs(r["package_native"]["trajectory_summary"]["author_avg"] - r["harmonized"]["package_like_author_avg"])
        for r in rows
    ]

    representative = {
        "title_fallback_examples": [r["reddit_id"] for r in rows if r["original_reconstructed"].get("title_fallback")][:8],
        "package_unobserved_author_examples": [r["reddit_id"] for r in rows if not r["package_native"]["author_avg_is_observed"]][:8],
        "author_shift_eligible_examples": [r["reddit_id"] for r in rows if r["harmonized"]["author_shift_eligible"]][:8],
        "largest_original_vs_package_author_mean": sorted(
            (
                {
                    "reddit_id": r["reddit_id"],
                    "original_author_mean": r["original_reconstructed"]["author_mean_sentiment"],
                    "package_author_avg": r["package_native"]["trajectory_summary"]["author_avg"],
                    "abs_diff": abs(
                        r["original_reconstructed"]["author_mean_sentiment"]
                        - r["package_native"]["trajectory_summary"]["author_avg"]
                    ),
                    "title_fallback": r["original_reconstructed"].get("title_fallback"),
                    "used_body": r["original_reconstructed"].get("used_body_as_first_author_point"),
                    "package_author_observed": r["package_native"]["author_avg_is_observed"],
                }
                for r in rows
            ),
            key=lambda item: item["abs_diff"],
            reverse=True,
        )[:10],
        "original_vs_saved_mismatch_ids": [r["reddit_id"] for r in original_mismatches],
    }
    return {
        "n": n,
        "n_tps_corrected_1": n_tps,
        "n_gds_corrected_0": n_gds,
        "original_vs_saved": {
            "n_within_0.0005": n - len(original_mismatches),
            "n_mismatches": len(original_mismatches),
            "mismatch_ids": [r["reddit_id"] for r in original_mismatches],
            "max_abs_float_diff": max((r["original_vs_saved"]["max_abs_float_diff"] for r in rows), default=0.0),
        },
        "original_author_track": {
            "n_title_fallback": title_fallback,
            "n_used_body_as_first_point": used_body,
            "n_community_mean_observed": sum(1 for r in rows if r["original_reconstructed"].get("community_mean_is_observed")),
        },
        "package_native": {
            "n_author_avg_observed": n - pkg_author_unobs,
            "n_author_avg_unobserved_zero": pkg_author_unobs,
            "n_community_avg_observed": n - pkg_comm_unobs,
            "n_community_avg_unobserved_zero": pkg_comm_unobs,
            "n_author_avg_value_zero": pkg_author_zero,
            "note": "trajectory_summary author_avg uses author COMMENTS only and returns 0.0 when none exist.",
        },
        "harmonized": {
            "n_author_comment_mean_observed": harm_author_obs,
            "n_community_comment_mean_observed": harm_comm_obs,
            "n_author_shift_eligible_ge4": author_shift_elig,
            "n_community_shift_eligible_ge4": comm_shift_elig,
        },
        "definition_gaps": {
            "n_comment_order_string_vs_parsed_diff": order_diff,
            "mean_abs_original_author_mean_vs_package_author_avg": _mean(abs_orig_pkg_author),
            "mean_abs_original_vs_harmonized_original_like_author": _mean(abs_orig_harm_author),
            "mean_abs_package_vs_harmonized_package_like_author": _mean(abs_pkg_harm_author),
            "reason_counts": dict(reason_counts),
        },
        "representative_cases": representative,
        "limitations": [
            "Package-native author_avg is not the C1-prime author track.",
            "A package 0.0 average is not observed neutrality when the role has no comments.",
            "Historical labels are the old 86 TPS / 456 GDS definitions, not S2.",
            "Protected tests were not scored.",
        ],
    }


def run_historical_542(loaded: dict[str, Any]) -> dict[str, Any]:
    analyzer = SentimentIntensityAnalyzer()
    labels = loaded["historical_labels"]["by_id"]
    posts = loaded["historical_raw"]["merged_by_id_original_rule"]
    saved = loaded["c1_prime"]["by_id"]
    missing_raw = sorted(rid for rid in labels if rid not in posts)
    missing_saved = sorted(rid for rid in labels if rid not in saved)
    if missing_raw or missing_saved:
        raise RuntimeError(
            f"historical comparison blocked: missing_raw={missing_raw[:20]} missing_saved={missing_saved[:20]}"
        )
    rows = []
    for reddit_id in loaded["historical_labels"]["inventory"]["ids"]:
        rows.append(compare_one(reddit_id, labels[reddit_id], posts[reddit_id], saved[reddit_id], analyzer))
    summary = summarize_rows(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(RESULTS_DIR / "historical_542_original_vs_package.jsonl", rows)
    write_json(RESULTS_DIR / "historical_542_summary.json", summary)
    return {"n": len(rows), "missing_raw": missing_raw, "missing_saved": missing_saved, "summary": summary}
