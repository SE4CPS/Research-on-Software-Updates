"""Descriptive TPS vs GDS sentiment contrasts and sensitivities."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Callable, Iterable

from full_resource_sentiment_v1.inference import (
    bootstrap_mean_median_ci,
    bootstrap_spearman,
    paired_difference_summary,
    stratified_mean_diff,
)
from full_resource_sentiment_v1.tps_gds_sentiment_analysis.dictionary import FROZEN_DEF, HISTORICAL_DEF

SEED = 20261102
N_BOOT = 2000

POST_LEVEL_FIELDS = (
    "title_compound",
    "body_compound",
    "combined_compound",
    "body_minus_title",
)
CADENCE_FIELDS = (
    "author_mean",
    "community_mean",
    "author_shift",
    "first_reply_compound",
    "last_reply_compound",
    "last_minus_first",
    "order_slope_normalized",
    "elapsed_hours",
)
ALL_FIELDS = POST_LEVEL_FIELDS + CADENCE_FIELDS


def _vals(rows: Iterable[dict[str, Any]], field: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        value = row.get(field)
        if value is None:
            continue
        if field == "elapsed_hours" and not row.get("elapsed_eligible"):
            continue
        if field == "order_slope_normalized" and row.get("order_eligible") is False:
            continue
        if field == "last_minus_first" and row.get("order_eligible") is False:
            continue
        out.append(float(value))
    return out


def _obs_count(rows: list[dict[str, Any]], field: str) -> int:
    return len(_vals(rows, field))


def filter_rows(
    rows: list[dict[str, Any]],
    *,
    relevance: str | None = "relevant",
    discussions: set[str] | None = None,
    sentiment_definition: str | None = FROZEN_DEF,
    predicate: Callable[[dict[str, Any]], bool] | None = None,
) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if relevance is not None and row.get("final_relevance") != relevance:
            continue
        if discussions is not None and row.get("final_discussion") not in discussions:
            continue
        if sentiment_definition is not None and row.get("sentiment_definition") != sentiment_definition:
            continue
        if predicate is not None and not predicate(row):
            continue
        out.append(row)
    return out


def group_descriptives(rows: list[dict[str, Any]], fields: tuple[str, ...] = ALL_FIELDS) -> dict[str, Any]:
    coverage = {
        "n_rows": len(rows),
        "title_nonempty": sum(bool(r.get("title_nonempty")) for r in rows),
        "body_nonempty": sum(bool(r.get("body_nonempty")) for r in rows),
        "author_reply": sum(int(r.get("n_author_replies_nonempty") or 0) > 0 for r in rows),
        "community_reply": sum(int(r.get("n_community_replies_nonempty") or 0) > 0 for r in rows),
        "both_roles": sum(
            int(r.get("n_author_replies_nonempty") or 0) > 0 and int(r.get("n_community_replies_nonempty") or 0) > 0
            for r in rows
        ),
        "order_eligible": sum(bool(r.get("order_eligible")) for r in rows),
        "elapsed_eligible": sum(bool(r.get("elapsed_eligible")) for r in rows),
        "count_gap_true": sum(r.get("count_gap") is True for r in rows),
        "count_gap_false": sum(r.get("count_gap") is False for r in rows),
        "count_gap_missing": sum(r.get("count_gap") is None for r in rows),
        "decision_source": dict(Counter(str(r.get("decision_source")) for r in rows)),
        "selection_stratum": dict(Counter(str(r.get("selection_stratum")) for r in rows)),
    }
    metrics = {}
    for field in fields:
        values = _vals(rows, field)
        metrics[field] = {
            "n_observed": len(values),
            "summary": bootstrap_mean_median_ci(values, n_boot=N_BOOT, seed=SEED) if values else None,
            "fraction_lt_neg_0_05": (sum(v < -0.05 for v in values) / len(values)) if values else None,
            "fraction_gt_pos_0_05": (sum(v > 0.05 for v in values) / len(values)) if values else None,
        }
    return {"coverage": coverage, "metrics": metrics}


def contrast_gds_minus_tps(
    tps: list[dict[str, Any]],
    gds: list[dict[str, Any]],
    fields: tuple[str, ...] = ALL_FIELDS,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in fields:
        a = _vals(gds, field)
        b = _vals(tps, field)
        diff = stratified_mean_diff(a, b, n_boot=N_BOOT, seed=SEED, difference_name="gds_minus_tps")
        out[field] = diff
    return out


def cadence_block(rows_by_label: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Within-conversation cadence; frozen definition only."""
    result: dict[str, Any] = {"note": "Sentiment change is not issue resolution."}
    for label, rows in rows_by_label.items():
        author_shift_rows = [r for r in rows if r.get("author_shift") is not None]
        both = [
            r
            for r in rows
            if r.get("author_mean") is not None and r.get("community_mean") is not None
        ]
        order_rows = [r for r in rows if r.get("order_eligible") and r.get("last_minus_first") is not None]
        slope_rows = [r for r in rows if r.get("order_eligible") and r.get("order_slope_normalized") is not None]
        elapsed_rows = [r for r in rows if r.get("elapsed_eligible") and r.get("elapsed_hours") is not None]
        # author vs combined post among author-eligible
        post_vs_author = []
        for r in rows:
            if r.get("author_mean") is None or r.get("combined_compound") is None:
                continue
            post_vs_author.append(float(r["author_mean"]) - float(r["combined_compound"]))
        author_vs_community = []
        for r in both:
            author_vs_community.append(float(r["author_mean"]) - float(r["community_mean"]))
        # elapsed association with last_minus_first among dual-eligible
        dual = [
            r
            for r in rows
            if r.get("elapsed_eligible")
            and r.get("elapsed_hours") is not None
            and r.get("order_eligible")
            and r.get("last_minus_first") is not None
        ]
        spearman = None
        if len(dual) >= 3:
            spearman = bootstrap_spearman(
                [float(r["elapsed_hours"]) for r in dual],
                [float(r["last_minus_first"]) for r in dual],
                n_boot=N_BOOT,
                seed=SEED,
            )
        result[label] = {
            "n_rows": len(rows),
            "n_author_shift": len(author_shift_rows),
            "n_author_minus_post_derived": len(post_vs_author),
            "n_both_roles": len(both),
            "n_order_eligible_last_minus_first": len(order_rows),
            "n_order_slope": len(slope_rows),
            "n_elapsed": len(elapsed_rows),
            "n_elapsed_with_last_minus_first": len(dual),
            "author_shift": bootstrap_mean_median_ci(
                _vals(author_shift_rows, "author_shift"), n_boot=N_BOOT, seed=SEED
            )
            if author_shift_rows
            else None,
            "author_mean_minus_combined_post": bootstrap_mean_median_ci(post_vs_author, n_boot=N_BOOT, seed=SEED)
            if post_vs_author
            else None,
            "author_mean_minus_community_mean": bootstrap_mean_median_ci(
                author_vs_community, n_boot=N_BOOT, seed=SEED
            )
            if author_vs_community
            else None,
            "last_minus_first": bootstrap_mean_median_ci(
                _vals(order_rows, "last_minus_first"), n_boot=N_BOOT, seed=SEED
            )
            if order_rows
            else None,
            "order_slope_normalized": bootstrap_mean_median_ci(
                _vals(slope_rows, "order_slope_normalized"), n_boot=N_BOOT, seed=SEED
            )
            if slope_rows
            else None,
            "elapsed_hours": bootstrap_mean_median_ci(_vals(elapsed_rows, "elapsed_hours"), n_boot=N_BOOT, seed=SEED)
            if elapsed_rows
            else None,
            "spearman_elapsed_vs_last_minus_first": spearman,
        }
    return result


def length_diagnostics(tps: list[dict[str, Any]], gds: list[dict[str, Any]], other: list[dict[str, Any]]) -> dict[str, Any]:
    """Workstream-I-style length caution for title/body without mixing prefix estimands."""

    def length_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        paired = [r for r in rows if r.get("title_compound") is not None and r.get("body_compound") is not None]
        body_lens = [float(r["body_len"]) for r in paired if r.get("body_len") is not None]
        title_lens = [float(r["title_len"]) for r in paired if r.get("title_len") is not None]
        ratios = []
        for r in paired:
            if r.get("body_len") is None or r.get("title_len") is None:
                continue
            t = float(r["title_len"])
            if t <= 0:
                continue
            ratios.append(float(r["body_len"]) / t)
        # Stratify body−title by body length bands when present.
        by_band: dict[str, list[float]] = defaultdict(list)
        for r in paired:
            band = str(r.get("body_len_band") or "unknown")
            if r.get("body_minus_title") is not None:
                by_band[band].append(float(r["body_minus_title"]))
        return {
            "n_paired_title_body": len(paired),
            "n_with_body_len": len(body_lens),
            "body_len": bootstrap_mean_median_ci(body_lens, n_boot=N_BOOT, seed=SEED) if body_lens else None,
            "title_len": bootstrap_mean_median_ci(title_lens, n_boot=N_BOOT, seed=SEED) if title_lens else None,
            "body_over_title_len_ratio": bootstrap_mean_median_ci(ratios, n_boot=N_BOOT, seed=SEED) if ratios else None,
            "body_minus_title_overall": paired_difference_summary(
                [float(r["title_compound"]) for r in paired],
                [float(r["body_compound"]) for r in paired],
                n_boot=N_BOOT,
                seed=SEED,
                difference_name="body_minus_title",
            )
            if paired
            else None,
            "body_minus_title_by_body_len_band": {
                band: bootstrap_mean_median_ci(vals, n_boot=N_BOOT, seed=SEED) for band, vals in sorted(by_band.items())
            },
            "note": (
                "Full-text body−title is retained as its own estimand. "
                "Equal-length prefix contrasts from Workstream I are a different outcome and are not recomputed or mixed here."
            ),
        }

    # Between-group body length and body−title
    tps_body = [float(r["body_len"]) for r in tps if r.get("body_len") is not None and r.get("body_nonempty")]
    gds_body = [float(r["body_len"]) for r in gds if r.get("body_len") is not None and r.get("body_nonempty")]
    return {
        "TPS": length_summary(tps),
        "GDS": length_summary(gds),
        "Other": length_summary(other),
        "gds_minus_tps_body_len": stratified_mean_diff(
            gds_body, tps_body, n_boot=N_BOOT, seed=SEED, difference_name="gds_minus_tps_body_len"
        )
        if tps_body and gds_body
        else None,
        "gds_minus_tps_body_minus_title": contrast_gds_minus_tps(tps, gds, fields=("body_minus_title",)).get(
            "body_minus_title"
        ),
    }


def reply_count_bucket(row: dict[str, Any]) -> str:
    n = row.get("reply_role_total")
    if n is None:
        stored = row.get("n_stored_comments")
        if stored is None:
            return "unknown"
        n = int(stored)
    n = int(n)
    if n <= 0:
        return "0"
    if n == 1:
        return "1"
    if n <= 3:
        return "2_3"
    if n <= 10:
        return "4_10"
    return "11_plus"


def sensitivity_suite(all_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Sensitivity contrasts; master table always retains all rows."""

    def slice_contrast(name: str, predicate: Callable[[dict[str, Any]], bool] | None = None) -> dict[str, Any]:
        base = filter_rows(all_rows, sentiment_definition=FROZEN_DEF, predicate=predicate)
        tps = [r for r in base if r["final_discussion"] == "TPS"]
        gds = [r for r in base if r["final_discussion"] == "GDS"]
        fields = ("combined_compound", "body_compound", "title_compound", "author_mean", "last_minus_first")
        return {
            "slice": name,
            "n_relevant_frozen": len(base),
            "n_tps": len(tps),
            "n_gds": len(gds),
            "contrast": contrast_gds_minus_tps(tps, gds, fields=fields),
            "observed_n": {f: {"TPS": _obs_count(tps, f), "GDS": _obs_count(gds, f)} for f in fields},
        }

    slices = [
        slice_contrast("all_compatible_frozen_relevant"),
        slice_contrast("existing_decisions_only", lambda r: r["is_existing_decision"]),
        slice_contrast("new_model_decisions_only", lambda r: r["is_new_model_decision"]),
        slice_contrast("selection_older", lambda r: r.get("selection_stratum") == "older"),
        slice_contrast("selection_october_2026", lambda r: r.get("selection_stratum") == "october_2026"),
        slice_contrast("count_gap", lambda r: r.get("count_gap") is True),
        slice_contrast("no_count_gap", lambda r: r.get("count_gap") is False),
        slice_contrast("confidence_high", lambda r: r.get("confidence_band") == "high"),
        slice_contrast("confidence_lower_or_missing", lambda r: r.get("confidence_band") != "high"),
    ]
    # Reply-count groups
    reply_slices = []
    for bucket in ("0", "1", "2_3", "4_10", "11_plus"):
        reply_slices.append(slice_contrast(f"reply_bucket_{bucket}", lambda r, b=bucket: reply_count_bucket(r) == b))

    # Historical-only post-level (no cadence pooling)
    hist = filter_rows(all_rows, sentiment_definition=HISTORICAL_DEF)
    hist_tps = [r for r in hist if r["final_discussion"] == "TPS"]
    hist_gds = [r for r in hist if r["final_discussion"] == "GDS"]
    historical_block = {
        "n_relevant_historical": len(hist),
        "n_tps": len(hist_tps),
        "n_gds": len(hist_gds),
        "note": "Separate subset; cadence fields absent on historical definition in this table.",
        "contrast_post_level": contrast_gds_minus_tps(
            hist_tps, hist_gds, fields=("title_compound", "body_compound", "combined_compound", "author_mean", "community_mean")
        ),
    }
    return {
        "frozen_slices": slices,
        "reply_count_slices": reply_slices,
        "historical_relevant_separate": historical_block,
        "master_table_n_retained": len(all_rows),
    }


def run_primary_analysis(all_rows: list[dict[str, Any]]) -> dict[str, Any]:
    frozen_rel = filter_rows(all_rows, sentiment_definition=FROZEN_DEF)
    tps = [r for r in frozen_rel if r["final_discussion"] == "TPS"]
    gds = [r for r in frozen_rel if r["final_discussion"] == "GDS"]
    other = [r for r in frozen_rel if r["final_discussion"] == "Other"]
    unclear = [r for r in frozen_rel if r["final_discussion"] == "Unclear"]

    # Also report Other context descriptives, never merged into GDS.
    groups = {
        "TPS": group_descriptives(tps),
        "GDS": group_descriptives(gds),
        "Other": group_descriptives(other),
        "Unclear": group_descriptives(unclear, fields=POST_LEVEL_FIELDS),
    }
    primary_contrast = contrast_gds_minus_tps(tps, gds)
    other_vs_tps = {}
    for field in ALL_FIELDS:
        other_vs_tps[field] = stratified_mean_diff(
            _vals(other, field),
            _vals(tps, field),
            n_boot=N_BOOT,
            seed=SEED,
            difference_name="other_minus_tps",
        )

    return {
        "population": (
            f"final_relevance==relevant AND sentiment_definition=={FROZEN_DEF}; "
            "primary contrast GDS vs TPS; Other is context only"
        ),
        "n_relevant_frozen": len(frozen_rel),
        "n_by_discussion": {"TPS": len(tps), "GDS": len(gds), "Other": len(other), "Unclear": len(unclear)},
        "groups": groups,
        "gds_minus_tps": primary_contrast,
        "other_minus_tps": other_vs_tps,
        "cadence": cadence_block({"TPS": tps, "GDS": gds, "Other": other}),
        "length_diagnostics": length_diagnostics(tps, gds, other),
        "bootstrap": {"seed": SEED, "n_boot": N_BOOT, "unit": "post/thread equal weight"},
    }


def eligibility_table(all_rows: list[dict[str, Any]]) -> dict[str, Any]:
    frozen_rel = filter_rows(all_rows, sentiment_definition=FROZEN_DEF)
    out: dict[str, Any] = {}
    for label in ("TPS", "GDS", "Other"):
        rows = [r for r in frozen_rel if r["final_discussion"] == label]
        out[label] = {field: _obs_count(rows, field) for field in ALL_FIELDS}
        out[label]["n_rows"] = len(rows)
    return out
