"""Corrected VADER TPS vs GDS contrasts, length adjustment, and sensitivities."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Iterable

import numpy as np

from full_resource_sentiment_v1.inference import percentile_ci, stratified_mean_diff
from full_resource_sentiment_v1.tps_gds_sentiment_v2.dictionary import (
    FROZEN_DEF,
    HISTORICAL_DEF,
    N_BOOT,
    SEED,
)

PRIMARY_FIELDS = (
    "title_compound",
    "body_compound",
    "combined_compound",
    "author_mean",
    "community_mean",
    "author_shift",
    "original_post_to_author_mean",
    "last_minus_first",
    "order_slope_normalized",
    "elapsed_hours",
)


def _field_values(rows: Iterable[dict[str, Any]], field: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        if field == "author_shift" and not row.get("author_shift_eligible"):
            continue
        if field == "original_post_to_author_mean" and not row.get("post_to_author_eligible"):
            continue
        if field == "elapsed_hours" and not row.get("elapsed_eligible"):
            continue
        if field in {"last_minus_first", "order_slope_normalized"} and row.get("order_eligible") is False:
            continue
        value = row.get(field)
        if value is None:
            continue
        out.append(float(value))
    return out


def filter_frozen_relevant(
    rows: list[dict[str, Any]],
    *,
    predicate: Callable[[dict[str, Any]], bool] | None = None,
) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if row.get("final_relevance") != "relevant":
            continue
        if row.get("sentiment_definition") != FROZEN_DEF:
            continue
        if predicate is not None and not predicate(row):
            continue
        out.append(row)
    return out


def split_tps_gds(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    tps = [r for r in rows if r.get("final_discussion") == "TPS"]
    gds = [r for r in rows if r.get("final_discussion") == "GDS"]
    other = [r for r in rows if r.get("final_discussion") == "Other"]
    return tps, gds, other


def contrast(
    gds: list[dict[str, Any]],
    tps: list[dict[str, Any]],
    fields: tuple[str, ...] = PRIMARY_FIELDS,
    *,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in fields:
        out[field] = stratified_mean_diff(
            _field_values(gds, field),
            _field_values(tps, field),
            n_boot=n_boot,
            seed=seed,
            difference_name="gds_minus_tps",
        )
    return out


def eligibility_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tps, gds, other = split_tps_gds(rows)
    block: dict[str, Any] = {}
    for label, group in (("TPS", tps), ("GDS", gds), ("Other", other)):
        block[label] = {
            "n_rows": len(group),
            **{field: len(_field_values(group, field)) for field in PRIMARY_FIELDS},
            "n_author_shift_eligible_flag": sum(1 for r in group if r.get("author_shift_eligible")),
            "n_post_to_author_eligible_flag": sum(1 for r in group if r.get("post_to_author_eligible")),
            "author_shift_definition": "later_minus_earlier_author_replies_ge4",
            "post_to_author_definition": "author_mean_minus_combined_compound_ge1_author_reply",
        }
    return block


def _band_contrast_point(
    tps: list[dict[str, Any]],
    gds: list[dict[str, Any]],
    *,
    score_field: str,
    band_key: str,
) -> dict[str, Any]:
    """Standardized GDS-TPS contrast across bands with common support only."""
    pooled = [r for r in (tps + gds) if r.get(band_key) is not None and r.get(score_field) is not None]
    # For body score, also require body_len; for title score require title_len — already via band.
    band_names = sorted({str(r[band_key]) for r in pooled})
    band_rows: dict[str, dict[str, list[float]]] = {}
    n_pooled = len(pooled)
    details = []
    weighted_sum = 0.0
    weight_sum = 0.0
    for name in band_names:
        t_vals = [float(r[score_field]) for r in tps if r.get(band_key) == name and r.get(score_field) is not None]
        g_vals = [float(r[score_field]) for r in gds if r.get(band_key) == name and r.get(score_field) is not None]
        n_band_pooled = sum(1 for r in pooled if r.get(band_key) == name)
        has_both = len(t_vals) > 0 and len(g_vals) > 0
        entry = {
            "band": name,
            "n_tps": len(t_vals),
            "n_gds": len(g_vals),
            "n_pooled_in_band": n_band_pooled,
            "pooled_share": (n_band_pooled / n_pooled) if n_pooled else None,
            "mean_tps": float(np.mean(t_vals)) if t_vals else None,
            "mean_gds": float(np.mean(g_vals)) if g_vals else None,
            "mean_gds_minus_tps": (float(np.mean(g_vals) - np.mean(t_vals)) if has_both else None),
            "common_support": has_both,
            "included_in_standardized": has_both,
        }
        details.append(entry)
        if has_both and n_pooled:
            w = n_band_pooled / n_pooled
            # Re-normalize later over included bands only.
            weighted_sum += w * float(entry["mean_gds_minus_tps"])
            weight_sum += w
            band_rows[name] = {"tps": t_vals, "gds": g_vals, "n_pooled": n_band_pooled}

    if weight_sum > 0:
        # Normalize weights over bands with common support so they sum to 1 among included bands.
        included_share = weight_sum
        standardized = weighted_sum / included_share
        renormalized = True
    else:
        standardized = None
        included_share = 0.0
        renormalized = False

    excluded = [d["band"] for d in details if not d["common_support"]]
    return {
        "estimand": f"length_band_standardized_gds_minus_tps_on_{score_field}_by_{band_key}",
        "score_field": score_field,
        "band_key": band_key,
        "n_tps_eligible": sum(1 for r in tps if r.get(score_field) is not None and r.get(band_key) is not None),
        "n_gds_eligible": sum(1 for r in gds if r.get(score_field) is not None and r.get(band_key) is not None),
        "n_pooled_eligible": n_pooled,
        "bands": details,
        "bands_missing_a_class": excluded,
        "standardized_mean_diff": standardized,
        "weights_renormalized_over_common_support_bands": renormalized,
        "common_support_pooled_share": included_share if n_pooled else None,
        "_band_rows_for_boot": band_rows,
        "_pooled_n": n_pooled,
    }


def _bootstrap_standardized(
    tps: list[dict[str, Any]],
    gds: list[dict[str, Any]],
    *,
    score_field: str,
    band_key: str,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict[str, Any]:
    point = _band_contrast_point(tps, gds, score_field=score_field, band_key=band_key)
    # Bootstrap at post level within class, then recompute standardized contrast.
    t_idx = [r for r in tps if r.get(score_field) is not None and r.get(band_key) is not None]
    g_idx = [r for r in gds if r.get(score_field) is not None and r.get(band_key) is not None]
    if not t_idx or not g_idx or point["standardized_mean_diff"] is None:
        point["mean_diff_ci95"] = None
        point["n_bootstrap"] = n_boot
        point["seed"] = seed
        point.pop("_band_rows_for_boot", None)
        return point

    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(n_boot):
        t_s = [t_idx[i] for i in rng.integers(0, len(t_idx), size=len(t_idx))]
        g_s = [g_idx[i] for i in rng.integers(0, len(g_idx), size=len(g_idx))]
        redo = _band_contrast_point(t_s, g_s, score_field=score_field, band_key=band_key)
        if redo["standardized_mean_diff"] is not None:
            draws.append(float(redo["standardized_mean_diff"]))
    point["mean_diff_ci95"] = percentile_ci(draws) if draws else None
    point["n_bootstrap"] = n_boot
    point["seed"] = seed
    point["n_defined_resamples"] = len(draws)
    # Unadjusted companion on same eligible rows
    point["unadjusted_on_same_eligible"] = stratified_mean_diff(
        [float(r[score_field]) for r in g_idx],
        [float(r[score_field]) for r in t_idx],
        n_boot=n_boot,
        seed=seed,
        difference_name="gds_minus_tps_unadjusted_same_eligible",
    )
    point.pop("_band_rows_for_boot", None)
    point.pop("_pooled_n", None)
    return point


def length_adjusted_block(tps: list[dict[str, Any]], gds: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "note": (
            "Prespecified character bands. Standardized contrast uses pooled eligible-post "
            "weights renormalized over bands with both classes. No extrapolation through empty cells. "
            "Within-post body-minus-title bands are NOT used as between-group adjustment."
        ),
        "body_compound_by_body_band": _bootstrap_standardized(
            tps, gds, score_field="body_compound", band_key="body_band"
        ),
        "title_compound_by_title_band": _bootstrap_standardized(
            tps, gds, score_field="title_compound", band_key="title_band"
        ),
        "combined_compound_by_body_band": _bootstrap_standardized(
            tps, gds, score_field="combined_compound", band_key="body_band"
        ),
    }


def secondary_regression_adjustment(
    tps: list[dict[str, Any]],
    gds: list[dict[str, Any]],
    *,
    score_field: str = "combined_compound",
) -> dict[str, Any]:
    """OLS of score ~ GDS + log1p(body_len) + title_len + oct2026 + existing_decision.

    Secondary only; reported separately from band-standardized estimand.
    """
    rows = []
    for label, group, gds_flag in (("TPS", tps, 0.0), ("GDS", gds, 1.0)):
        for r in group:
            if r.get(score_field) is None or r.get("body_len") is None or r.get("title_len") is None:
                continue
            rows.append(
                {
                    "y": float(r[score_field]),
                    "gds": gds_flag,
                    "log_body": float(np.log1p(float(r["body_len"]))),
                    "title_len": float(r["title_len"]),
                    "oct2026": 1.0 if r.get("selection_stratum") == "october_2026" else 0.0,
                    "existing": 1.0 if r.get("is_existing_decision") else 0.0,
                    "label": label,
                }
            )
    if len(rows) < 10:
        return {"status": "insufficient_rows", "n": len(rows), "score_field": score_field}
    y = np.asarray([r["y"] for r in rows], dtype=float)
    x = np.column_stack(
        [
            np.ones(len(rows)),
            [r["gds"] for r in rows],
            [r["log_body"] for r in rows],
            [r["title_len"] for r in rows],
            [r["oct2026"] for r in rows],
            [r["existing"] for r in rows],
        ]
    )
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    # Residual bootstrap of coefficient on GDS
    fitted = x @ beta
    resid = y - fitted
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(N_BOOT):
        yb = fitted + resid[rng.integers(0, len(resid), size=len(resid))]
        bb, *_ = np.linalg.lstsq(x, yb, rcond=None)
        boots.append(float(bb[1]))
    return {
        "status": "ok",
        "estimand": "secondary_ols_gds_coefficient_adjusted_for_log_body_title_date_origin",
        "score_field": score_field,
        "n": len(rows),
        "n_tps": sum(1 for r in rows if r["label"] == "TPS"),
        "n_gds": sum(1 for r in rows if r["label"] == "GDS"),
        "coefficients": {
            "intercept": float(beta[0]),
            "gds": float(beta[1]),
            "log1p_body_len": float(beta[2]),
            "title_len": float(beta[3]),
            "selection_october_2026": float(beta[4]),
            "existing_decision": float(beta[5]),
        },
        "gds_coefficient_boot_ci95": percentile_ci(boots),
        "n_bootstrap": N_BOOT,
        "seed": SEED,
        "note": "Secondary; not a substitute for the band-standardized contrast.",
    }


def sensitivity_suite(all_rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "combined_compound",
        "body_compound",
        "title_compound",
        "author_shift",
        "original_post_to_author_mean",
    )

    def one(name: str, pred: Callable[[dict[str, Any]], bool] | None = None) -> dict[str, Any]:
        base = filter_frozen_relevant(all_rows, predicate=pred)
        tps, gds, _other = split_tps_gds(base)
        return {
            "slice": name,
            "n_tps": len(tps),
            "n_gds": len(gds),
            "contrast": contrast(gds, tps, fields=fields),
            "eligibility": {
                field: {"TPS": len(_field_values(tps, field)), "GDS": len(_field_values(gds, field))}
                for field in fields
            },
        }

    frozen_slices = [
        one("all_frozen_relevant"),
        one("existing_decisions_only", lambda r: r["is_existing_decision"]),
        one("new_model_decisions_only", lambda r: r["is_new_model_decision"]),
        one("selection_older", lambda r: r.get("selection_stratum") == "older"),
        one("selection_october_2026", lambda r: r.get("selection_stratum") == "october_2026"),
        one("count_gap", lambda r: r.get("count_gap") is True),
        one("no_count_gap", lambda r: r.get("count_gap") is False),
        one("confidence_high", lambda r: str(r.get("decision_confidence") or "").lower() == "high"),
        one("confidence_not_high", lambda r: str(r.get("decision_confidence") or "").lower() != "high"),
    ]
    reply_slices = [
        one(f"reply_bucket_{b}", lambda r, bucket=b: r.get("reply_bucket") == bucket)
        for b in ("0", "1", "2_3", "4_10", "11_plus")
    ]

    hist = [
        r
        for r in all_rows
        if r.get("final_relevance") == "relevant" and r.get("sentiment_definition") == HISTORICAL_DEF
    ]
    h_tps, h_gds, _ = split_tps_gds(hist)
    historical = {
        "note": "Separate historical definition; cadence/author_shift often unavailable; not pooled into frozen.",
        "n_relevant": len(hist),
        "n_tps": len(h_tps),
        "n_gds": len(h_gds),
        "contrast_post_level": contrast(
            h_gds,
            h_tps,
            fields=("title_compound", "body_compound", "combined_compound", "author_mean", "community_mean"),
        ),
    }
    return {
        "frozen_slices": frozen_slices,
        "reply_count_slices": reply_slices,
        "historical_separate": historical,
        "master_table_n": len(all_rows),
    }


def provenance_author_shift(gds: list[dict[str, Any]], tps: list[dict[str, Any]]) -> dict[str, Any]:
    """Break down author_shift eligible rows by decision_source (exploratory)."""

    def by_source(rows: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for r in rows:
            if r.get("author_shift_eligible"):
                counts[str(r.get("decision_source"))] += 1
        return dict(counts)

    return {
        "TPS_author_shift_by_source": by_source(tps),
        "GDS_author_shift_by_source": by_source(gds),
        "note": "Small GDS author_shift cell is exploratory; prior 2k-boot CI was unstable near zero.",
    }


def run_stage1(all_rows: list[dict[str, Any]]) -> dict[str, Any]:
    frozen = filter_frozen_relevant(all_rows)
    tps, gds, other = split_tps_gds(frozen)
    primary = contrast(gds, tps)
    other_vs_tps = {}
    for field in PRIMARY_FIELDS:
        other_vs_tps[field] = stratified_mean_diff(
            _field_values(other, field),
            _field_values(tps, field),
            n_boot=N_BOOT,
            seed=SEED,
            difference_name="other_minus_tps",
        )
    return {
        "population": f"final_relevance==relevant AND sentiment_definition=={FROZEN_DEF}",
        "n_relevant_frozen": len(frozen),
        "n_by_discussion": {"TPS": len(tps), "GDS": len(gds), "Other": len(other)},
        "bootstrap": {"n_boot": N_BOOT, "seed": SEED, "unit": "post"},
        "eligibility": eligibility_counts(frozen),
        "gds_minus_tps_unadjusted": primary,
        "other_minus_tps_context": other_vs_tps,
        "author_shift_provenance": provenance_author_shift(gds, tps),
        "length_adjusted": length_adjusted_block(tps, gds),
        "secondary_ols_combined": secondary_regression_adjustment(tps, gds, score_field="combined_compound"),
        "secondary_ols_body": secondary_regression_adjustment(tps, gds, score_field="body_compound"),
        "cliffs_delta_note": "Cliff's delta accompanies unadjusted stratified_mean_diff outputs.",
        "validation_status": "200_case_review_deferred_zero_saved_decisions",
    }
