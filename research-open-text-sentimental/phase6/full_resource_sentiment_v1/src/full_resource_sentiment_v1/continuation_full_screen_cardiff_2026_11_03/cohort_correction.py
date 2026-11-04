"""Supplemental reporting fix: matched-ID TextBlob join effect; later_author incomparable."""

from __future__ import annotations

from typing import Any

import numpy as np

from full_resource_sentiment_v1.continuation_full_screen_cardiff_2026_11_03 import N_BOOT, SEED
from full_resource_sentiment_v1.continuation_full_screen_cardiff_2026_11_03.common import (
    CORR_RESTRICTED,
    EXACT_JOIN_RESTRICTED,
    OUT_ROOT,
    load_jsonl,
    utcnow,
    write_json,
)
from full_resource_sentiment_v1.master_research_corrections_2026_11_03.textblob_checks import (
    independent_stratified_mean_diff,
)


def run_cohort_correction() -> dict[str, Any]:
    dest = OUT_ROOT / "phase0_cohort_reporting_correction"
    dest.mkdir(parents=True, exist_ok=True)

    old = load_jsonl(CORR_RESTRICTED / "matched_role_sentiment_units.jsonl")
    new = load_jsonl(EXACT_JOIN_RESTRICTED / "analysis_units_exact_join.jsonl")

    def by_unit(rows, unit):
        return {
            str(u["reddit_id"]): u
            for u in rows
            if u.get("unit") == unit and u.get("textblob") is not None and u.get("label") in {"TPS", "GDS"}
        }

    old_fa = by_unit(old, "first_author_reply")
    new_fa = by_unit(new, "first_author_reply")
    common_ids = sorted(set(old_fa) & set(new_fa))
    only_new = sorted(set(new_fa) - set(old_fa))
    only_old = sorted(set(old_fa) - set(new_fa))

    # Paired scores on identical IDs
    paired_diffs = []
    changed = 0
    for rid in common_ids:
        o = float(old_fa[rid]["textblob"])
        n = float(new_fa[rid]["textblob"])
        if abs(o - n) > 1e-12:
            changed += 1
        paired_diffs.append(n - o)

    # GDS−TPS on identical ID sets (old scores vs new scores)
    def contrast_on_ids(mapping, ids):
        g = [float(mapping[i]["textblob"]) for i in ids if mapping[i]["label"] == "GDS"]
        t = [float(mapping[i]["textblob"]) for i in ids if mapping[i]["label"] == "TPS"]
        return independent_stratified_mean_diff(g, t, seed=SEED + 60, n_boot=N_BOOT)

    old_on_common = contrast_on_ids(old_fa, common_ids)
    new_on_common = contrast_on_ids(new_fa, common_ids)
    # Prior published used 98/955 vs repaired 98/956 — document but do not use as join effect
    prior_published = {
        "note": "Prior delta mixed one added TPS ID with join changes — not a pure join effect",
        "prior_role_order_n": {"gds": 98, "tps": 955},
        "exact_join_n": {"gds": 98, "tps": 956},
        "prior_quoted_delta": -0.0004081165502794887,
    }

    # later_author: incomparable estimands
    old_later = by_unit(old, "later_author_reply")
    new_later = by_unit(new, "later_author_reply")
    new_second = by_unit(new, "second_author_reply")
    later_note = {
        "status": "INCOMPARABLE — do not use old/new later_author delta as join-repair effect",
        "old_later_author_reply_definition": (
            "In prior role-order stage2: last author reply when n_author>=2 "
            "(includes second-of-two as 'later')"
        ),
        "new_later_author_reply_definition": (
            "In exact-join analysis units: last author reply only when n_author>=3 "
            "(second reply is separate unit second_author_reply)"
        ),
        "old_n": {"total_with_tb": len(old_later), "gds": sum(1 for u in old_later.values() if u["label"]=="GDS"),
                  "tps": sum(1 for u in old_later.values() if u["label"]=="TPS")},
        "new_later_n": {"total_with_tb": len(new_later),
                        "gds": sum(1 for u in new_later.values() if u["label"]=="GDS"),
                        "tps": sum(1 for u in new_later.values() if u["label"]=="TPS")},
        "new_second_author_n": len(new_second),
        "vader_unchanged_by_text_join": True,
        "action": "Removed from join-repair delta table; report separately if needed as definitional change",
    }

    report = {
        "utc": utcnow(),
        "sealed_trees_untouched": True,
        "first_author_textblob_matched_ids": {
            "n_common_ids": len(common_ids),
            "n_only_new": len(only_new),
            "n_only_old": len(only_old),
            "only_new_ids": only_new,
            "n_textblob_changed_among_common": changed,
            "mean_paired_new_minus_old": float(np.mean(paired_diffs)) if paired_diffs else None,
            "median_paired_new_minus_old": float(np.median(paired_diffs)) if paired_diffs else None,
            "n_identical_scores": len(common_ids) - changed,
            "gds_minus_tps_on_common_ids_using_old_scores": old_on_common,
            "gds_minus_tps_on_common_ids_using_new_scores": new_on_common,
            "join_effect_on_contrast": {
                "mean_diff_new_minus_old_contrast": (
                    None
                    if old_on_common.get("mean_diff") is None or new_on_common.get("mean_diff") is None
                    else float(new_on_common["mean_diff"]) - float(old_on_common["mean_diff"])
                ),
                "interpretation": (
                    "Difference between GDS−TPS contrasts computed on the SAME Reddit IDs "
                    "using old role-order TextBlob vs exact-join TextBlob — isolates join effect"
                ),
            },
        },
        "prior_published_delta_not_pure_join": prior_published,
        "later_author_vader_old_vs_new": later_note,
        "rule": "Quote matched-ID join effect for first_author TextBlob; never quote later_author as join delta",
    }
    write_json(dest / "MATCHED_COHORT_JOIN_EFFECT.json", report)
    return report
