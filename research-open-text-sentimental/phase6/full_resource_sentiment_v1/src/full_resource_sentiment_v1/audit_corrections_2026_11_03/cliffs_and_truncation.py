"""Items 1–2: corrected Cliff's delta tables; stratum-matched Cardiff truncation."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np

from full_resource_sentiment_v1.audit_corrections_2026_11_03 import N_BOOT, SEED
from full_resource_sentiment_v1.audit_corrections_2026_11_03.common import (
    CARDIFF_SCORES,
    DATASET_JSONL,
    FROZEN_DEF,
    OUT_ROOT,
    SEALED_CORR,
    UNION_V2,
    load_jsonl,
    utcnow,
    write_json,
)
from full_resource_sentiment_v1.master_research_corrections_2026_11_03.textblob_checks import (
    independent_stratified_mean_diff,
)


def _sign(x: float | None, thr: float = 0.0) -> int | None:
    if x is None:
        return None
    if abs(float(x)) < thr:
        return 0
    return int(np.sign(float(x)))


def _agree(rows: list[dict[str, Any]], thr: float = 0.0) -> dict[str, Any]:
    n = cv = ct = vt = 0
    for r in rows:
        if r.get("vader") is None or r.get("textblob") is None or r.get("cardiff_polarity_sign") is None:
            continue
        n += 1
        cs = _sign(r["cardiff_polarity_sign"], 0.0)
        vs = _sign(r["vader"], thr)
        ts = _sign(r["textblob"], thr)
        if cs == vs:
            cv += 1
        if cs == ts:
            ct += 1
        if vs == ts:
            vt += 1
    denom = max(n, 1)
    return {
        "n_complete_scores": n,
        "cardiff_vs_vader": cv / denom,
        "cardiff_vs_textblob": ct / denom,
        "vader_vs_textblob": vt / denom,
        "neutral_threshold": thr,
    }


def run_cliffs_regeneration() -> dict[str, Any]:
    dest = OUT_ROOT / "item1_cliffs_delta_correction"
    dest.mkdir(parents=True, exist_ok=True)
    import json as _json

    prior_rq = _json.loads((SEALED_CORR / "stage3_rq1_rq3/RQ1_RQ3_RESULTS.json").read_text(encoding="utf-8"))

    # Rebuild contrasts from Cardiff scores + dataset for identical means, corrected cliffs
    cardiff = load_jsonl(CARDIFF_SCORES)
    frozen_posts = {
        r["reddit_id"]: r
        for r in cardiff
        if r.get("stratum") == "original_6425_frozen_comparable" and r.get("unit") == "original_post"
    }
    ai_posts = {
        r["reddit_id"]: r for r in cardiff if r.get("stratum") == "ai_assisted_v2" and r.get("unit") == "original_post"
    }

    def contrast_posts(posts: dict[str, dict], labels: dict[str, str], seed: int):
        g_v, t_v, g_t, t_t, g_c, t_c = [], [], [], [], [], []
        for rid, lab in labels.items():
            p = posts.get(rid)
            if not p:
                continue
            bucket = (g_v, g_t, g_c) if lab == "GDS" else (t_v, t_t, t_c)
            if p.get("vader") is not None:
                bucket[0].append(float(p["vader"]))
            if p.get("textblob") is not None:
                bucket[1].append(float(p["textblob"]))
            if p.get("cardiff_polarity_sign") is not None:
                bucket[2].append(float(p["cardiff_polarity_sign"]))
        return {
            "vader": independent_stratified_mean_diff(g_v, t_v, seed=seed, n_boot=N_BOOT),
            "textblob": independent_stratified_mean_diff(g_t, t_t, seed=seed + 1, n_boot=N_BOOT),
            "cardiff_sign": independent_stratified_mean_diff(g_c, t_c, seed=seed + 2, n_boot=N_BOOT),
        }

    frozen_labels = {
        str(r["reddit_id"]): r["final_discussion"]
        for r in load_jsonl(DATASET_JSONL)
        if r.get("final_relevance") == "relevant"
        and r.get("final_discussion") in {"TPS", "GDS"}
        and r.get("sentiment_definition") == FROZEN_DEF
    }
    ai_labels = {
        r["reddit_id"]: r["final_discussion"]
        for r in load_jsonl(UNION_V2)
        if r.get("record_origin") == "ai_assisted_screen_candidate_v2"
        and r.get("final_discussion") in {"TPS", "GDS"}
        and not r.get("exact_dup_with_train")
    }

    corrected = {
        "utc": utcnow(),
        "cliffs_fix": (
            "Cliff's δ now P(GDS>TPS)−P(GDS<TPS); prior sealed reports used reversed sign. "
            "Means/CIs unchanged in definition."
        ),
        "frozen_original_posts": contrast_posts(frozen_posts, frozen_labels, SEED + 100),
        "ai_assisted_v2_excl_exact_train_dups": contrast_posts(ai_posts, ai_labels, SEED + 200),
        "prior_sealed_cliffs_examples": {
            "ai_vader_was": ((prior_rq.get("rq1_ai_assisted_v2_excl_exact_train_dups") or {}).get("rq1_post_vader") or {}).get(
                "cliffs_delta"
            ),
            "ai_vader_should_be_negated": True,
            "note": "Compare corrected.ai_assisted...vader.cliffs_delta ≈ +0.2137",
        },
        "preserves_sealed_tree": str(SEALED_CORR),
    }
    write_json(dest / "CORRECTED_CLIFFS_RQ1_POST.json", corrected)
    return corrected


def run_truncation_correction() -> dict[str, Any]:
    dest = OUT_ROOT / "item2_cardiff_truncation_correction"
    dest.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(CARDIFF_SCORES)

    def filt(pred):
        return [r for r in rows if pred(r)]

    frozen = filt(lambda r: r.get("stratum") == "original_6425_frozen_comparable")
    ai = filt(lambda r: r.get("stratum") == "ai_assisted_v2")
    frozen_complete = [r for r in frozen if r.get("vader") is not None and r.get("textblob") is not None]
    ai_complete = [r for r in ai if r.get("vader") is not None and r.get("textblob") is not None]

    by_unit = {}
    for unit in sorted({r.get("unit") for r in frozen_complete}):
        urows = [r for r in frozen_complete if r.get("unit") == unit]
        by_unit[unit] = {
            "truncated": _agree([r for r in urows if r.get("truncated")]),
            "nontruncated": _agree([r for r in urows if not r.get("truncated")]),
            "n_truncated": sum(1 for r in urows if r.get("truncated")),
            "n_nontruncated": sum(1 for r in urows if not r.get("truncated")),
        }

    # Purpose-specific truncated counts for original_post
    purpose_trunc = {"frozen": Counter(), "ai": Counter()}
    for r in frozen:
        if r.get("unit") == "original_post":
            key = f"{r.get('label')}_{'trunc' if r.get('truncated') else 'nontrunc'}"
            purpose_trunc["frozen"][key] += 1
    for r in ai:
        if r.get("unit") == "original_post":
            key = f"{r.get('label')}_{'trunc' if r.get('truncated') else 'nontrunc'}"
            purpose_trunc["ai"][key] += 1

    # Length-band matched sensitivity within frozen original_post
    posts = [r for r in frozen_complete if r.get("unit") == "original_post"]
    bands = [(0, 500), (500, 1500), (1500, 5000), (5000, 10**9)]
    length_matched = []
    for lo, hi in bands:
        band_rows = [r for r in posts if lo <= int(r.get("text_chars") or 0) < hi]
        length_matched.append(
            {
                "chars": f"[{lo},{hi})",
                "truncated": _agree([r for r in band_rows if r.get("truncated")]),
                "nontruncated": _agree([r for r in band_rows if not r.get("truncated")]),
            }
        )

    report = {
        "utc": utcnow(),
        "original_scores_preserved": True,
        "cardiff_scores_path": str(CARDIFF_SCORES),
        "bug_fixed": (
            "Prior truncated_only n=956 mixed frozen+AI; nontruncated n=6273 was frozen-only. "
            "Now both arms share stratum and eligibility."
        ),
        "frozen_all_units_complete_scores": {
            "truncated": _agree([r for r in frozen_complete if r.get("truncated")]),
            "nontruncated": _agree([r for r in frozen_complete if not r.get("truncated")]),
            "all": _agree(frozen_complete),
            "n_truncated_rows_any_scores": sum(1 for r in frozen if r.get("truncated")),
            "n_ai_truncated_rows_any_scores": sum(1 for r in ai if r.get("truncated")),
        },
        "ai_posts_complete_scores": {
            "truncated": _agree([r for r in ai_complete if r.get("truncated")]),
            "nontruncated": _agree([r for r in ai_complete if not r.get("truncated")]),
            "all": _agree(ai_complete),
        },
        "frozen_by_unit_complete_scores": by_unit,
        "purpose_specific_original_post_truncation_counts": {
            "frozen": dict(purpose_trunc["frozen"]),
            "ai": dict(purpose_trunc["ai"]),
        },
        "frozen_original_post_length_band_matched": length_matched,
        "rq1_nontruncated_sensitivity_frozen_posts": _agree(
            [r for r in posts if not r.get("truncated")]
        ),
        "rq1_truncated_frozen_posts": _agree([r for r in posts if r.get("truncated")]),
        "note": "Agreement ≠ accuracy; truncation correlates with length/unit type",
    }
    write_json(dest / "CARDIFF_TRUNCATION_CORRECTED.json", report)
    return report
