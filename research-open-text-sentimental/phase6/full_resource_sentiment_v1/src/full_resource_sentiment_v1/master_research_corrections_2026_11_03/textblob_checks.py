"""Independent TextBlob interval recompute, length-band, source/period sensitivity."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from full_resource_sentiment_v1.master_research_corrections_2026_11_03 import N_BOOT, TB_BOOT_SEED
from full_resource_sentiment_v1.master_research_corrections_2026_11_03.common import (
    DATASET_JSONL,
    FROZEN_DEF,
    ORIG_OUT,
    ORIG_RESTRICTED,
    OUT_ROOT,
    POST_SCORES,
    load_jsonl,
    sha256_text,
    utcnow,
    write_json,
)


def _percentile_ci(samples: np.ndarray) -> list[float]:
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def independent_stratified_mean_diff(
    group_a: list[float],
    group_b: list[float],
    *,
    n_boot: int = N_BOOT,
    seed: int = TB_BOOT_SEED,
    difference_name: str = "gds_minus_tps",
) -> dict[str, Any]:
    """Separate bootstrap implementation/seed from Phase C's stratified_mean_diff."""
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    if a.size == 0 or b.size == 0:
        return {
            "difference_name": difference_name,
            "n_a": int(a.size),
            "n_b": int(b.size),
            "mean_diff": None,
            "mean_diff_ci95": None,
            "implementation": "corrections_independent_numpy_bootstrap",
            "seed": seed,
            "n_bootstrap": n_boot,
        }
    rng = np.random.default_rng(seed)
    # Vectorized stratified resample of means
    idx_a = rng.integers(0, a.size, size=(n_boot, a.size))
    idx_b = rng.integers(0, b.size, size=(n_boot, b.size))
    mean_diffs = a[idx_a].mean(axis=1) - b[idx_b].mean(axis=1)
    median_diffs = np.median(a[idx_a], axis=1) - np.median(b[idx_b], axis=1)
    # Cliff's δ via searchsorted (O((n_a+n_b) log n_b)).
    # For difference_name GDS−TPS with group_a=GDS, group_b=TPS:
    # δ = P(GDS>TPS) − P(GDS<TPS) = (n_less_TPS_than_GDS − n_more_TPS_than_GDS) / (n_a*n_b).
    # Prior bug used (more−less), reversing the stated GDS−TPS convention.
    b_sorted = np.sort(b)
    less = np.searchsorted(b_sorted, a, side="left").sum()  # # TPS < each GDS
    more = (b.size - np.searchsorted(b_sorted, a, side="right")).sum()  # # TPS > each GDS
    cliffs = float((less - more) / (a.size * b.size))

    return {
        "difference_name": difference_name,
        "implementation": "corrections_independent_numpy_bootstrap",
        "n_a": int(a.size),
        "n_b": int(b.size),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_diff": float(a.mean() - b.mean()),
        "mean_diff_ci95": _percentile_ci(mean_diffs),
        "median_a": float(np.median(a)),
        "median_b": float(np.median(b)),
        "median_diff": float(np.median(a) - np.median(b)),
        "median_diff_ci95": _percentile_ci(median_diffs),
        "cliffs_delta": float(cliffs),
        "cliffs_delta_convention": "P(a>b)-P(a<b) with a=group_a/GDS, b=group_b/TPS",
        "n_bootstrap": n_boot,
        "seed": seed,
        "bootstrap_unit": "thread_stratified",
    }


def _body_band(n_chars: int) -> str:
    if n_chars <= 0:
        return "empty"
    if n_chars < 100:
        return "1_99"
    if n_chars <= 500:
        return "100_500"
    return "501_plus"


def run_textblob_checks() -> dict[str, Any]:
    dest = OUT_ROOT / "textblob_scientific_checks"
    dest.mkdir(parents=True, exist_ok=True)

    tb_rows = {
        str(r["reddit_id"]): r for r in load_jsonl(ORIG_RESTRICTED / "textblob_tps_gds_scores.jsonl")
    }
    texts = {
        str(r["reddit_id"]): r for r in load_jsonl(ORIG_RESTRICTED / "text_thread_manifest_restricted.jsonl")
    }
    post_meta = {str(r["reddit_id"]): r for r in load_jsonl(POST_SCORES)}

    records = []
    hash_mismatches = 0
    for r in load_jsonl(DATASET_JSONL):
        if r.get("final_relevance") != "relevant":
            continue
        if r.get("final_discussion") not in {"TPS", "GDS"}:
            continue
        if r.get("sentiment_definition") != FROZEN_DEF:
            continue
        rid = str(r["reddit_id"])
        tb = tb_rows.get(rid)
        if not tb:
            continue
        doc = texts.get(rid) or {}
        title = doc.get("title") or ""
        body = doc.get("body") or ""
        # Hash check vs frozen capture text used for VADER/TextBlob
        t_ok = tb.get("title_sha256") == sha256_text(title)
        b_ok = tb.get("body_sha256") == sha256_text(body)
        if not (t_ok and b_ok):
            hash_mismatches += 1
        pm = post_meta.get(rid) or {}
        records.append(
            {
                "reddit_id": rid,
                "label": r["final_discussion"],
                "decision_source": r.get("decision_source"),
                "selection_stratum": r.get("selection_stratum"),
                "body_chars": int(tb.get("body_chars") or len(body)),
                "title_chars": int(tb.get("title_chars") or len(title)),
                "body_band": _body_band(int(tb.get("body_chars") or len(body))),
                "post_body_len_band": pm.get("body_len_band"),
                "tb_combined": tb.get("textblob_combined_polarity"),
                "tb_body": tb.get("textblob_body_polarity"),
                "tb_title": tb.get("textblob_title_polarity"),
                "vader_combined": (r.get("sentiment") or {}).get("combined_compound"),
                "hash_match_title_body": bool(t_ok and b_ok),
            }
        )

    gds = [x for x in records if x["label"] == "GDS"]
    tps = [x for x in records if x["label"] == "TPS"]

    def vals(rows: list[dict[str, Any]], key: str) -> list[float]:
        return [float(r[key]) for r in rows if r.get(key) is not None]

    # Independent recompute of primary contrasts
    independent = {
        "textblob_combined_polarity": independent_stratified_mean_diff(
            vals(gds, "tb_combined"), vals(tps, "tb_combined")
        ),
        "textblob_body_polarity": independent_stratified_mean_diff(vals(gds, "tb_body"), vals(tps, "tb_body")),
        "textblob_title_polarity": independent_stratified_mean_diff(
            vals(gds, "tb_title"), vals(tps, "tb_title")
        ),
        "vader_combined_compound_matched_ids": independent_stratified_mean_diff(
            vals(gds, "vader_combined"), vals(tps, "vader_combined"), seed=TB_BOOT_SEED + 1
        ),
    }

    # Compare to Phase C reported point estimates
    phase_c = {}
    pc_path = ORIG_OUT / "phase_c_sentiment_comparison/MATCHED_SENTIMENT_COMPARISON.json"
    if pc_path.exists():
        phase_c = json.loads(Path(pc_path).read_text(encoding="utf-8"))

    # Length-band standardized: within each band, GDS−TPS; also common-ID union of bands with both labels
    band_results = {}
    for band in ("1_99", "100_500", "501_plus"):
        g = [x for x in gds if x["body_band"] == band and x.get("tb_combined") is not None]
        t = [x for x in tps if x["body_band"] == band and x.get("tb_combined") is not None]
        band_results[band] = independent_stratified_mean_diff(vals(g, "tb_combined"), vals(t, "tb_combined"))

    # Common-ID denominators: posts with nonempty body in 100_500+501 (Codex diagnostic used 3120)
    common_body = [x for x in records if x.get("tb_combined") is not None and x["body_chars"] > 0]
    common_est = independent_stratified_mean_diff(
        vals([x for x in common_body if x["label"] == "GDS"], "tb_combined"),
        vals([x for x in common_body if x["label"] == "TPS"], "tb_combined"),
    )

    # Source / period sensitivity on combined polarity
    sensitivity = {}
    for key, getter in (
        ("selection_stratum", lambda r: r.get("selection_stratum")),
        (
            "decision_source_bucket",
            lambda r: (
                "terra"
                if r.get("decision_source") == "gpt-6.1-terra_ai_classification"
                else "existing_or_other"
            ),
        ),
    ):
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in records:
            buckets[str(getter(r))].append(r)
        sensitivity[key] = {}
        for bname, brows in sorted(buckets.items()):
            g = [x for x in brows if x["label"] == "GDS"]
            t = [x for x in brows if x["label"] == "TPS"]
            if len(g) < 5 or len(t) < 20:
                sensitivity[key][bname] = {"n_gds": len(g), "n_tps": len(t), "status": "insufficient_n"}
                continue
            sensitivity[key][bname] = independent_stratified_mean_diff(
                vals(g, "tb_combined"), vals(t, "tb_combined")
            )

    report = {
        "utc": utcnow(),
        "n_records": len(records),
        "n_gds": len(gds),
        "n_tps": len(tps),
        "title_body_hash_mismatches_vs_restricted_manifest": hash_mismatches,
        "scale_warning": (
            "TextBlob polarity and VADER compound are different instruments/scales; "
            "do not subtract raw scores across models as if commensurate."
        ),
        "independent_contrasts_seed": TB_BOOT_SEED,
        "independent_contrasts": independent,
        "phase_c_point_estimates_reference": {
            k: (phase_c.get("contrasts_gds_minus_tps") or {}).get(k, {}).get("mean_diff")
            for k in (
                "textblob_combined_polarity",
                "textblob_body_polarity",
                "textblob_title_polarity",
                "vader_combined_compound_matched_ids",
            )
        },
        "length_band_textblob_combined": band_results,
        "common_nonempty_body_textblob_combined": common_est,
        "sensitivity": sensitivity,
        "no_human_sentiment_reference": True,
    }
    write_json(dest / "TEXTBLOB_CHECKS.json", report)

    with (dest / "tables_textblob_checks.csv").open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(
            handle,
            fieldnames=["contrast", "mean_diff", "ci_low", "ci_high", "n_gds", "n_tps", "seed", "note"],
        )
        w.writeheader()
        for name, est in independent.items():
            ci = est.get("mean_diff_ci95") or [None, None]
            w.writerow(
                {
                    "contrast": name,
                    "mean_diff": est.get("mean_diff"),
                    "ci_low": ci[0],
                    "ci_high": ci[1],
                    "n_gds": est.get("n_a"),
                    "n_tps": est.get("n_b"),
                    "seed": est.get("seed"),
                    "note": "independent_bootstrap",
                }
            )
        for band, est in band_results.items():
            ci = est.get("mean_diff_ci95") or [None, None]
            w.writerow(
                {
                    "contrast": f"tb_combined_band_{band}",
                    "mean_diff": est.get("mean_diff"),
                    "ci_low": ci[0],
                    "ci_high": ci[1],
                    "n_gds": est.get("n_a"),
                    "n_tps": est.get("n_b"),
                    "seed": est.get("seed"),
                    "note": "length_band",
                }
            )

    (dest / "FINDINGS.md").write_text(
        f"""# TextBlob scientific checks

**UTC:** {report['utc']}  
Independent bootstrap seed **{TB_BOOT_SEED}** (≠ Phase C seed). Title/body hash mismatches vs restricted manifest: **{hash_mismatches}**.

## Primary independent GDS−TPS (TextBlob combined)
mean_diff={independent['textblob_combined_polarity'].get('mean_diff')}, CI95={independent['textblob_combined_polarity'].get('mean_diff_ci95')}, n={independent['textblob_combined_polarity'].get('n_a')}/{independent['textblob_combined_polarity'].get('n_b')}

## Length bands (TextBlob combined)
See `tables_textblob_checks.csv`. Within-model only — do not subtract TextBlob from VADER raw scores.

## Status
No human sentiment reference; intervals provisional even when independently recomputed.
""",
        encoding="utf-8",
    )
    (dest / "LIMITATIONS.md").write_text(
        """# TextBlob check limitations

- Independent bootstrap validates computation, not ground-truth sentiment accuracy.
- Length bands are descriptive strata; not causal length adjustment of the Stage-1 form unless stated.
- Source/period buckets inherit mixed label provenance.
""",
        encoding="utf-8",
    )
    return report
