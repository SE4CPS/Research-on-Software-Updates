"""Item 4: length-band / regression-adjusted RQ1 + source/date/origin sensitivities."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from full_resource_sentiment_v1.audit_corrections_2026_11_03 import N_BOOT, SEED
from full_resource_sentiment_v1.audit_corrections_2026_11_03.common import (
    CARDIFF_SCORES,
    DATASET_JSONL,
    FROZEN_DEF,
    OUT_ROOT,
    POST_SCORES,
    UNION_V2,
    load_jsonl,
    utcnow,
    write_json,
)
from full_resource_sentiment_v1.master_research_corrections_2026_11_03.textblob_checks import (
    independent_stratified_mean_diff,
)


def _band(n: int) -> str:
    if n < 100:
        return "0_99"
    if n < 500:
        return "100_499"
    if n < 1500:
        return "500_1499"
    if n < 4000:
        return "1500_3999"
    return "4000_plus"


def _ols_adjusted(y: np.ndarray, gds: np.ndarray, log_len: np.ndarray) -> dict[str, Any]:
    """y ~ gds + log_len; report GDS coefficient (adjusted mean difference)."""
    X = np.column_stack([np.ones(len(y)), gds.astype(float), log_len])
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return {"gds_coef": None, "ok": False}
    return {
        "gds_coef_adjusted": float(beta[1]),
        "intercept": float(beta[0]),
        "log_len_coef": float(beta[2]),
        "ok": True,
        "n": int(len(y)),
        "n_gds": int(gds.sum()),
        "n_tps": int((1 - gds).sum()),
    }


def _boot_adjusted(y, gds, log_len, seed, n_boot=N_BOOT):
    rng = np.random.default_rng(seed)
    coefs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        res = _ols_adjusted(y[idx], gds[idx], log_len[idx])
        if res.get("ok") and res.get("gds_coef_adjusted") is not None:
            coefs.append(res["gds_coef_adjusted"])
    if not coefs:
        return None
    a = np.asarray(coefs)
    point = _ols_adjusted(y, gds, log_len)
    return {
        **point,
        "gds_coef_ci95": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))],
        "n_bootstrap": n_boot,
        "seed": seed,
    }


def run_rq1_adjusted() -> dict[str, Any]:
    dest = OUT_ROOT / "item4_rq1_length_source_sensitivity"
    dest.mkdir(parents=True, exist_ok=True)
    post_meta = {str(r["reddit_id"]): r for r in load_jsonl(POST_SCORES)}

    def build_rows(stratum: str) -> list[dict[str, Any]]:
        rows = []
        if stratum == "frozen":
            labels = {
                str(r["reddit_id"]): {
                    "label": r["final_discussion"],
                    "decision_source": r.get("decision_source"),
                }
                for r in load_jsonl(DATASET_JSONL)
                if r.get("final_relevance") == "relevant"
                and r.get("final_discussion") in {"TPS", "GDS"}
                and r.get("sentiment_definition") == FROZEN_DEF
            }
            scores = {
                r["reddit_id"]: r
                for r in load_jsonl(CARDIFF_SCORES)
                if r.get("stratum") == "original_6425_frozen_comparable" and r.get("unit") == "original_post"
            }
        else:
            labels = {
                r["reddit_id"]: {
                    "label": r["final_discussion"],
                    "decision_source": r.get("decision_source"),
                }
                for r in load_jsonl(UNION_V2)
                if r.get("record_origin") == "ai_assisted_screen_candidate_v2"
                and r.get("final_discussion") in {"TPS", "GDS"}
                and not r.get("exact_dup_with_train")
            }
            scores = {
                r["reddit_id"]: r
                for r in load_jsonl(CARDIFF_SCORES)
                if r.get("stratum") == "ai_assisted_v2" and r.get("unit") == "original_post"
            }
        for rid, meta in labels.items():
            sc = scores.get(rid)
            if not sc:
                continue
            pm = post_meta.get(rid) or {}
            chars = int(sc.get("text_chars") or 0)
            rows.append(
                {
                    "reddit_id": rid,
                    "label": meta["label"],
                    "decision_source": meta.get("decision_source"),
                    "vader": sc.get("vader"),
                    "textblob": sc.get("textblob"),
                    "cardiff_sign": sc.get("cardiff_polarity_sign"),
                    "text_chars": chars,
                    "log_len": float(np.log1p(chars)),
                    "length_band": _band(chars),
                    "source": pm.get("source"),
                    "created_month": pm.get("created_month"),
                    "date_band": pm.get("date_band"),
                    "truncated": sc.get("truncated"),
                }
            )
        return rows

    def analyze(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
        out: dict[str, Any] = {"n": len(rows)}
        for inst in ("vader", "textblob", "cardiff_sign"):
            g = [float(r[inst]) for r in rows if r["label"] == "GDS" and r.get(inst) is not None]
            t = [float(r[inst]) for r in rows if r["label"] == "TPS" and r.get(inst) is not None]
            unadj = independent_stratified_mean_diff(g, t, seed=seed, n_boot=N_BOOT)
            y = np.asarray(
                [float(r[inst]) for r in rows if r.get(inst) is not None], dtype=float
            )
            gds = np.asarray(
                [1.0 if r["label"] == "GDS" else 0.0 for r in rows if r.get(inst) is not None],
                dtype=float,
            )
            log_len = np.asarray(
                [float(r["log_len"]) for r in rows if r.get(inst) is not None], dtype=float
            )
            # Align y/gds/log_len to same filter order
            filt = [r for r in rows if r.get(inst) is not None]
            y = np.asarray([float(r[inst]) for r in filt], dtype=float)
            gds = np.asarray([1.0 if r["label"] == "GDS" else 0.0 for r in filt], dtype=float)
            log_len = np.asarray([float(r["log_len"]) for r in filt], dtype=float)
            adj = _boot_adjusted(y, gds, log_len, seed + 10)
            # Length-band standardized: mean within-band GDS−TPS, then unweighted mean of bands with both groups
            band_diffs = []
            band_detail = {}
            by_band = defaultdict(list)
            for r in filt:
                by_band[r["length_band"]].append(r)
            for band, br in sorted(by_band.items()):
                gg = [float(x[inst]) for x in br if x["label"] == "GDS"]
                tt = [float(x[inst]) for x in br if x["label"] == "TPS"]
                band_detail[band] = {"n_gds": len(gg), "n_tps": len(tt)}
                if gg and tt:
                    d = float(np.mean(gg) - np.mean(tt))
                    band_diffs.append(d)
                    band_detail[band]["mean_diff"] = d
            out[inst] = {
                "unadjusted": unadj,
                "regression_adjusted_log_length": adj,
                "length_band_standardized_mean_of_band_diffs": (
                    float(np.mean(band_diffs)) if band_diffs else None
                ),
                "n_bands_with_both_groups": len(band_diffs),
                "band_detail": band_detail,
                "common_support_note": "Bands without both GDS and TPS excluded from standardized mean",
            }
            seed += 20
        # Source / date / origin sensitivities (VADER primary)
        sens = {}
        for field in ("source", "date_band", "decision_source"):
            by = defaultdict(list)
            for r in rows:
                by[str(r.get(field) or "unknown")].append(r)
            sens[field] = {}
            for k, br in sorted(by.items(), key=lambda kv: -len(kv[1])):
                if len(br) < 30:
                    continue
                g = [float(r["vader"]) for r in br if r["label"] == "GDS" and r.get("vader") is not None]
                t = [float(r["vader"]) for r in br if r["label"] == "TPS" and r.get("vader") is not None]
                if len(g) < 5 or len(t) < 5:
                    continue
                sens[field][k] = independent_stratified_mean_diff(g, t, seed=seed, n_boot=min(N_BOOT, 5000))
                seed += 1
        out["vader_sensitivities"] = sens
        out["nontruncated_only_vader"] = independent_stratified_mean_diff(
            [float(r["vader"]) for r in rows if r["label"] == "GDS" and r.get("vader") is not None and not r.get("truncated")],
            [float(r["vader"]) for r in rows if r["label"] == "TPS" and r.get("vader") is not None and not r.get("truncated")],
            seed=seed + 50,
            n_boot=N_BOOT,
        )
        return out

    frozen_rows = build_rows("frozen")
    ai_rows = build_rows("ai")
    report = {
        "utc": utcnow(),
        "labels_unchanged": True,
        "frozen": analyze(frozen_rows, SEED + 500),
        "ai_assisted_v2_excl_exact_train_dups": analyze(ai_rows, SEED + 600),
        "note": (
            "Adjusted estimates are descriptive sensitivity; AI selection may still induce bias. "
            "Not human-confirmed purpose accuracy."
        ),
    }
    write_json(dest / "RQ1_ADJUSTED_SENSITIVITY.json", report)
    return report
