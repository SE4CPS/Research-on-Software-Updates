"""Phase C: matched-text TextBlob scoring + Cardiff canary/block; reuse Stage-1 VADER."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from full_resource_sentiment_v1.inference import stratified_mean_diff
from full_resource_sentiment_v1.master_research_2026_11_03.common import (
    DATASET_JSONL,
    FROZEN_DEF,
    OUT_ROOT,
    RESTRICTED_ROOT,
    STAGE1_DIR,
    load_jsonl,
    sha256_file,
    sha256_text,
    utcnow,
    write_json,
    write_jsonl,
)
from full_resource_sentiment_v1.master_research_2026_11_03 import N_BOOT, SEED


def _score_textblob(text: str) -> dict[str, float | None]:
    from textblob import TextBlob

    if not (text or "").strip():
        return {"polarity": None, "subjectivity": None}
    s = TextBlob(text).sentiment
    return {"polarity": float(s.polarity), "subjectivity": float(s.subjectivity)}


def cardiff_cache_status() -> dict[str, Any]:
    hub = Path.home() / ".cache/huggingface/hub/models--cardiffnlp--twitter-roberta-base-sentiment-latest"
    snap = hub / "snapshots/3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7"
    noexist = hub / ".no_exist/3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7"
    present = sorted(p.name for p in snap.iterdir()) if snap.exists() else []
    missing = sorted(p.name for p in noexist.iterdir()) if noexist.exists() else []
    # Torch/transformers not installed in research or master env by policy (large download).
    try:
        import importlib.util

        has_torch = importlib.util.find_spec("torch") is not None
        has_tf = importlib.util.find_spec("transformers") is not None
    except Exception:
        has_torch, has_tf = False, False
    return {
        "refs_main": "3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7",
        "snapshot_files": present,
        "marked_no_exist": missing,
        "pytorch_model_bin_present": "pytorch_model.bin" in present,
        "tokenizer_json_in_snapshot": "tokenizer.json" in present,
        "torch_installed": has_torch,
        "transformers_installed": has_tf,
        "status": "BLOCKED_NO_TORCH_NO_LARGE_DOWNLOAD",
        "note": (
            "Weights partially cached; running Cardiff requires installing torch/transformers "
            "(large download) which is not authorized in this phase. Canary not executed."
        ),
    }


def run_phase_c() -> dict[str, Any]:
    dest = OUT_ROOT / "phase_c_sentiment_comparison"
    dest.mkdir(parents=True, exist_ok=True)
    texts = {
        str(r["reddit_id"]): r for r in load_jsonl(RESTRICTED_ROOT / "text_thread_manifest_restricted.jsonl")
    }
    frame = load_jsonl(DATASET_JSONL)
    eligible = [
        r
        for r in frame
        if r.get("final_relevance") == "relevant"
        and r.get("final_discussion") in {"TPS", "GDS"}
        and r.get("sentiment_definition") == FROZEN_DEF
    ]
    # Score TextBlob on identical title/body
    scored = []
    for row in eligible:
        rid = str(row["reddit_id"])
        doc = texts.get(rid) or {"title": "", "body": ""}
        title, body = doc.get("title") or "", doc.get("body") or ""
        combined = (title + "\n" + body).strip()
        t_tb = _score_textblob(title)
        b_tb = _score_textblob(body) if body.strip() else {"polarity": None, "subjectivity": None}
        c_tb = _score_textblob(combined) if combined else {"polarity": None, "subjectivity": None}
        sent = row.get("sentiment") or {}
        scored.append(
            {
                "reddit_id": rid,
                "final_discussion": row.get("final_discussion"),
                "decision_source": row.get("decision_source"),
                "title_chars": len(title),
                "body_chars": len(body),
                "title_sha256": sha256_text(title),
                "body_sha256": sha256_text(body),
                "textblob_title_polarity": t_tb["polarity"],
                "textblob_title_subjectivity": t_tb["subjectivity"],
                "textblob_body_polarity": b_tb["polarity"],
                "textblob_body_subjectivity": b_tb["subjectivity"],
                "textblob_combined_polarity": c_tb["polarity"],
                "textblob_combined_subjectivity": c_tb["subjectivity"],
                "vader_title_compound": sent.get("title_compound"),
                "vader_body_compound": sent.get("body_compound"),
                "vader_combined_compound": sent.get("combined_compound"),
            }
        )

    write_jsonl(RESTRICTED_ROOT / "textblob_tps_gds_scores.jsonl", scored)
    # Public aggregate only
    tps = [r for r in scored if r["final_discussion"] == "TPS"]
    gds = [r for r in scored if r["final_discussion"] == "GDS"]

    def vals(rows: list[dict[str, Any]], key: str) -> list[float]:
        return [float(r[key]) for r in rows if r.get(key) is not None]

    contrasts = {
        "textblob_combined_polarity": stratified_mean_diff(
            vals(gds, "textblob_combined_polarity"),
            vals(tps, "textblob_combined_polarity"),
            n_boot=N_BOOT,
            seed=SEED,
            difference_name="gds_minus_tps",
        ),
        "textblob_body_polarity": stratified_mean_diff(
            vals(gds, "textblob_body_polarity"),
            vals(tps, "textblob_body_polarity"),
            n_boot=N_BOOT,
            seed=SEED,
            difference_name="gds_minus_tps",
        ),
        "textblob_title_polarity": stratified_mean_diff(
            vals(gds, "textblob_title_polarity"),
            vals(tps, "textblob_title_polarity"),
            n_boot=N_BOOT,
            seed=SEED,
            difference_name="gds_minus_tps",
        ),
        "vader_combined_compound_matched_ids": stratified_mean_diff(
            vals(gds, "vader_combined_compound"),
            vals(tps, "vader_combined_compound"),
            n_boot=N_BOOT,
            seed=SEED,
            difference_name="gds_minus_tps",
        ),
    }

    # Within-model agreement: Spearman of VADER combined vs TextBlob polarity on common IDs
    from full_resource_sentiment_v1.inference import bootstrap_spearman

    both = [
        r
        for r in scored
        if r.get("vader_combined_compound") is not None and r.get("textblob_combined_polarity") is not None
    ]
    spearman = bootstrap_spearman(
        [float(r["vader_combined_compound"]) for r in both],
        [float(r["textblob_combined_polarity"]) for r in both],
        n_boot=N_BOOT,
        seed=SEED,
    )

    cardiff = cardiff_cache_status()
    stage1_ref = {
        "path": str(STAGE1_DIR),
        "findings_exists": (STAGE1_DIR / "FINDINGS.md").exists(),
        "reused_not_overwritten": True,
    }

    from importlib.metadata import version as pkg_version

    report = {
        "utc": utcnow(),
        "population": "frozen relevant TPS/GDS (matched IDs for TextBlob+VADER)",
        "n_tps": len(tps),
        "n_gds": len(gds),
        "textblob_version": pkg_version("textblob"),
        "contrasts_gds_minus_tps": contrasts,
        "vader_textblob_spearman_combined": spearman,
        "cardiff": cardiff,
        "stage1_vader_reuse": stage1_ref,
        "scale_note": "Do not subtract TextBlob polarity from VADER compound as if same scale.",
        "accuracy_claim": "No human sentiment reference; model disagreement ≠ which is correct.",
    }
    write_json(dest / "MATCHED_SENTIMENT_COMPARISON.json", report)
    # Compact CSV of contrasts
    import csv

    with (dest / "tables_gds_minus_tps.csv").open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(
            handle,
            fieldnames=[
                "measure",
                "mean_diff",
                "ci_low",
                "ci_high",
                "n_gds",
                "n_tps",
                "cliffs_delta",
            ],
        )
        w.writeheader()
        for name, stats in contrasts.items():
            ci = stats.get("mean_diff_ci95") or [None, None]
            w.writerow(
                {
                    "measure": name,
                    "mean_diff": stats.get("mean_diff"),
                    "ci_low": ci[0],
                    "ci_high": ci[1],
                    "n_gds": stats.get("n_a"),
                    "n_tps": stats.get("n_b"),
                    "cliffs_delta": stats.get("cliffs_delta"),
                }
            )

    def fmt(s: dict[str, Any] | None) -> str:
        if not s or s.get("mean_diff") is None:
            return "n/a"
        ci = s.get("mean_diff_ci95") or [None, None]
        return f"{s['mean_diff']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] (n={s.get('n_a')}/{s.get('n_b')})"

    (dest / "FINDINGS.md").write_text(
        f"""# Phase C findings — matched-text sentiment comparison

**UTC:** {report['utc']}  
Population: frozen relevant TPS **{len(tps)}** / GDS **{len(gds)}** (identical IDs).

## TextBlob 0.19.0 PatternAnalyzer (polarity)
- Combined polarity GDS−TPS: {fmt(contrasts['textblob_combined_polarity'])}
- Body polarity GDS−TPS: {fmt(contrasts['textblob_body_polarity'])}
- Title polarity GDS−TPS: {fmt(contrasts['textblob_title_polarity'])}

## VADER on same IDs (reuse Stage 1 scores)
- Combined compound GDS−TPS: {fmt(contrasts['vader_combined_compound_matched_ids'])}
- Spearman(VADER combined, TextBlob polarity): {spearman.get('spearman')} CI {spearman.get('spearman_ci95')}

## Cardiff
**Blocked:** {cardiff['status']} — {cardiff['note']}

## Scientific note
Accepted Stage 1 VADER results are reused, not overwritten. Cross-model raw-score subtraction is not performed.
""",
        encoding="utf-8",
    )
    (dest / "LIMITATIONS.md").write_text(
        """# Phase C limitations

- TextBlob and VADER use different scales; within-model contrasts only for magnitude claims.
- Cardiff not scored (no torch/transformers install; large download not authorized).
- No human sentiment labels → cannot rank tool accuracy.
- Title/body length sensitivity for TextBlob not fully band-standardized in this phase (see Stage 1 for VADER length adjustment).
""",
        encoding="utf-8",
    )
    write_json(dest / "ENVIRONMENT.json", {"textblob": report["textblob_version"], "cardiff": cardiff})
    return report
