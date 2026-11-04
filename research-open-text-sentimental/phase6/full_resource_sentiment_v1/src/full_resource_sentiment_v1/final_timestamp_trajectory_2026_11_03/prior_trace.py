"""Read-only prior A/B experiment trace (no retraining)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import sha256_file
from . import paths as P


def _exists_hash(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def run_prior_trace(out: Path) -> dict[str, Any]:
    root = P.ROOT  # phase6/full_resource_sentiment_v1
    repo = root.parents[1]  # research-open-text-sentimental
    candidates = {
        "c1_prime_metrics": [
            repo / "data" / "c1_prime" / "c1_prime_metrics.csv",
        ],
        "c1_prime_summary": [
            repo / "analysis" / "outputs" / "c1_prime_divergence" / "summary.json",
        ],
        "b_fresh_results": [
            repo / "phase3" / "fresh_test_20261007" / "RESULTS_FIXED_MODEL.md",
        ],
        "b_diagnostic": [
            repo / "phase3" / "diagnostic_results" / "analysis.json",
        ],
        "b_sentistrength_se": [
            repo
            / "phase5"
            / "final_improvement"
            / "results"
            / "sentistrength_se_predictions.jsonl",
        ],
        "four_model_rq_table": [
            P.RESULTS
            / "pre_manuscript_evidence_reconciliation_2026-11-03"
            / "tables"
            / "FOUR_MODEL_MATCHED_RQ1_RQ2.json",
        ],
    }

    found: dict[str, Any] = {}
    for key, paths in candidates.items():
        hit = None
        for p in paths:
            info = _exists_hash(p)
            if info["exists"]:
                hit = info
                break
        found[key] = hit or {"exists": False, "tried": [str(p) for p in paths]}

    details: dict[str, Any] = {"artifacts": found}
    notes: list[str] = []

    # Historical 542 divergence if present
    summ = found.get("c1_prime_summary") or {}
    if summ.get("exists"):
        try:
            s = json.loads(Path(summ["path"]).read_text(encoding="utf-8"))
            details["c1_prime_summary_keys"] = list(s.keys())[:40]
            details["c1_prime_summary_excerpt"] = {k: s[k] for k in list(s.keys())[:12]}
            notes.append(
                "Historical C1-prime summary.json located; reuse as historical comparison only "
                "(broad-GDS / title-fallback caveats). Do not place beside strict-GDS p-values as replication."
            )
        except Exception as e:
            notes.append(f"C1-prime summary unreadable: {e}")
    else:
        notes.append("Historical 542 C1-prime summary.json not found — trace incomplete.")

    metrics = found.get("c1_prime_metrics") or {}
    if metrics.get("exists"):
        n = sum(1 for _ in open(metrics["path"], encoding="utf-8")) - 1
        details["c1_prime_metrics_n_rows"] = n
        notes.append(f"c1_prime_metrics.csv data rows={n} (expected ~542).")
        if n != 542:
            notes.append(f"GAP: c1_prime_metrics row count {n} ≠ 542.")
    else:
        notes.append("GAP: c1_prime_metrics.csv not found.")

    fresh = found.get("b_fresh_results") or {}
    if fresh.get("exists"):
        notes.append("B fresh RESULTS_FIXED_MODEL.md found; text-only purpose-classification task, not strict-GDS accuracy.")
    else:
        notes.append("GAP: B fresh RESULTS_FIXED_MODEL.md not found.")

    diag = found.get("b_diagnostic") or {}
    if diag.get("exists"):
        try:
            d = json.loads(Path(diag["path"]).read_text(encoding="utf-8"))
            details["b_diagnostic_top_keys"] = list(d.keys())[:30]
            # common keys for n
            for key in ("n", "n_confirmed", "n_posts", "denominator", "n_sample"):
                if key in d:
                    details[f"b_diagnostic_{key}"] = d[key]
            notes.append(
                "B diagnostic analysis.json found; treat 237 confirmed / 250 sample as separately scoped ancillary evidence."
            )
        except Exception as e:
            notes.append(f"B diagnostic unreadable: {e}")
    else:
        notes.append("GAP: B diagnostic analysis.json not found.")

    se = found.get("b_sentistrength_se") or {}
    if se.get("exists"):
        n = sum(1 for _ in open(se["path"], encoding="utf-8") if _.strip())
        details["sentistrength_se_n"] = n
        if n != 2050:
            notes.append(f"GAP: SentiStrength-SE predictions n={n} (expected 2050).")
        else:
            notes.append(
                "SentiStrength-SE predictions n=2050 matched; B-task only — no current TPS/GDS post/reply scores."
            )
    else:
        notes.append("GAP: SentiStrength-SE predictions.jsonl not found.")

    fm = found.get("four_model_rq_table") or {}
    if fm.get("exists"):
        notes.append("Current four-model matched RQ table present; do not rerun scoring.")
    else:
        notes.append("GAP: FOUR_MODEL_MATCHED_RQ1_RQ2.json missing — unexpected.")

    # Overlap note: historical 542 vs preserved 6425 — ID overlap check if metrics have reddit ids
    details["overlap_policy"] = (
        "Old 542 overlap the preserved 6,425 broad-GDS universe; they are not an extra independent sample. "
        "520 historical-only union rows lack broad timestamps and stay outside the trajectory frame."
    )

    md_lines = [
        "# PRIOR EXPERIMENT TRACE",
        "",
        "Read-only verification against saved A/B artifacts. No retraining. No manuscript claims finalized here.",
        "",
        "## Artifact presence",
        "",
    ]
    for k, v in found.items():
        if v.get("exists"):
            md_lines.append(f"- **{k}**: found `{v['path']}` sha256=`{v['sha256'][:16]}…`")
        else:
            md_lines.append(f"- **{k}**: NOT FOUND (tried {v.get('tried')})")
    md_lines += ["", "## Notes", ""]
    md_lines.extend(f"- {n}" for n in notes)
    md_lines += ["", "## Policy", "", details["overlap_policy"], ""]

    return {
        "details": details,
        "notes": notes,
        "markdown": "\n".join(md_lines),
        "status": "TRACE_ONLY_AWAITING_CODEX",
    }
