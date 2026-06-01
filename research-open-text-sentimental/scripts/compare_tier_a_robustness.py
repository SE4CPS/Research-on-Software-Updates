#!/usr/bin/env python3
"""
Compare full C1 vs Tier-A robustness (classification + divergence + ablations).

Runs Tier-A pipelines if outputs missing, then writes comparison report.

Usage:
  python3 scripts/compare_tier_a_robustness.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = _REPO_ROOT / "analysis" / "outputs" / "tier_a_robustness"


def _load_json(p: Path) -> dict:
    if p.exists():
        with p.open(encoding="utf-8") as f:
            return json.load(f)
    return {}


def _run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd))
    subprocess.run(cmd, cwd=_REPO_ROOT, check=True)


def main() -> int:
    cv_full = _REPO_ROOT / "evaluation/outputs/paper_cv_full/summary.json"
    cv_tier = _REPO_ROOT / "evaluation/outputs/paper_cv_tier_a/summary.json"
    div_full = _REPO_ROOT / "analysis/outputs/c1_prime_divergence/comparisons_full.csv"
    div_tier = _REPO_ROOT / "analysis/outputs/c1_prime_divergence/comparisons_tier_a.csv"
    abl_full = _REPO_ROOT / "evaluation/outputs/ablations/ablations_v1/ablation_table.csv"
    abl_tier = _REPO_ROOT / "evaluation/outputs/ablations/ablations_tier_a/ablation_table.csv"

    if not cv_tier.exists():
        _run([sys.executable, "scripts/run_evaluation.py", "--run-name", "paper_cv_tier_a", "--tier-a-only"])
    if not div_tier.exists() or not ( _REPO_ROOT / "analysis/outputs/c1_prime_divergence/divergence_report_tier_a.md").exists():
        _run([sys.executable, "scripts/analyze_c1_prime.py", "--tier-a-only"])
    if not abl_tier.exists():
        _run([sys.executable, "scripts/run_ablations.py", "--run-name", "ablations_tier_a", "--tier-a-only"])

    full = _load_json(cv_full)
    tier = _load_json(cv_tier)

    lines = [
        "# Tier-A Robustness Comparison",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        "",
        "Tier-A = software/dev/platform subreddit allowlist (see `evaluation/config.py`).",
        "",
        "## Classification (5-fold CV, macro-F1 / TPS-F1)",
        "",
        "| Model | Full macro-F1 | Tier-A macro-F1 | Full TPS-F1 | Tier-A TPS-F1 |",
        "|-------|---------------|-----------------|-------------|---------------|",
    ]

    for model in ["naive_bayes", "logistic_regression", "vader_rules"]:
        fm = full.get("models", {}).get(model, {}).get("aggregate", {})
        tm = tier.get("models", {}).get(model, {}).get("aggregate", {})
        if fm and tm:
            lines.append(
                f"| {model} | {fm.get('f1_macro_mean', 0):.3f} | {tm.get('f1_macro_mean', 0):.3f} | "
                f"{fm.get('f1_tps_mean', 0):.3f} | {tm.get('f1_tps_mean', 0):.3f} |"
            )

    if div_full.exists() and div_tier.exists():
        df_f = pd.read_csv(div_full)
        df_t = pd.read_csv(div_tier)
        div_row_f = df_f[df_f["metric"] == "sentiment_divergence"].iloc[0]
        div_row_t = df_t[df_t["metric"] == "sentiment_divergence"].iloc[0]
        lines.extend(
            [
                "",
                "## Divergence (TPS − GDS mean |author − community|)",
                "",
                f"- **Full (n≈542):** Δ={div_row_f['mean_diff_tps_minus_gds']:.4f}, MW p={div_row_f['mannwhitney_p']:.4f}",
                f"- **Tier-A (n≈304):** Δ={div_row_t['mean_diff_tps_minus_gds']:.4f}, MW p={div_row_t['mannwhitney_p']:.4f}",
                "",
            ]
        )

    if abl_full.exists() and abl_tier.exists():
        af = pd.read_csv(abl_full)
        at = pd.read_csv(abl_tier)
        lines.extend(["## Ablation macro-F1 (title-only vs full thread)", ""])
        for aid in ["D1_title_only", "D3_full_thread", "D4_full_plus_trajectory"]:
            rf = af[af["ablation_id"] == aid]
            rt = at[at["ablation_id"] == aid]
            if len(rf) and len(rt):
                lines.append(
                    f"- **{aid}:** Full {rf.iloc[0]['macro_f1_mean']:.3f} | Tier-A {rt.iloc[0]['macro_f1_mean']:.3f}"
                )

    lines.extend(
        [
            "",
            "## Interpretation (template)",
            "",
            "- If Tier-A metrics stay in the same ballpark, findings are **not driven only** by Bitcoin/transformers toys.",
            "- If divergence p-value weakens on Tier-A, report both but emphasize direction consistency.",
            "- If title-only still wins on Tier-A, the title signal is **robust** in software subs.",
            "",
        ]
    )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tier_a_robustness_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT / 'tier_a_robustness_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
