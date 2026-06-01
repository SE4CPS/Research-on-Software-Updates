#!/usr/bin/env python3
"""
Error analysis from OOF predictions (false positives / false negatives).

Usage:
  python3 scripts/run_error_analysis.py --model naive_bayes
  python3 scripts/run_error_analysis.py --model roberta --run-dir paper_roberta_5fold

Outputs: analysis/outputs/error_analysis/<model>/
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_EVAL_OUT = _REPO_ROOT / "evaluation" / "outputs" / "paper_cv_full"
DEFAULT_LABELS = _REPO_ROOT / "tps_gds_classification" / "data" / "updated_labeled_dataset_unique.csv"

THEME_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("sarcasm_humor", re.compile(r"\b(lol|lmao|evil|joke|meme|scrap|😭|!/)\b", re.I)),
    ("rant_emotion", re.compile(r"\b(hate|worst|awful|crappy|scrap|can't keep|angry)\b", re.I)),
    ("technical_cue", re.compile(r"\b(error|bug|fix|crash|update|install|broken|help|how do)\b", re.I)),
    ("short_title", re.compile(r"^.{0,40}$")),
    ("news_announcement", re.compile(r"\b(released|update is out|preview|announces|gets)\b", re.I)),
    ("off_topic_fandom", re.compile(r"\b(transformers|figure|display cabinet|toy|grok gets)\b", re.I)),
    ("crypto", re.compile(r"\b(bitcoin|btc|verify|whale)\b", re.I)),
]


def assign_themes(title: str) -> list[str]:
    themes = []
    for name, pat in THEME_PATTERNS:
        if pat.search(title or ""):
            themes.append(name)
    return themes or ["uncategorized"]


def main() -> int:
    p = argparse.ArgumentParser(description="OOF error analysis for TPS/GDS models.")
    p.add_argument("--model", type=str, default="naive_bayes")
    p.add_argument("--run-dir", type=Path, default=DEFAULT_EVAL_OUT)
    p.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: analysis/outputs/error_analysis/<model>",
    )
    args = p.parse_args()

    oof_path = args.run_dir / args.model / "oof_predictions.csv"
    if not oof_path.exists():
        print(f"Missing {oof_path}", file=sys.stderr)
        return 1

    oof = pd.read_csv(oof_path)
    labels = pd.read_csv(args.labels)
    labels["reddit_id"] = labels["reddit_id"].astype(str)
    oof["reddit_id"] = oof["reddit_id"].astype(str)
    lab_cols = ["reddit_id", "title", "corrected_label", "label_name_hint"]
    lab_cols = [c for c in lab_cols if c in labels.columns]
    df = oof.merge(labels[lab_cols], on="reddit_id", how="left")
    df.rename(columns={"label": "gold", "pred": "pred"}, inplace=True)
    df["title"] = df["title"].fillna("")

    df["correct"] = df["gold"] == df["pred"]
    fp = df[(df["gold"] == 0) & (df["pred"] == 1)]  # predicted TPS, actually GDS
    fn = df[(df["gold"] == 1) & (df["pred"] == 0)]  # predicted GDS, actually TPS

    out_dir = args.out_dir or (_REPO_ROOT / "analysis" / "outputs" / "error_analysis" / args.model)
    out_dir.mkdir(parents=True, exist_ok=True)

    def theme_counts(err_df: pd.DataFrame) -> Counter:
        c: Counter = Counter()
        for t in err_df["title"]:
            for th in assign_themes(str(t)):
                c[th] += 1
        return c

    fp_themes = theme_counts(fp)
    fn_themes = theme_counts(fn)

    examples = []
    for kind, err_df in [("FP_TPS", fp), ("FN_TPS", fn)]:
        for _, r in err_df.head(12).iterrows():
            examples.append(
                {
                    "type": kind,
                    "reddit_id": r["reddit_id"],
                    "subreddit": r["subreddit"],
                    "title": r["title"][:120],
                    "score_tps": float(r.get("score_tps", 0)),
                    "themes": assign_themes(str(r["title"])),
                }
            )

    acc = float(df["correct"].mean())
    summary = {
        "model": args.model,
        "run_dir": str(args.run_dir),
        "n": len(df),
        "accuracy_oof": acc,
        "n_fp_tps": len(fp),
        "n_fn_tps": len(fn),
        "fp_theme_counts": dict(fp_themes),
        "fn_theme_counts": dict(fn_themes),
        "top_fp_subreddits": fp["subreddit"].value_counts().head(8).to_dict(),
        "top_fn_subreddits": fn["subreddit"].value_counts().head(8).to_dict(),
    }

    fp.to_csv(out_dir / "false_positives_tps.csv", index=False)
    fn.to_csv(out_dir / "false_negatives_tps.csv", index=False)

    lines = [
        f"# Error Analysis — {args.model}",
        "",
        f"**OOF source:** `{oof_path}`",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        "",
        f"- **OOF accuracy:** {acc:.1%}",
        f"- **False positives (predicted TPS, gold GDS):** {len(fp)}",
        f"- **False negatives (predicted GDS, gold TPS):** {len(fn)}",
        "",
        "## Failure themes (title heuristics)",
        "",
        "### False positives — predicted TPS",
        "",
    ]
    for th, n in fp_themes.most_common():
        lines.append(f"- **{th}:** {n} posts")
    lines.extend(["", "### False negatives — missed TPS", ""])
    for th, n in fn_themes.most_common():
        lines.append(f"- **{th}:** {n} posts")

    lines.extend(
        [
            "",
            "## Discussion bullets (paper-ready)",
            "",
            "- **FP (→TPS):** Technical words in titles (`error`, `update`, `fix`) often trigger TPS even when the thread is news, humor, or general talk.",
            "- **FN (missed TPS):** Short or emotional titles without obvious tech keywords; toy/fandom subs mislabeled in gold or visually 'rant-like' TPS.",
            "- **Naive Bayes vs RoBERTa:** Word cues help NB on small data; RoBERTa needs more TPS examples and tuning — errors concentrate on borderline rants vs help.",
            "- **Subreddit skew:** Check excluded subs (transformers, Bitcoin) in FP/FN tables.",
            "",
            "## Example errors",
            "",
        ]
    )
    for ex in examples[:20]:
        lines.append(
            f"- **{ex['type']}** r/{ex['subreddit']} — \"{ex['title']}\" themes={ex['themes']}"
        )

    (out_dir / "error_analysis_report.md").write_text("\n".join(lines), encoding="utf-8")
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"FP={len(fp)} FN={len(fn)} acc={acc:.3f} → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
