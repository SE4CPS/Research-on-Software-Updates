#!/usr/bin/env python3
"""
VADER rule-based baseline for TPS vs GDS (verified labels).

Uses VADER compound on `text_raw` (title + body + comments).

Two rule sets (see --rule-style):
- emotion_first: neutral band + technical cues → TPS; strong |compound| → GDS (matches the
  written plan; often collapses to “all GDS” because Reddit text is rarely “neutral” for VADER).
- technical_first (default): technical-cue regex → TPS first; then emotion / fallback (usable baseline).

Same undersampling, stratified 70/15/15 split, and defaults as train_naive_bayes.py.

Run from repo root:
  python tps_gds_classification/scripts/train_vader_baseline.py
  python tps_gds_classification/scripts/train_vader_baseline.py --rule-style emotion_first
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from tps_gds_data import (
    load_verified_frame,
    stratified_train_val_test,
    undersample_gds,
    write_unique_labels_csv,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_LABELS = _PROJECT_ROOT / "data" / "updated_labeled_dataset.csv"
_DEFAULT_JSON = _PROJECT_ROOT / "data" / "tps_gds_dataset.json"
_UNIQUE_LABELS_OUT = _PROJECT_ROOT / "data" / "updated_labeled_dataset_unique.csv"
_DEFAULT_OUT = _PROJECT_ROOT / "outputs" / "vader_baseline"

# Problem-solving / technical cues (word boundaries; case-insensitive).
_TECH_CUES = re.compile(
    r"\b("
    r"error|bug|bugs|fix|fixed|fixing|crash|crashed|crashing|failed|failing|failure|"
    r"exception|traceback|stack\s*trace|compile|compiling|build|building|install|"
    r"installation|uninstall|patch|patches|regression|reproduce|repro|workaround|"
    r"broken|not\s+working|doesn'?t\s+work|won'?t\s+work|does\s+not\s+work|"
    r"issue|issues|problem|problems|solve|solved|solution|debug|debugging|"
    r"version|update\s+broke|after\s+update|after\s+updating|upgrade|downgrade|"
    r"dependency|dependencies|config|configuration|logs?|terminal|command|kernel|"
    r"docker|container|npm|pip|gradle|cmake|build\s+error|runtime\s+error|"
    r"how\s+do\s+i|how\s+to|help\s+with|anyone\s+know|stuck\s+on"
    r")\b",
    re.IGNORECASE,
)


def has_technical_cues(text: str) -> bool:
    if not text or not isinstance(text, str):
        return False
    return _TECH_CUES.search(text) is not None


def vader_rule_predict(
    text_raw: str,
    analyzer: SentimentIntensityAnalyzer,
    neutral_band: float,
    rule_style: str,
) -> tuple[int, float]:
    """
    Returns (predicted_label, compound). TPS = 1, GDS = 0.

    rule_style:
      - "emotion_first" (default): strong |compound| → GDS; else neutral+technical → TPS; else GDS.
      - "technical_first": technical cues → TPS unless sentiment is extremely negative (rant-only);
        else strong emotion → GDS; else GDS. Often works better when VADER marks technical text as non-neutral.
    """
    text_raw = text_raw or ""
    compound = float(analyzer.polarity_scores(text_raw)["compound"])
    neutral = -neutral_band <= compound <= neutral_band
    strong_emotion = abs(compound) > neutral_band
    tech = has_technical_cues(text_raw)

    if rule_style == "technical_first":
        # Complain-only / extreme negativity without problem-solving wording → GDS
        if compound < -0.6 and not tech:
            return 0, compound
        if tech:
            return 1, compound
        if strong_emotion:
            return 0, compound
        return 0, compound

    # emotion_first (original plan)
    if strong_emotion:
        return 0, compound
    if neutral and tech:
        return 1, compound
    return 0, compound


def main() -> int:
    p = argparse.ArgumentParser(description="VADER rule baseline TPS vs GDS.")
    p.add_argument("--labels", type=Path, default=_DEFAULT_LABELS)
    p.add_argument("--data-json", type=Path, default=_DEFAULT_JSON)
    p.add_argument("--unique-labels-out", type=Path, default=_UNIQUE_LABELS_OUT)
    p.add_argument("--gds-sample-size", type=int, default=175)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument(
        "--neutral-band",
        type=float,
        default=0.15,
        help=(
            "VADER compound within [-b,b] counts as neutral for the TPS rule. "
            "Default 0.15 — stricter 0.05 often labels everything emotional as GDS; tune on validation."
        ),
    )
    p.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT)
    p.add_argument(
        "--rule-style",
        choices=("emotion_first", "technical_first"),
        default="technical_first",
        help=(
            "emotion_first: match written research plan (often predicts all-GDS with VADER). "
            "technical_first: prefer TPS when technical cues fire (recommended for reporting alongside NB)."
        ),
    )
    args = p.parse_args()

    n_unique = write_unique_labels_csv(args.labels, args.unique_labels_out)
    df = load_verified_frame(args.labels, args.data_json)
    if len(df) != n_unique:
        print(
            f"Warning: merged rows ({len(df)}) != unique label rows ({n_unique}).",
            file=sys.stderr,
        )

    df_bal = undersample_gds(
        df,
        n_gds=args.gds_sample_size,
        label_col="label",
        random_state=args.random_state,
    )

    model_df = df_bal[
        ["text", "text_raw", "label", "reddit_id", "num_comments_in_post", "_id", "subreddit"]
    ].copy()

    X_train, X_val, X_test, y_train, y_val, y_test = stratified_train_val_test(
        model_df,
        y_col="label",
        test_size=0.15,
        val_size=0.15,
        random_state=args.random_state,
    )

    analyzer = SentimentIntensityAnalyzer()

    def predict_series(series: pd.Series) -> list[int]:
        out: list[int] = []
        for t in series:
            pred, _ = vader_rule_predict(
                str(t), analyzer, args.neutral_band, args.rule_style
            )
            out.append(pred)
        return out

    y_val_pred = pd.Series(predict_series(X_val["text_raw"]), index=y_val.index)
    y_pred = pd.Series(predict_series(X_test["text_raw"]), index=y_test.index)

    val_f1 = f1_score(y_val, y_val_pred, pos_label=1, average="binary")
    acc = accuracy_score(y_test, y_pred)
    f1_tps = f1_score(y_test, y_pred, pos_label=1, average="binary")
    f1_macro = f1_score(y_test, y_pred, average="macro")
    prec, rec, _, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", pos_label=1, zero_division=0
    )

    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["GDS", "TPS"],
        digits=4,
        zero_division=0,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.rule_style == "emotion_first":
        rule_lines = [
            "If |compound| > neutral_band → predict GDS (strong emotional tone).",
            "Else if neutral and technical_cues_regex(text_raw) → predict TPS.",
            "Else → predict GDS.",
        ]
    else:
        rule_lines = [
            "If compound < -0.6 and no technical cues → predict GDS (rant-only).",
            "Else if technical_cues_regex(text_raw) → predict TPS.",
            "Else if |compound| > neutral_band → predict GDS.",
            "Else → predict GDS.",
        ]
    rule_doc = {
        "neutral_band": args.neutral_band,
        "rule_style": args.rule_style,
        "rules": rule_lines,
        "text_for_vader": "text_raw (title + body + comments, uncleaned)",
    }

    metrics = {
        "model": "vader_rule_baseline",
        "verified_unique_posts": int(len(df)),
        "after_gds_undersample_n": int(len(df_bal)),
        "n_tps_after_undersample": int((df_bal["label"] == 1).sum()),
        "n_gds_after_undersample": int((df_bal["label"] == 0).sum()),
        "train_size": int(len(y_train)),
        "val_size": int(len(y_val)),
        "test_size": int(len(y_test)),
        "val_f1_tps": float(val_f1),
        "test_accuracy": float(acc),
        "test_f1_tps": float(f1_tps),
        "test_f1_macro": float(f1_macro),
        "test_precision_tps": float(prec),
        "test_recall_tps": float(rec),
        "random_state": args.random_state,
        "gds_sample_size": args.gds_sample_size,
        "rule_style": args.rule_style,
        "rule_spec": rule_doc,
    }
    with (args.out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with (args.out_dir / "classification_report.txt").open("w", encoding="utf-8") as f:
        f.write(report)

    with (args.out_dir / "rule_description.json").open("w", encoding="utf-8") as f:
        json.dump(rule_doc, f, indent=2)

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["GDS", "TPS"],
        ax=ax,
        colorbar=True,
    )
    ax.set_title("VADER rule baseline — test set")
    fig.tight_layout()
    fig.savefig(args.out_dir / "confusion_matrix_test.png", dpi=150)
    plt.close(fig)

    # Save test predictions for error analysis
    test_out = X_test[["reddit_id", "text_raw"]].copy()
    test_out["label_true"] = y_test.values
    test_out["label_pred"] = y_pred.values
    test_out["vader_compound"] = [
        analyzer.polarity_scores(str(t))["compound"] for t in X_test["text_raw"]
    ]
    test_out.to_csv(args.out_dir / "test_predictions.csv", index=False)

    def save_split(name: str, X_part: pd.DataFrame, y_part: pd.Series) -> None:
        out = X_part.copy()
        out["label"] = y_part.values
        out.to_csv(args.out_dir / f"{name}.csv", index=False)

    save_split("train_split", X_train, y_train)
    save_split("val_split", X_val, y_val)
    save_split("test_split", X_test, y_test)

    print(report)
    print(json.dumps(metrics, indent=2))
    print(f"Outputs: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
