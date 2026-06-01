#!/usr/bin/env python3
"""
Naive Bayes (TF-IDF + MultinomialNB) for TPS vs GDS using manually verified labels.

1. Deduplicate `updated_labeled_dataset.csv` by reddit_id → save `updated_labeled_dataset_unique.csv`
2. Merge with `tps_gds_dataset.json` (deduped by reddit_id) for full text
3. Undersample GDS (keep all TPS), stratified train/val/test 70/15/15
4. Train with balanced sample weights on the training set

Run from repo root:
  python tps_gds_classification/scripts/train_naive_bayes.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

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
_DEFAULT_OUT = _PROJECT_ROOT / "outputs" / "naive_bayes"


def main() -> int:
    p = argparse.ArgumentParser(description="Train Naive Bayes TPS vs GDS (verified labels).")
    p.add_argument("--labels", type=Path, default=_DEFAULT_LABELS)
    p.add_argument("--data-json", type=Path, default=_DEFAULT_JSON)
    p.add_argument("--unique-labels-out", type=Path, default=_UNIQUE_LABELS_OUT)
    p.add_argument("--gds-sample-size", type=int, default=175, help="Random GDS count after undersampling.")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument(
        "--max-tfidf-features",
        type=int,
        default=12_000,
    )
    p.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT)
    args = p.parse_args()

    n_unique = write_unique_labels_csv(args.labels, args.unique_labels_out)

    df = load_verified_frame(args.labels, args.data_json)
    if len(df) != n_unique:
        print(
            f"Warning: merged rows ({len(df)}) != unique label rows ({n_unique}). "
            "Check reddit_id overlap between CSV and JSON.",
            file=sys.stderr,
        )

    df_bal = undersample_gds(
        df,
        n_gds=args.gds_sample_size,
        label_col="label",
        random_state=args.random_state,
    )

    model_df = df_bal[
        ["text", "label", "reddit_id", "num_comments_in_post", "_id", "subreddit"]
    ].copy()

    X_train, X_val, X_test, y_train, y_val, y_test = stratified_train_val_test(
        model_df,
        y_col="label",
        test_size=0.15,
        val_size=0.15,
        random_state=args.random_state,
    )

    sw_train = compute_sample_weight("balanced", y_train)

    pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=args.max_tfidf_features,
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
            ("clf", MultinomialNB()),
        ]
    )

    pipe.fit(X_train["text"], y_train, clf__sample_weight=sw_train)

    y_val_pred = pipe.predict(X_val["text"])
    y_pred = pipe.predict(X_test["text"])

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

    metrics = {
        "model": "naive_bayes_multinomial",
        "vectorizer": "tfidf",
        "verified_unique_posts": int(len(df)),
        "after_gds_undersample_n": int(len(df_bal)),
        "n_tps_after_undersample": int((df_bal["label"] == 1).sum()),
        "n_gds_after_undersample": int((df_bal["label"] == 0).sum()),
        "train_size": int(len(y_train)),
        "val_size": int(len(y_val)),
        "test_size": int(len(y_test)),
        "training_sample_weight": "balanced",
        "val_f1_tps": float(val_f1),
        "test_accuracy": float(acc),
        "test_f1_tps": float(f1_tps),
        "test_f1_macro": float(f1_macro),
        "test_precision_tps": float(prec),
        "test_recall_tps": float(rec),
        "random_state": args.random_state,
        "gds_sample_size": args.gds_sample_size,
    }
    with (args.out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with (args.out_dir / "classification_report.txt").open("w", encoding="utf-8") as f:
        f.write(report)

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["GDS", "TPS"],
        ax=ax,
        colorbar=True,
    )
    ax.set_title("Naive Bayes — test set")
    fig.tight_layout()
    fig.savefig(args.out_dir / "confusion_matrix_test.png", dpi=150)
    plt.close(fig)

    joblib.dump(pipe, args.out_dir / "nb_pipeline.joblib")

    def save_split(name: str, X_part: pd.DataFrame, y_part: pd.Series) -> None:
        out = X_part.copy()
        out["label"] = y_part.values
        out.to_csv(args.out_dir / f"{name}.csv", index=False)

    save_split("train_split", X_train, y_train)
    save_split("val_split", X_val, y_val)
    save_split("test_split", X_test, y_test)

    meta = {
        "unique_labels_csv": str(args.unique_labels_out),
        "n_rows_unique_labels": n_unique,
        "merged_posts": len(df),
        "imbalance": "GDS undersampled + balanced sample_weight on train",
    }
    with (args.out_dir / "run_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(report)
    print(json.dumps(metrics, indent=2))
    print(f"Unique labels saved: {args.unique_labels_out}")
    print(f"Outputs: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
