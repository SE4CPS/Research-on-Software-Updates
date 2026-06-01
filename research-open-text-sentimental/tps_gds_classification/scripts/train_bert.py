#!/usr/bin/env python3
"""
Fine-tune bert-base-uncased for TPS vs GDS (binary) with verified labels.

- Same pipeline as Naive Bayes: merge labels + JSON, undersample GDS, stratified 70/15/15
- Input: text_raw or text, max 512 tokens
- Class-weighted cross-entropy (balanced weights from training labels)

Run from repo root:
  pip install torch transformers datasets accelerate
  python tps_gds_classification/scripts/train_bert.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

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
_DEFAULT_OUT = _PROJECT_ROOT / "outputs" / "bert"


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class WeightedLossTrainer(Trainer):
    """Cross-entropy with per-class weights."""

    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.clone().detach()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        w = self.class_weights.to(logits.device)
        loss_fct = torch.nn.CrossEntropyLoss(weight=w)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def make_compute_metrics():
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "f1_macro": float(f1_score(labels, preds, average="macro")),
            "f1_tps": float(
                f1_score(labels, preds, pos_label=1, average="binary", zero_division=0)
            ),
        }

    return compute_metrics


def main() -> int:
    p = argparse.ArgumentParser(description="Fine-tune BERT for TPS vs GDS.")
    p.add_argument("--labels", type=Path, default=_DEFAULT_LABELS)
    p.add_argument("--data-json", type=Path, default=_DEFAULT_JSON)
    p.add_argument("--unique-labels-out", type=Path, default=_UNIQUE_LABELS_OUT)
    p.add_argument("--gds-sample-size", type=int, default=175)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--model-name", type=str, default="bert-base-uncased")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT)
    p.add_argument(
        "--text-column",
        choices=("text_raw", "text"),
        default="text_raw",
        help="text_raw = natural language; text = cleaned.",
    )
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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
        ["text_raw", "text", "label", "reddit_id", "num_comments_in_post", "_id", "subreddit"]
    ].copy()

    X_train, X_val, X_test, y_train, y_val, y_test = stratified_train_val_test(
        model_df,
        y_col="label",
        test_size=0.15,
        val_size=0.15,
        random_state=args.random_state,
    )

    _ = pick_device()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "GDS", 1: "TPS"},
        label2id={"GDS": 0, "TPS": 1},
    )

    text_col = args.text_column

    def to_hf(X: pd.DataFrame, y: pd.Series) -> Dataset:
        texts = X[text_col].fillna("").astype(str).tolist()
        labs = y.astype(int).tolist()
        return Dataset.from_dict({"text": texts, "labels": labs})

    train_ds = to_hf(X_train, y_train)
    val_ds = to_hf(X_val, y_val)
    test_ds = to_hf(X_test, y_test)

    def tokenize_batch(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )

    train_tok = train_ds.map(tokenize_batch, batched=True, remove_columns=["text"])
    val_tok = val_ds.map(tokenize_batch, batched=True, remove_columns=["text"])
    test_tok = test_ds.map(tokenize_batch, batched=True, remove_columns=["text"])

    classes = np.array([0, 1])
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train.values)
    class_weights = torch.tensor(cw, dtype=torch.float32)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    use_fp16 = torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_tps",
        greater_is_better=True,
        save_total_limit=2,
        seed=args.seed,
        logging_steps=10,
        report_to="none",
        fp16=use_fp16,
    )

    trainer = WeightedLossTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=make_compute_metrics(),
    )

    trainer.train()

    best_path = out_dir / "best_model"
    best_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best_path))
    tokenizer.save_pretrained(str(best_path))

    pred_output = trainer.predict(test_tok)
    logits = pred_output.predictions
    y_pred = np.argmax(logits, axis=-1)
    y_true = y_test.values

    acc = accuracy_score(y_true, y_pred)
    f1_tps = f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    prec, rec, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )

    val_metrics = trainer.evaluate(val_tok)

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["GDS", "TPS"],
        digits=4,
        zero_division=0,
    )

    def _floatify(d):
        out = {}
        for k, v in d.items():
            if isinstance(v, (int, float)):
                out[k] = float(v)
        return out

    metrics = {
        "model": "bert-base-uncased_finetuned",
        "model_name": args.model_name,
        "max_length": args.max_length,
        "text_column": text_col,
        "verified_unique_posts": int(len(df)),
        "after_gds_undersample_n": int(len(df_bal)),
        "n_tps_after_undersample": int((df_bal["label"] == 1).sum()),
        "n_gds_after_undersample": int((df_bal["label"] == 0).sum()),
        "train_size": int(len(y_train)),
        "val_size": int(len(y_val)),
        "test_size": int(len(y_test)),
        "class_weights_balanced": class_weights.tolist(),
        "val_eval": _floatify(val_metrics),
        "test_accuracy": float(acc),
        "test_f1_tps": float(f1_tps),
        "test_f1_macro": float(f1_macro),
        "test_precision_tps": float(prec),
        "test_recall_tps": float(rec),
        "random_state": args.random_state,
        "gds_sample_size": args.gds_sample_size,
        "training_args": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
    }

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with (out_dir / "classification_report.txt").open("w", encoding="utf-8") as f:
        f.write(report)

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["GDS", "TPS"],
        ax=ax,
        colorbar=True,
    )
    ax.set_title("BERT — test set")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix_test.png", dpi=150)
    plt.close(fig)

    def save_split(name: str, X_part: pd.DataFrame, y_part: pd.Series) -> None:
        out = X_part.copy()
        out["label"] = y_part.values
        out.to_csv(out_dir / f"{name}.csv", index=False)

    save_split("train_split", X_train, y_train)
    save_split("val_split", X_val, y_val)
    save_split("test_split", X_test, y_test)

    test_pred_df = X_test[["reddit_id", text_col]].copy()
    test_pred_df["label_true"] = y_true
    test_pred_df["label_pred"] = y_pred
    test_pred_df.to_csv(out_dir / "test_predictions.csv", index=False)

    print(report)
    slim = {k: v for k, v in metrics.items() if k != "val_eval"}
    print(json.dumps(slim, indent=2))
    print(f"Saved model to {best_path}")
    print(f"Outputs under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
