"""
Stratified k-fold cross-validation runner for TPS/GDS models.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from evaluation.config import EvaluationConfig
from evaluation.data_loader import (
    TRAJECTORY_FEATURE_COLS,
    load_c1_extended_frame,
    prepare_modeling_frame,
)
from evaluation.metrics import (
    FoldMetrics,
    aggregate_fold_metrics,
    bootstrap_ci,
    compute_confusion,
    compute_metrics,
)
from evaluation.models.registry import build_model


def _feature_columns(df: pd.DataFrame) -> list[str]:
    base = ["text", "text_raw", "reddit_id", "subreddit", "label"]
    optional = ["text_title_only", "text_title_body", "title_raw", "body_raw"]
    traj = [c for c in TRAJECTORY_FEATURE_COLS if c in df.columns]
    extra = ["author_more_negative"]
    return [c for c in base + optional + traj + extra if c in df.columns]


def run_cross_validation(config: EvaluationConfig, run_name: str | None = None) -> Path:
    """
    Run stratified CV for all models in config.models.
    Writes summary JSON, per-model fold metrics, OOF predictions, confusion matrices.
    """
    df = load_c1_extended_frame(
        config.labels_path,
        config.data_json_path,
        c1_prime_csv=config.c1_prime_csv,
        tier_a_only=config.tier_a_only,
    )
    df = prepare_modeling_frame(
        df,
        undersample_gds=config.undersample_gds,
        gds_sample_size=config.gds_sample_size,
        random_state=config.random_state,
    )

    out_dir = config.resolved_output_dir(run_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_full = df[_feature_columns(df)]
    y_full = df["label"].to_numpy(dtype=int)

    meta: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_samples": int(len(df)),
        "n_tps": int((y_full == 1).sum()),
        "n_gds": int((y_full == 0).sum()),
        "n_folds": config.n_folds,
        "random_state": config.random_state,
        "undersample_gds": config.undersample_gds,
        "tier_a_only": config.tier_a_only,
        "models": list(config.models),
        "labels_path": str(config.labels_path),
        "data_json_path": str(config.data_json_path),
    }
    with (out_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    skf = StratifiedKFold(
        n_splits=config.n_folds,
        shuffle=True,
        random_state=config.random_state,
    )

    summary: dict[str, Any] = {"models": {}}

    for model_name in config.models:
        fold_metrics_list: list[FoldMetrics] = []
        oof_pred = np.full(len(y_full), -1, dtype=int)
        oof_score = np.full(len(y_full), np.nan, dtype=float)

        model_dir = out_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_full, y_full)):
            model = build_model(model_name, config)
            X_train = X_full.iloc[train_idx]
            X_test = X_full.iloc[test_idx]
            y_train = y_full[train_idx]
            y_test = y_full[test_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_score = model.decision_function_tps(X_test)

            fm = compute_metrics(y_test, y_pred, y_score)
            fold_metrics_list.append(fm)

            oof_pred[test_idx] = y_pred
            oof_score[test_idx] = y_score

            cm = compute_confusion(y_test, y_pred)
            fold_record = {
                "fold": fold_idx,
                "metrics": fm.to_dict(),
                "confusion_matrix": cm.tolist(),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
            }
            with (model_dir / f"fold_{fold_idx}.json").open("w", encoding="utf-8") as f:
                json.dump(fold_record, f, indent=2)

        agg = aggregate_fold_metrics(fold_metrics_list)
        boot: dict[str, dict[str, float]] = {}
        valid = oof_pred >= 0
        yt = y_full[valid]
        yp = oof_pred[valid]
        ys = oof_score[valid]
        for metric in ("accuracy", "f1_macro", "f1_tps", "pr_auc_tps"):
            est, lo, hi = bootstrap_ci(
                yt,
                yp,
                ys,
                metric=metric,
                n_samples=config.bootstrap_samples,
                ci=config.bootstrap_ci,
                random_state=config.random_state,
            )
            boot[metric] = {"estimate": est, "ci_lower": lo, "ci_upper": hi}

        oof_df = df.loc[valid, ["reddit_id", "subreddit", "label"]].copy()
        oof_df["pred"] = yp
        oof_df["score_tps"] = ys
        oof_df.to_csv(model_dir / "oof_predictions.csv", index=False)

        cm_oof = compute_confusion(yt, yp)
        pd.DataFrame(
            cm_oof,
            index=["true_GDS", "true_TPS"],
            columns=["pred_GDS", "pred_TPS"],
        ).to_csv(model_dir / "confusion_matrix_oof.csv")

        model_summary = {
            "fold_metrics": [f.to_dict() for f in fold_metrics_list],
            "aggregate": agg,
            "bootstrap_oof": boot,
            "params": build_model(model_name, config).get_params(),
        }
        with (model_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(model_summary, f, indent=2)

        summary["models"][model_name] = {
            "aggregate": agg,
            "bootstrap_oof": boot,
        }

        print(f"\n=== {model_name} ===")
        print(
            f"  macro-F1: {agg['f1_macro_mean']:.4f} ± {agg['f1_macro_std']:.4f}  "
            f"(OOF bootstrap {boot['f1_macro']['ci_lower']:.3f}–{boot['f1_macro']['ci_upper']:.3f})"
        )
        print(
            f"  TPS-F1:   {agg['f1_tps_mean']:.4f} ± {agg['f1_tps_std']:.4f}  "
            f"(OOF PR-AUC {boot['pr_auc_tps']['estimate']:.3f})"
        )

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Pointer for paper scripts
    latest = config.output_dir / "latest"
    if latest.exists() and latest.is_symlink():
        latest.unlink()
    elif latest.exists() and not latest.is_symlink():
        pass  # leave existing dir
    try:
        if not latest.exists():
            latest.symlink_to(out_dir.name)
    except OSError:
        # Windows/sandbox may not support symlinks; copy summary only
        with (out_dir / "summary.json").open(encoding="utf-8") as src:
            data = src.read()
        latest.mkdir(parents=True, exist_ok=True)
        (latest / "summary.json").write_text(data, encoding="utf-8")

    print(f"\nOutputs written to: {out_dir}")
    return out_dir
