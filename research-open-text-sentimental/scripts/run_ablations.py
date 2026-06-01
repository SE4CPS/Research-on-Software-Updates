#!/usr/bin/env python3
"""
Ablation study (RQ3): text granularity + trajectory / subreddit features.

Uses 5-fold stratified CV with Logistic Regression (fast, interpretable).

Usage:
  python3 scripts/run_ablations.py
  python3 scripts/run_ablations.py --run-name ablations_v1

Outputs: evaluation/outputs/ablations/<run_name>/
  summary.json
  ablation_table.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.config import EvaluationConfig  # noqa: E402
from evaluation.data_loader import load_c1_extended_frame, prepare_modeling_frame  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    FoldMetrics,
    aggregate_fold_metrics,
    bootstrap_ci,
    compute_metrics,
)
from evaluation.models.logistic_hybrid import LogisticHybridModel  # noqa: E402
from evaluation.models.logistic_regression import LogisticRegressionModel  # noqa: E402
from evaluation.runner import _feature_columns  # noqa: E402


@dataclass
class AblationSpec:
    ablation_id: str
    description: str
    text_column: str
    use_hybrid: bool = False
    use_trajectory: bool = False
    use_subreddit: bool = False


ABLATIONS: tuple[AblationSpec, ...] = (
    AblationSpec("D1_title_only", "Title only (raw)", "text_title_only"),
    AblationSpec("D2_title_body", "Title + post body", "text_title_body"),
    AblationSpec("D3_full_thread", "Full thread (cleaned TF-IDF text)", "text"),
    AblationSpec(
        "D4_full_plus_trajectory",
        "Full thread + C1′ trajectory features",
        "text",
        use_hybrid=True,
        use_trajectory=True,
    ),
    AblationSpec(
        "D5_full_plus_subreddit",
        "Full thread + subreddit one-hot (leakage check)",
        "text",
        use_hybrid=True,
        use_subreddit=True,
    ),
)


def _build_model(spec: AblationSpec, config: EvaluationConfig):
    if spec.use_hybrid:
        return LogisticHybridModel(
            text_column=spec.text_column,
            max_features=config.max_tfidf_features,
            random_state=config.random_state,
            use_trajectory=spec.use_trajectory,
            use_subreddit=spec.use_subreddit,
        )
    return LogisticRegressionModel(
        text_column=spec.text_column,
        max_features=config.max_tfidf_features,
        random_state=config.random_state,
    )


def run_ablation_cv(
    df: pd.DataFrame,
    spec: AblationSpec,
    config: EvaluationConfig,
) -> dict[str, Any]:
    X_full = df[_feature_columns(df)]
    y_full = df["label"].to_numpy(dtype=int)

    skf = StratifiedKFold(
        n_splits=config.n_folds,
        shuffle=True,
        random_state=config.random_state,
    )

    fold_metrics_list: list[FoldMetrics] = []
    oof_pred = np.full(len(y_full), -1, dtype=int)
    oof_score = np.full(len(y_full), np.nan, dtype=float)

    for train_idx, test_idx in skf.split(X_full, y_full):
        model = _build_model(spec, config)
        X_train, X_test = X_full.iloc[train_idx], X_full.iloc[test_idx]
        y_train, y_test = y_full[train_idx], y_full[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_score = model.decision_function_tps(X_test)
        fold_metrics_list.append(compute_metrics(y_test, y_pred, y_score))
        oof_pred[test_idx] = y_pred
        oof_score[test_idx] = y_score

    agg = aggregate_fold_metrics(fold_metrics_list)
    valid = oof_pred >= 0
    yt, yp, ys = y_full[valid], oof_pred[valid], oof_score[valid]
    boot = {}
    for metric in ("f1_macro", "f1_tps", "pr_auc_tps"):
        est, lo, hi = bootstrap_ci(
            yt, yp, ys, metric=metric,
            n_samples=config.bootstrap_samples,
            random_state=config.random_state,
        )
        boot[metric] = {"estimate": est, "ci_lower": lo, "ci_upper": hi}

    return {
        "ablation_id": spec.ablation_id,
        "description": spec.description,
        "text_column": spec.text_column,
        "aggregate": agg,
        "bootstrap_oof": boot,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="TPS/GDS ablation study (5-fold CV).")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--tier-a-only", action="store_true")
    args = p.parse_args()

    config = EvaluationConfig(
        n_folds=args.n_folds,
        random_state=args.random_state,
        tier_a_only=args.tier_a_only,
    )

    df = load_c1_extended_frame(
        config.labels_path,
        config.data_json_path,
        c1_prime_csv=config.c1_prime_csv,
        tier_a_only=config.tier_a_only,
    )
    df = prepare_modeling_frame(df, undersample_gds=False, gds_sample_size=175, random_state=42)

    run_name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = config.output_dir / "ablations" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    rows = []

    for spec in ABLATIONS:
        print(f"\n--- {spec.ablation_id}: {spec.description} ---")
        res = run_ablation_cv(df, spec, config)
        results.append(res)
        agg = res["aggregate"]
        boot = res["bootstrap_oof"]
        rows.append(
            {
                "ablation_id": spec.ablation_id,
                "description": spec.description,
                "macro_f1_mean": agg["f1_macro_mean"],
                "macro_f1_std": agg["f1_macro_std"],
                "tps_f1_mean": agg["f1_tps_mean"],
                "tps_f1_std": agg["f1_tps_std"],
                "pr_auc_mean": agg["pr_auc_tps_mean"],
                "macro_f1_oof": boot["f1_macro"]["estimate"],
                "macro_f1_ci_low": boot["f1_macro"]["ci_lower"],
                "macro_f1_ci_high": boot["f1_macro"]["ci_upper"],
            }
        )
        print(
            f"  macro-F1: {agg['f1_macro_mean']:.4f} ± {agg['f1_macro_std']:.4f}  "
            f"TPS-F1: {agg['f1_tps_mean']:.4f}"
        )

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "ablation_table.csv", index=False)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(df),
        "n_folds": config.n_folds,
        "model": "logistic_regression or logistic_hybrid",
        "ablations": results,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote {out_dir / 'ablation_table.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
