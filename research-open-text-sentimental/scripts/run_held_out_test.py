#!/usr/bin/env python3
"""
Locked 15% held-out test (seed=42) + McNemar vs pairwise model comparison.

Usage:
  python3 scripts/run_held_out_test.py
  python3 scripts/run_held_out_test.py --models naive_bayes,logistic_regression

Outputs: evaluation/outputs/held_out_test/
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.config import EvaluationConfig
from evaluation.data_loader import load_c1_extended_frame, prepare_modeling_frame
from evaluation.metrics import compute_metrics
from evaluation.models.registry import build_model
from evaluation.runner import _feature_columns

_TPS_SCRIPTS = _REPO_ROOT / "tps_gds_classification" / "scripts"
if str(_TPS_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_TPS_SCRIPTS))
from tps_gds_data import stratified_train_val_test  # noqa: E402


def mcnemar(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """McNemar on two classifiers (binary correct/incorrect)."""
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    b = int((~correct_a & correct_b).sum())  # B wrong, A right -> actually b = A wrong B right
    c = int((correct_a & ~correct_b).sum())
    # standard: b = A correct B wrong, c = A wrong B correct
    b = int((correct_a & ~correct_b).sum())
    c = int((~correct_a & correct_b).sum())
    if b + c == 0:
        return {"b": b, "c": c, "p_value": 1.0, "statistic": 0.0}
    stat = (abs(b - c) - 1) ** 2 / (b + c) if b + c else 0
    p = float(stats.chi2.sf(stat, df=1))
    return {"b": b, "c": c, "p_value": p, "statistic": float(stat)}


def main() -> int:
    p = argparse.ArgumentParser(description="Locked 15% held-out test + McNemar.")
    p.add_argument("--models", type=str, default="naive_bayes,logistic_regression,vader_rules")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=_REPO_ROOT / "evaluation" / "outputs" / "held_out_test")
    args = p.parse_args()

    config = EvaluationConfig(random_state=args.random_state)
    df = load_c1_extended_frame(config.labels_path, config.data_json_path, config.c1_prime_csv)
    df = prepare_modeling_frame(df, False, 175, args.random_state)

    feat = [c for c in _feature_columns(df) if c != "label"]
    model_df = df[feat + ["label"]].copy()
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_train_val_test(
        model_df, y_col="label", random_state=args.random_state
    )
    # Train on train+val (85%), test on locked 15%
    X_tr = pd.concat([X_train, X_val], ignore_index=True)
    y_tr = np.concatenate([y_train.to_numpy(dtype=int), y_val.to_numpy(dtype=int)])
    y_te = y_test.to_numpy(dtype=int)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    preds: dict[str, np.ndarray] = {}
    rows = []

    for name in models:
        model = build_model(name, config)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_test)
        score = model.decision_function_tps(X_test)
        m = compute_metrics(y_te, pred, score)
        preds[name] = pred
        rows.append(
            {
                "model": name,
                "n_test": len(y_te),
                "accuracy": m.accuracy,
                "f1_macro": m.f1_macro,
                "f1_tps": m.f1_tps,
                "pr_auc_tps": m.pr_auc_tps,
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(args.out_dir / "held_out_metrics.csv", index=False)

    mcnemar_results = {}
    for i, a in enumerate(models):
        for b in models[i + 1 :]:
            mcnemar_results[f"{a}_vs_{b}"] = mcnemar(y_te, preds[a], preds[b])

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "70+15 train+val / 15% test (stratified, seed=42)",
        "n_train_val": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "metrics": rows,
        "mcnemar": mcnemar_results,
    }
    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(table.to_string(index=False))
    print(f"Wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
