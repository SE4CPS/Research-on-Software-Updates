#!/usr/bin/env python3
"""
ICMLA 2026 — unified TPS/GDS evaluation harness.

Single entry point for stratified cross-validation and paper-ready metrics.

Usage (from research-open-text-sentimental/):
  python scripts/run_evaluation.py
  python scripts/run_evaluation.py --models majority,logistic_regression,naive_bayes,vader_rules
  python scripts/run_evaluation.py --tier-a-only --n-folds 5
  python scripts/run_evaluation.py --undersample-gds  # legacy 261-sample pilot

Outputs: evaluation/outputs/<timestamp>/summary.json
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.config import EvaluationConfig  # noqa: E402
from evaluation.models.registry import list_models  # noqa: E402
from evaluation.runner import run_cross_validation  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(
        description="Run stratified CV for TPS vs GDS classifiers (ICMLA infrastructure).",
    )
    p.add_argument(
        "--models",
        type=str,
        default=",".join(EvaluationConfig().models),
        help=f"Comma-separated model names: {', '.join(list_models())}",
    )
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--labels", type=Path, default=EvaluationConfig().labels_path)
    p.add_argument("--data-json", type=Path, default=EvaluationConfig().data_json_path)
    p.add_argument("--output-dir", type=Path, default=EvaluationConfig().output_dir)
    p.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Subfolder under output-dir (default: UTC timestamp)",
    )
    p.add_argument("--undersample-gds", action="store_true")
    p.add_argument("--gds-sample-size", type=int, default=175)
    p.add_argument("--tier-a-only", action="store_true")
    p.add_argument("--bootstrap-samples", type=int, default=1000)
    p.add_argument("--max-tfidf-features", type=int, default=12_000)
    p.add_argument("--vader-rule-style", choices=("technical_first", "emotion_first"), default="technical_first")
    p.add_argument("--vader-neutral-band", type=float, default=0.15)
    p.add_argument(
        "--include-roberta",
        action="store_true",
        help="Add roberta to model list (requires torch, transformers; slow).",
    )
    p.add_argument("--roberta-epochs", type=int, default=2)
    p.add_argument("--roberta-max-length", type=int, default=256)
    args = p.parse_args()

    run_name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.include_roberta and "roberta" not in model_list:
        model_list.append("roberta")
    model_tuple = tuple(model_list)
    unknown = set(model_tuple) - set(list_models())
    if unknown:
        print(f"Unknown models: {unknown}. Available: {list_models()}", file=sys.stderr)
        return 1

    config = EvaluationConfig(
        labels_path=args.labels,
        data_json_path=args.data_json,
        output_dir=args.output_dir,
        random_state=args.random_state,
        n_folds=args.n_folds,
        models=model_tuple,
        undersample_gds=args.undersample_gds,
        gds_sample_size=args.gds_sample_size,
        tier_a_only=args.tier_a_only,
        bootstrap_samples=args.bootstrap_samples,
        max_tfidf_features=args.max_tfidf_features,
        vader_rule_style=args.vader_rule_style,
        vader_neutral_band=args.vader_neutral_band,
        roberta_epochs=args.roberta_epochs,
        roberta_max_length=args.roberta_max_length,
    )

    run_cross_validation(config, run_name=run_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
