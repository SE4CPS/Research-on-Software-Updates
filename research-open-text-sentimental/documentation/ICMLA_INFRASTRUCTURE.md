# ICMLA 2026 — Research Infrastructure

Publication-oriented code paths (no dashboard dependencies).

## Documents

| File | Purpose |
|------|---------|
| `ANNOTATION_GUIDELINES.md` | TPS/GDS labeling protocol |
| `DATA_CARD.md` | Corpus provenance and cohorts C1 / C2 / C1′ |
| `documentation/RESEARCH_BLUEPRINT_ICMLA2026.md` | Locked paper roadmap |

## Commands (from `research-open-text-sentimental/`)

```bash
# C1′ trajectories + divergence (542 posts)
python3 scripts/build_c1_prime.py

# 5-fold stratified CV (full 542 by default)
python3 scripts/run_evaluation.py --run-name paper_cv_full

# TPS vs GDS divergence statistics (Mann–Whitney, bootstrap CI, reports)
python3 scripts/analyze_c1_prime.py --plot

# RoBERTa + baselines (slow; pip install -r requirements-ml.txt)
python3 scripts/run_evaluation.py --run-name paper_with_roberta --include-roberta

# Ablation study D1–D5 (Logistic Regression, 5-fold)
python3 scripts/run_ablations.py --run-name ablations_v1

# Human evaluation template (80 stratified posts)
python3 scripts/export_human_eval_template.py

# Legacy undersampled pilot (261 samples)
python3 scripts/run_evaluation.py --undersample-gds --run-name pilot_261
```

## Layout

```
evaluation/          # CV harness, models, metrics
trajectory/          # C1′ VADER trajectories
analysis/            # C1′ statistical analysis
data/c1_prime/       # C1′ outputs
evaluation/outputs/  # CV run artifacts
analysis/outputs/    # Divergence reports & figures
scripts/             # CLI entry points
```

## Extending

- New sklearn/transformer models: implement `evaluation/models/base.py` subclass, register in `evaluation/models/registry.py`.
- RoBERTa: add `evaluation/models/roberta.py` and register without changing runner.
