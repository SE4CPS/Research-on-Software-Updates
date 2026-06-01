# Tier-A Robustness Comparison

**Generated:** 2026-05-27T02:37:23.869867+00:00

Tier-A = software/dev/platform subreddit allowlist (see `evaluation/config.py`).

## Classification (5-fold CV, macro-F1 / TPS-F1)

| Model | Full macro-F1 | Tier-A macro-F1 | Full TPS-F1 | Tier-A TPS-F1 |
|-------|---------------|-----------------|-------------|---------------|
| naive_bayes | 0.659 | 0.703 | 0.463 | 0.535 |
| logistic_regression | 0.632 | 0.622 | 0.352 | 0.348 |
| vader_rules | 0.409 | 0.300 | 0.330 | 0.325 |

## Divergence (TPS − GDS mean |author − community|)

- **Full (n≈542):** Δ=0.0837, MW p=0.0092
- **Tier-A (n≈304):** Δ=0.1053, MW p=0.0125

## Ablation macro-F1 (title-only vs full thread)

- **D1_title_only:** Full 0.758 | Tier-A 0.695
- **D3_full_thread:** Full 0.632 | Tier-A 0.622
- **D4_full_plus_trajectory:** Full 0.647 | Tier-A 0.591

## Interpretation (template)

- If Tier-A metrics stay in the same ballpark, findings are **not driven only** by Bitcoin/transformers toys.
- If divergence p-value weakens on Tier-A, report both but emphasize direction consistency.
- If title-only still wins on Tier-A, the title signal is **robust** in software subs.
