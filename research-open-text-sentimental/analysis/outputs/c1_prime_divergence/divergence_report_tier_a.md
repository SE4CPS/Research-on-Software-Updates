# C1′ Divergence Analysis — Tier-A software/dev subreddits

**Posts:** 304 (TPS=53, GDS=251)

## Author more negative than community

| Class | n | Rate |
|-------|---|------|
| TPS | 53 | 0.717 |
| GDS | 251 | 0.5378 |

- Rate difference (TPS − GDS): **0.1791**
- Bootstrap 95% CI: [0.0362, 0.3035]
- Mann–Whitney p: 0.016942

## Continuous metrics (TPS vs GDS)

Positive `mean_diff` → higher in TPS. Welch *t*-test and Mann–Whitney U; Cohen's *d*; bootstrap 95% CI on mean difference (TPS − GDS).

| Metric | Mean TPS | Mean GDS | Δ mean | Cohen's d | Welch p | MW p | Bootstrap CI (Δ) |
|--------|----------|----------|--------|-----------|---------|------|------------------|
| sentiment_divergence | 0.4121 | 0.3068 | 0.1053 | 0.3834 | 0.018462 | 0.012492 | [0.0206, 0.1886] |
| author_mean_sentiment | 0.0319 | 0.2001 | -0.1682 | -0.4324 | 0.005436 | 0.005304 | [-0.2823, -0.0562] |
| community_mean_sentiment | 0.1999 | 0.2 | -0.0002 | -0.0007 | 0.996389 | 0.756245 | [-0.0743, 0.0703] |
| author_trend | 0.0047 | 0.0125 | -0.0078 | -0.0274 | 0.867436 | 0.808041 | [-0.0979, 0.0845] |
| community_trend | -0.1617 | -0.042 | -0.1197 | -0.3763 | 0.022068 | 0.006721 | [-0.2209, -0.0216] |
| author_volatility | 0.1749 | 0.1369 | 0.0379 | 0.1291 | 0.412115 | 0.41367 | [-0.0485, 0.1259] |
| community_volatility | 0.4935 | 0.5075 | -0.014 | -0.0557 | 0.71628 | 0.592757 | [-0.0876, 0.065] |
| author_early_late_shift | -0.0122 | -0.0106 | -0.0016 | -0.0162 | 0.908773 | 0.609448 | [-0.0301, 0.025] |
| community_early_late_shift | -0.023 | -0.018 | -0.005 | -0.0204 | 0.89347 | 0.964734 | [-0.0812, 0.0668] |

## Interpretation notes

- **sentiment_divergence** = |author_mean − community_mean| (magnitude, not direction).
- Author track: body → OP comments → title fallback (see DATA_CARD.md).
- Non-significant *p* does not imply equivalence; report effect sizes and CIs.
