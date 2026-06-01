# C1′ Divergence Analysis — Full C1′ (542 labeled)

**Posts:** 542 (TPS=86, GDS=456)

## Author more negative than community

| Class | n | Rate |
|-------|---|------|
| TPS | 86 | 0.6977 |
| GDS | 456 | 0.5373 |

- Rate difference (TPS − GDS): **0.1604**
- Bootstrap 95% CI: [0.0509, 0.2686]
- Mann–Whitney p: 0.006007

## Continuous metrics (TPS vs GDS)

Positive `mean_diff` → higher in TPS. Welch *t*-test and Mann–Whitney U; Cohen's *d*; bootstrap 95% CI on mean difference (TPS − GDS).

| Metric | Mean TPS | Mean GDS | Δ mean | Cohen's d | Welch p | MW p | Bootstrap CI (Δ) |
|--------|----------|----------|--------|-----------|---------|------|------------------|
| sentiment_divergence | 0.371 | 0.2872 | 0.0837 | 0.3293 | 0.009415 | 0.009221 | [0.0229, 0.1464] |
| author_mean_sentiment | 0.0091 | 0.1911 | -0.182 | -0.5029 | 4.7e-05 | 2.5e-05 | [-0.2636, -0.0978] |
| community_mean_sentiment | 0.1683 | 0.194 | -0.0257 | -0.1181 | 0.331788 | 0.308531 | [-0.0784, 0.0263] |
| author_trend | 0.0433 | 0.0026 | 0.0406 | 0.1376 | 0.274128 | 0.155335 | [-0.0277, 0.1125] |
| community_trend | -0.0945 | -0.0309 | -0.0636 | -0.2221 | 0.083285 | 0.026128 | [-0.1424, 0.0068] |
| author_volatility | 0.1694 | 0.1536 | 0.0159 | 0.0536 | 0.658527 | 0.851723 | [-0.0517, 0.0854] |
| community_volatility | 0.4753 | 0.482 | -0.0068 | -0.0284 | 0.816012 | 0.783795 | [-0.0635, 0.0535] |
| author_early_late_shift | 0.0031 | -0.0084 | 0.0115 | 0.0939 | 0.425444 | 0.961263 | [-0.0143, 0.0433] |
| community_early_late_shift | -0.0134 | -0.0201 | 0.0067 | 0.0287 | 0.809307 | 0.828545 | [-0.0487, 0.0623] |

## Interpretation notes

- **sentiment_divergence** = |author_mean − community_mean| (magnitude, not direction).
- Author track: body → OP comments → title fallback (see DATA_CARD.md).
- Non-significant *p* does not imply equivalence; report effect sizes and CIs.
