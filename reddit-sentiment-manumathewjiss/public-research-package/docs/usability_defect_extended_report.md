# Usability vs Defect — Extended Sample Comparison

**Generated:** 2026-02-04 11:59:05  
**Usability posts:** 30  
**Defect posts:** 30  

## Summary table

| Metric | Usability | Defect |
|--------|-----------|--------|
| Mean author sentiment | 0.3815 | 0.2535 |
| Mean community sentiment | 0.3597 | 0.158 |
| Mean divergence | 0.1705 | 0.2107 |
| Median divergence | 0.1608 | 0.1712 |
| % author more negative | 43.3% | 36.7% |

## Trend & volatility

| Metric | Usability | Defect |
|--------|-----------|--------|
| Mean author trend | -0.063 | 0.0559 |
| Mean community trend | 0.0021 | 0.0094 |
| Mean author volatility | 0.4845 | 0.4557 |
| Mean community volatility | 0.4686 | 0.4847 |

## Conclusion

Comparison of auto-labeled usability vs defect posts (extended sample). Defect-related threads show lower mean author sentiment than usability-related threads in this sample; divergence and % author more negative can be compared across categories above.

## Visualizations

- **`visualizations/usability_defect_extended_summary.png`** — Aggregate comparison: mean author vs community, divergence, % author more negative (3 panels).
- **`visualizations/usability_defect_distributions.png`** — Histograms of author mean, community mean, and divergence by category (usability vs defect overlaid).
- **`visualizations/usability_defect_scatter.png`** — Scatter: author mean vs community mean per post; points colored by category (usability vs defect).
- **`visualizations/usability_defect_boxplots.png`** — Box plots: author mean, community mean, divergence by category.
- **`visualizations/usability_defect_trend_volatility.png`** — Mean trend and mean volatility (author vs community) by category.

To regenerate all Task 2 figures: `python3 scripts/visualize_usability_defect_extended.py` (plus `compare_usability_defect_extended.py` for the summary chart).
