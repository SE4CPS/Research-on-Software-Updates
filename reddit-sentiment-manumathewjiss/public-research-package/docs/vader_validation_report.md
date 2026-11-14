# VADER Validation Report (Task 3)

**Date:** 2026-02-17 19:35:24  
**Labeled posts:** 25 (manual labels in `data/vader_validation_template.csv`)  

## Agreement with VADER

| VADER reference | Accuracy |
|-----------------|----------|
| Title sentiment (VADER label) | 40.0% (10/25 correct) |
| Author mean bucket (VADER)   | 56.0% (14/25 correct) |

## Confusion matrix: Human vs VADER title label

|  | Human Positive | Human Neutral | Human Negative |
|--|---|---|---|
| **VADER Positive** | 3 | 4 | 1 |
| **VADER Neutral** | 6 | 3 | 0 |
| **VADER Negative** | 2 | 2 | 4 |

## Confusion matrix: Human vs VADER author-mean bucket

|  | Human Positive | Human Neutral | Human Negative |
|--|---|---|---|
| **VADER author Positive** | 11 | 8 | 2 |
| **VADER author Neutral** | 0 | 0 | 0 |
| **VADER author Negative** | 0 | 1 | 3 |

## Interpretation

- **Accuracy:** Proportion of posts where human label matches VADER.
- **Title label:** VADER sentiment of the post title only.
- **Author mean bucket:** VADER compound averaged over author replies, then bucketed (Positive ≥ 0.05, Negative ≤ -0.05, else Neutral).
