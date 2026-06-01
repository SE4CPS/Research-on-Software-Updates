# Model comparison — TPS vs GDS (test set)

Same pipeline for all models: verified labels (542 unique posts merged with JSON), GDS undersampled to **175** with all TPS kept (**261** rows), stratified split **70% / 15% / 15%** (`random_state=42`). **Test set = 40** examples.

| Model | Val F1 (TPS) | Test accuracy | Test F1 (TPS) | Test F1 (macro) | Test precision (TPS) | Test recall (TPS) |
|-------|----------------|---------------|---------------|-----------------|------------------------|-------------------|
| **VADER rule baseline** (`technical_first`, neutral band 0.15) | 0.476 | 0.500 | 0.545 | 0.495 | 0.387 | 0.923 |
| **Naive Bayes** (TF-IDF + MultinomialNB, balanced sample weights) | 0.516 | **0.700** | **0.571** | **0.670** | **0.533** | 0.615 |
| **BERT** (`bert-base-uncased`, fine-tuned, max length 512, weighted CE) | 0.545† | 0.650 | 0.563 | 0.635 | 0.474 | 0.692 |

† BERT val F1 (TPS) is from the **final training epoch** validation (`eval_f1_tps` after epoch 4), not a separate early metric key.

## Source metrics files

- `outputs/vader_baseline/metrics.json`
- `outputs/naive_bayes/metrics.json`
- `outputs/bert/metrics.json`

## Summary (test set)

By **test F1 (macro)** and **test accuracy**, **Naive Bayes** is strongest in this run. **BERT** is second on macro F1; **VADER** (`technical_first`) has the highest **TPS recall** but low **TPS precision** and lowest overall accuracy.
