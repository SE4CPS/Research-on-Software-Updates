# Example Reconstruction

A worked example: trains and evaluates all six classifiers from Table 1
(Multinomial Naive Bayes, SVM, Logistic Regression, Random Forest, Gradient
Boosting, KNN) against the real 1,000-note dataset, using labels generated
from `../label/label_*.py`. Not a replication of the paper's exact
pipeline — see disclosed choices at the top of `train_and_evaluate.py`
(label-reduction rule, 80/20 split, `CountVectorizer` features, near-default
hyperparameters).

Built with AI assistance (Claude).

## Results (80/20 split)

| Classifier | Component Type | Security Risk | Release Type |
|---|---|---|---|
| Multinomial Naive Bayes | 66.0% | 96.5% | 55.5% |
| SVM (Linear Kernel) | 78.5% | 99.0% | 56.0% |
| Logistic Regression | 77.0% | 98.5% | 57.5% |
| Random Forest | 72.0% | 99.0% | 60.0% |
| Gradient Boosting | 81.5% | 99.5% | 56.5% |
| K-Nearest Neighbors | 52.0% | 88.5% | 44.0% |
| *(majority-class baseline)* | 18.0% | 52.5% | 45.0% |
| **Paper's Table 1** | 58–88% | 81–99% | Incorrect (all six) |

Cohen's Kappa (chance-adjusted agreement) tells a clearer story than raw
accuracy: Component Type 0.47–0.80 (moderate–substantial), Security Risk
0.77–0.99 (substantial–almost perfect), **Release Type 0.20–0.35 (poor–fair,
never reaching "moderate")** — full numbers in `kappa_results.csv`.

## Why Release Type doesn't work

The label is derived from the release's version number (`x.0.0` = major,
`x.y.0` = minor, else patch — see `release_type.py`). But
`CountVectorizer`'s default tokenizer drops tokens shorter than 2
characters, so single-digit version components are usually invisible to the
classifier:

```
"4.0.4"   -> ['vite', 'frontend', 'build', 'tool']        # version: gone entirely
"3.0.0"   -> ['koa', 'beta', 'node', 'js']                 # version: gone entirely
"121.0.0" -> ['linux', 'dist', 'linspire', 'prod', '121']  # only the "121" survives
```

The classifier is effectively being asked to predict a label from
information that mostly isn't present in its own input. That's a more
specific, mechanistic explanation than "the labels were ambiguous" (the
paper's own explanation, Section 4.3) — both are true, and probably
compound each other.

## Reproducing this

```
pip install pandas pyarrow scikit-learn matplotlib
python train_and_evaluate.py
```

Downloads nothing itself — point `DATA_PATH` at a local copy of the
Hugging Face dataset (parquet or CSV; see main README for the dataset link).
