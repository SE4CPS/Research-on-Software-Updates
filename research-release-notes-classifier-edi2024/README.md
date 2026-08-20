# Release Notes Classification

Supplementary code for the "Triage Software Update Impact via Release Notes
Classification" paper — citation, PDF, dataset link, and full context here:
<https://se4cps.github.io/lab/research/>

## What's here

The paper's pipeline is: *release notes &rarr; labels &rarr; train six classifiers
(Naive Bayes, SVM, Logistic Regression, Random Forest, Gradient Boosting, KNN)
&rarr; evaluate*. `label/` contains only the **labeling step** — the
keyword/rule-based logic that assigns each release note its category, not the
six classifiers themselves (that training/evaluation code hasn't been
published here).

| File | Paper category | What it does |
|---|---|---|
| `label_component_type.py` | Component Type | ~28 keyword-matched categories (OS, browser, database, API, cloud, mobile, etc.) |
| `label_security_type.py` | Security Risk | Security/privacy/legal/compliance keyword categories; force-labels `SECURITY` when `isCve` is true |
| `label_breaking_type.py` | Release Type *(see caveat)* | Failure/impact keyword categories + a regex that flags `x.0`/`x.0.0` version bumps as breaking |

Each file uses keyword/substring matching only — no model is trained, so
these are labeling/tagging logic, not classifiers in the ML sense.

`example-reconstruction/` trains and evaluates all six classifiers named in
the paper against the real dataset, using these labels, with results
compared to the paper's Table 1 — see its own README for details and caveats.

## Caveats

- **Not the exact 2024 code.** These are the current production version of
  the labeler (history starts March 2025, over a year after the paper);
  keyword lists have since been revised. Treat them as a faithful
  reconstruction of the methodology, not a byte-for-byte snapshot.
- **"Breaking Type" vs. "Release Type":** the paper's third category is named
  Release Type; `label_breaking_type.py` is the closest live equivalent.
  Related, but not confirmed to be an identical label taxonomy.
- **The paper itself reports Release Type classification failed** — Table 1
  shows all six classifiers scored "Incorrect" on it. `example-reconstruction/`
  digs into why.
- **The six-classifier training/evaluation code** (Multinomial Naive Bayes,
  SVM, Logistic Regression, Random Forest, Gradient Boosting, KNN) as
  originally run for the paper is not included here — `example-reconstruction/`
  is a disclosed rebuild, not a recovery of the original.
- **K-Means clustering step** (an earlier, unsupervised part of the pipeline)
  survives in a different repo's git history:
  [`se4cps/Research-on-Software-Updates`](https://github.com/se4cps/Research-on-Software-Updates),
  commit `cb8ffc9` (2024-01-06), folder `cluster-top-50/`.

## Usage

No dependencies, no database — pure Python standard library:

```python
from label.label_component_type import classify_text as label_component

label_component("Fixed a memory leak in the Chrome extension API.")
# -> ['BROWSER', 'API']
```

Or run any file directly to see it label a sample string:

```
python label/label_component_type.py
python label/label_security_type.py
python label/label_breaking_type.py
```

To label the full Hugging Face dataset, load it and apply the relevant
`classify_text()` to each release note's text.

## License

MIT — see [LICENSE](LICENSE). The paper is CC BY-NC-ND.
