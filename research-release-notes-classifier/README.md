# Release Notes Classification

Supplementary materials for:

> Solomon Berhe, Vanessa Khan, Omhier Khan, Nathan Pader, Ali Zain Farooqi, Marc Maynard, Foutse Khomh.
> **"Triage Software Update Impact via Release Notes Classification."**
> *Procedia Computer Science*, Vol. 238, pp. 618-622, 2024.
> DOI: [10.1016/j.procs.2024.06.069](https://doi.org/10.1016/j.procs.2024.06.069) · paper PDF included in this repo.

**Dataset:** the 1,000-release-note dataset used in the paper is on Hugging Face:
<https://huggingface.co/datasets/sberhe/2023-1000-software-release-notes>

## What's here

The paper's pipeline is: *release notes &rarr; labels &rarr; train six classifiers
(Naive Bayes, SVM, Logistic Regression, Random Forest, Gradient Boosting, KNN)
&rarr; evaluate*. `classifiers/` contains only the **labeling step** — the
keyword/rule-based logic that assigns each release note its category, not the
six classifiers themselves (that training/evaluation code hasn't been located
in any repo searched so far).

| File | Paper category | What it does |
|---|---|---|
| `classify_component_type.py` | Component Type | ~28 keyword-matched categories (OS, browser, database, API, cloud, mobile, etc.) |
| `classify_security_type.py` | Security Risk | Security/privacy/legal/compliance keyword categories; force-labels `SECURITY` when `isCve` is true |
| `classify_breaking_type.py` | Release Type *(see caveat)* | Failure/impact keyword categories + a regex that flags `x.0`/`x.0.0` version bumps as breaking |

## Caveats

- **Not the exact 2024 code.** This is the current production version of the
  labeler (history starts March 2025, over a year after the paper); keyword
  lists have since been revised. Treat it as a faithful reconstruction of the
  methodology, not a byte-for-byte snapshot.
- **"Breaking Type" vs. "Release Type":** the paper's third category is named
  Release Type; this codebase's closest live equivalent is named Breaking
  Type. Related, but not confirmed to be an identical label taxonomy.
- **The paper itself reports Release Type classification failed** — Table 1
  shows all six classifiers scored "Incorrect" on it, with the discussion
  section attributing this to ambiguous labels (`.0.0`, `major`, `breaking`,
  `dependency`) causing over/underfitting. Worth knowing going in.
- **K-Means clustering step** (an earlier, unsupervised part of the pipeline)
  survives in a different repo's git history:
  [`se4cps/Research-on-Software-Updates`](https://github.com/se4cps/Research-on-Software-Updates),
  commit `cb8ffc9` (2024-01-06), folder `cluster-top-50/`.

## Usage

No dependencies, no database — pure Python standard library:

```python
from classifiers.classify_component_type import classify_text as classify_component

classify_component("Fixed a memory leak in the Chrome extension API.")
# -> ['BROWSER', 'API']
```

Or run any file directly to see it classify a sample string:

```
python classifiers/classify_component_type.py
python classifiers/classify_security_type.py
python classifiers/classify_breaking_type.py
```

To label the full Hugging Face dataset, load it and apply the relevant
`classify_text()` to each release note's text.

## License

MIT — see [LICENSE](LICENSE). The paper is CC BY-NC-ND.
