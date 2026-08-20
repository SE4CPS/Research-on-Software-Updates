# Release Notes Classification

Supplementary code for the paper. Citation, PDF, dataset link, full context:
<https://se4cps.github.io/lab/research/>

## What's here

`label/` contains keyword/rule-based labeling logic for Component Type,
Security Risk, and Release Type (`label_breaking_type.py`). No model is
trained; this is tagging logic, not the paper's six classifiers.

| File | Category |
|---|---|
| `label_component_type.py` | Component Type |
| `label_security_type.py` | Security Risk |
| `label_breaking_type.py` | Release Type *(see caveats)* |

`example-reconstruction/` trains and evaluates all six classifiers from
Table 1 against the real dataset using these labels, compared to the
paper's results. See its own README.

## Caveats

- Not the exact 2024 code. Current production labeler; keyword lists
  revised since.
- "Breaking Type" vs. the paper's "Release Type": related, not confirmed
  identical.
- Paper reports Release Type failed for all six classifiers (Table 1:
  "Incorrect"). `example-reconstruction/` explains why.
- The original six-classifier training code isn't included here.
  `example-reconstruction/` is a disclosed rebuild, not a recovery.
- The earlier K-Means clustering step is in this repo's own git history,
  commit `cb8ffc9` (2024-01-06), folder `cluster-top-50/` (since removed).

## Usage

No dependencies, no database. Pure Python standard library:

```python
from label.label_component_type import classify_text as label_component

label_component("Fixed a memory leak in the Chrome extension API.")
# -> ['BROWSER', 'API']
```

```
python label/label_component_type.py
python label/label_security_type.py
python label/label_breaking_type.py
```

## License

MIT. See [LICENSE](LICENSE). The paper is CC BY-NC-ND.
