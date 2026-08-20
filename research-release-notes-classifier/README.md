# Release Notes Classification

Supplementary materials for:

> Solomon Berhe, Vanessa Khan, Omhier Khan, Nathan Pader, Ali Zain Farooqi, Marc Maynard, Foutse Khomh.
> **"Triage Software Update Impact via Release Notes Classification."**
> *Procedia Computer Science*, Vol. 238, pp. 618-622, 2024.
> DOI: [10.1016/j.procs.2024.06.069](https://doi.org/10.1016/j.procs.2024.06.069)
> Open-access PDF: <https://publications.polymtl.ca/59770/1/59770_Triage_software.pdf>

## Dataset

The 1,000-release-note dataset used in the paper is on Hugging Face:
<https://huggingface.co/datasets/sberhe/2023-1000-software-release-notes>

## What's in this repo

`classifiers/` contains the **keyword/rule-based classifiers** currently used
in production (part of the [releasetrain.io](https://releasetrain.io/)
pipeline) for the paper's three label categories:

| File | Category | Notes |
|---|---|---|
| `classify_component_type.py` | Component Type | ~28 categories (OS, browser, database, API, framework, cloud, mobile, etc.), each a keyword list |
| `classify_security_type.py` | Security Risk | Security, privacy, legal, compliance, fraud, identity/access, cloud security categories; also force-labels `SECURITY` when a version's `isCve` flag is true |
| `classify_breaking_type.py` | *see note below* | Failure/impact categories (Critical Failure, Limited Functionality, Performance Issues, Compatibility Issues, etc.) plus a regex-based major-version-bump detector (`x.0` / `x.0.0` → "Breaking Update") |

Each script's classification logic is a simple substring/keyword match per
category (see `classify_text()` in each file) — this is the "keyword-based"
labeling approach referenced in the paper, not manual annotation.

### Important caveats

- **This is the current, evolved production version of the classifiers, not
  the exact original 2024 code.** This repo's own history starts March 2025 —
  over a year after the paper's data collection — and the keyword lists have
  been revised since (categories added, bugs fixed). Treat it as a faithful
  reconstruction of the methodology, not a byte-for-byte snapshot of what
  generated the paper's exact reported numbers.
- **Naming mismatch on the third category:** the paper reports a **"Release
  Type"** category; the closest live equivalent in this codebase is named
  **"Breaking Type."** The two are related (both describe the character/impact
  of a release) but have not been confirmed to use an identical label
  taxonomy. Treat the mapping as provisional.
- **The original unsupervised step** (K-Means clustering used earlier in the
  pipeline) is preserved in git history in a different repo:
  [`se4cps/Research-on-Software-Updates`](https://github.com/se4cps/Research-on-Software-Updates),
  commit `cb8ffc9` (2024-01-06), folder `cluster-top-50/` (later removed in a
  2025-08 reorganization — recoverable via `git log --all -- cluster-top-50`).
- **The supervised classifier training code** for the six models reported in
  the paper (Logistic Regression, Naive Bayes, SVM, Random Forest, Gradient
  Boosting, KNN) was not found in any repository, public or private, searched
  so far. If you locate it, please open an issue/PR.

## Setup

```
pip install -r requirements.txt
cp .env.example .env   # fill in your own MongoDB connection string
python classifiers/classify_component_type.py
python classifiers/classify_security_type.py
python classifiers/classify_breaking_type.py
```

Each script expects a MongoDB database (`releasetrain`) with `versions` and
`reddit` collections shaped roughly like:

```jsonc
// versions
{
  "versionId": "...",
  "versionReleaseNotes": "...",
  "versionSearchTags": ["...", "..."],
  "versionReleaseDate": "YYYYMMDD",
  "isCve": false,
  "classification": {}
}

// reddit
{
  "redditId": "...",
  "title": "...",
  "tags": ["...", "..."],
  "created_utc": "YYYYMMDD",
  "classification": {}
}
```

No live database is required to read/reuse the `categories_*` keyword
dictionaries and `classify_text()` functions directly — that's the part most
useful for reproducing the paper's labeling methodology outside of this
pipeline.

## License

MIT — see [LICENSE](LICENSE). The paper itself is CC BY-NC-ND; link to it
rather than redistributing modified copies.
