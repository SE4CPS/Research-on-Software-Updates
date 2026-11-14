# Data Card — ReleaseTrain Reddit TPS/GDS & Trajectory Cohorts

**Version:** 1.0  
**Project:** `research-open-text-sentimental` (ICMLA 2026)  
**Last updated:** 2026-05-20  

---

## 1. Summary

| Artifact | Name | n | Primary use |
|----------|------|---|-------------|
| **C1** | ReleaseTrain-TPSGDS-542 | 542 | Supervised TPS vs GDS classification |
| **C2** | ReleaseTrain-Trajectory-324 | 324 | VADER trajectory / engagement-filtered cohort (legacy snapshot) |
| **C1′** | ReleaseTrain-TPSGDS-542-Trajectories | 542 | Unified classification + divergence + trajectory metrics |

**Positive class:** TPS (`corrected_label = 1`) · **Negative class:** GDS (`corrected_label = 0`)

---

## 2. Provenance and source
 
| Field | Value |
|-------|--------|
| **Provider** | [ReleaseTrain.io](https://releasetrain.io/) Reddit ingestion (SE4CPS research) |
| **API base** | `https://releasetrain.io/api/reddit` |
| **Filtered fetch** | `https://releasetrain.io/api/reddit/query/filter` |
| **Raw snapshots** | `tps_gds_classification/data/raw/tps_response.json`, `gds_response.json` |
| **Processed JSON** | `tps_gds_classification/data/tps_gds_dataset.json` |
| **Gold labels** | `tps_gds_classification/data/updated_labeled_dataset_unique.csv` |
| **C2 snapshot** | `data/enhanced_automated_sentiment_results.json` (fetched 2025-10-07) |
| **C1′ output** | `data/c1_prime/c1_prime_metrics.csv`, `c1_prime_dataset.json` |

Data are **public Reddit submissions and comments** collected through project infrastructure—not a new crawl for this card’s version 1.0.

---

## 3. Collection methodology

### 3.1 Candidate pools (not gold labels)

| Pool | Query parameters | Intended candidate class |
|------|------------------|---------------------------|
| TPS pool | `minComments=3`, `minScore=0.3`, `limit≤2000` | Technical / higher-engagement |
| GDS pool | `minComments=3`, `maxScore=0.5`, `limit≤2000` | Lower-score / discontent proxy |

Scripts: `tps_gds_classification/scripts/fetch_tps_gds_dataset.py`

### 3.2 Gold correction

Human annotators set **`corrected_label`** per post (see `ANNOTATION_GUIDELINES.md`).  

**API vs gold disagreement:** 130 / 542 rows (`24.0%`) where `label_current ≠ corrected_label`.

### 3.3 Text construction (classification)

Per post, from API record:

1. `title` + `author_description` (body)  
2. Up to **10** comment bodies (highest `score`, descending)  
3. **`text_raw`:** concatenation (newline-separated)  
4. **`text`:** `text_raw` after URL removal, lowercasing, alphanumeric normalization (`text_preprocessing.py`)

---

## 4. C1 — Classification dataset (542)

| Statistic | Value |
|-----------|--------|
| Unique `reddit_id` | 542 |
| TPS (`corrected_label=1`) | 86 (15.9%) |
| GDS (`corrected_label=0`) | 456 (84.1%) |
| Columns (labels file) | `reddit_id`, `url`, `subreddit`, `label_current`, `label_name_hint`, `title`, `corrected_label`, `notes` |

**Top subreddits (count):** Android (87), transformers (83), Bitcoin (56), chrome (42), linux (25), comfyui (16), neovim (16), immich (15), rust (14), Wordpress (13), …

**Imbalance:** Severe TPS minority—use stratified splits, class weights, **macro-F1**, **TPS-F1**, **PR-AUC**.

---

## 5. C2 — Trajectory cohort (324)

| Statistic | Value |
|-----------|--------|
| Source endpoint | `https://releasetrain.io/api/reddit` (bulk) |
| Posts fetched | 3,519 |
| Posts passing filters | 324 |
| Inclusion rules | ≥10 comments, ≥3 author replies, ≥5 community comments, quality ≥0.3 |
| Sentiment | NLTK VADER compound; labels ±0.05 |

**Top subreddits:** transformers (42), comfyui (33), Wordpress (27), rust (21), linux (16), …

**Note:** Subreddit distribution **differs** from C1 (different sampling frame).

---

## 6. C1′ — Unified trajectory metrics (542)

Built by `scripts/build_c1_prime.py` from **full comment trees** in raw API JSON (not top-10 snippet only).

**Author track (v1.0):** `author_description` (if present) → OP reply comments (chronological) → else post title compound. **Community track:** non-OP comments in chronological order.

| Output | Description |
|--------|-------------|
| `c1_prime_metrics.csv` | One row per labeled post: divergence, slopes, volatility, counts |
| `c1_prime_dataset.json` | Metrics + optional compact trajectories + metadata |
| `build_metadata.json` | Provenance, version, coverage stats |

**Coverage:** 542/542 labeled IDs expected in merged raw responses (`tps_response` + `gds_response`).

**Overlap C1 ∩ C2:** 18 posts (3.3% of C1, 5.6% of C2)—**insufficient** for joint analysis without C1′.

---

## 7. Subreddit filtering (paper tiers)

| Tier | Policy | Use in paper |
|------|--------|--------------|
| **A — Strict** | Software/dev/platform allowlist (see `evaluation/config.py` → `TIER_A_SUBREDDITS`) | Main results table |
| **B — Full** | All 542 / 324 | Appendix robustness |
| **C — Exclude** | Off-topic (e.g. collectible transformers, pure anime) | Sensitivity analysis |

---

## 8. Preprocessing

| Stage | Classification | Trajectories (C1′) |
|-------|----------------|----------------------|
| Token text | `text` cleaned | Per-comment `body` raw for VADER |
| Author vs community | N/A | `is_submitter` or `author` match |
| Ordering | N/A | `created_utc` ascending |
| Empty text | Dropped in fetch optional flag | Skip empty comment bodies |

---

## 9. Ethics and privacy

- **Public data:** Reddit content is publicly posted; users are pseudonymous (`u/…`).
- **No PII enrichment:** Dataset does not add external identity linkage.
- **Redistribution:** Follow Reddit API/terms and institutional IRB policy if applicable; research release should **anonymize** only if required by venue—default is public URLs in `url` column.
- **Sensitive content:** Threads may contain harassment or political content; used only aggregated for research.

---

## 10. Known biases and limitations

| Bias | Impact |
|------|--------|
| **API filter bias** | TPS/GDS candidates pre-stratified by score heuristics |
| **Annotator subjectivity** | Boundary between frustration (TPS) and rant (GDS) |
| **Class imbalance** | Models may favor GDS; accuracy is misleading |
| **Comment subsampling** | `text` uses top-10 by score; rare comments omitted |
| **English-centric** | VADER and lexicons tuned for English |
| **Temporal snapshot** | C2 from Oct 2025; drift not modeled |
| **Subreddit skew** | Android/Bitcoin/transformers over-represented in C1 |
| **VADER construct** | Sentiment ≠ discourse class; used for trajectories only |
| **C1 vs C2 mismatch** | Different inclusion rules—compare only via C1′ |

---

## 11. Recommended splits (evaluation harness)

| Protocol | Setting |
|----------|---------|
| **Primary** | 5-fold stratified CV on full 542 (no GDS undersampling) |
| **Secondary** | Single 70/15/15 holdout, `random_state=42` |
| **Legacy pilot** | GDS undersample to 175—**not** default for paper |

Entry point: `scripts/run_evaluation.py`

---

## 12. File index

```
tps_gds_classification/data/
  updated_labeled_dataset_unique.csv   # C1 labels
  tps_gds_dataset.json                 # C1 text features
  raw/tps_response.json, gds_response.json
data/
  enhanced_automated_sentiment_results.json  # C2
  c1_prime/                                  # C1′ outputs
evaluation/outputs/                          # CV runs
```

---

## 13. Version history

| Version | Date | Notes |
|---------|------|-------|
| 1.0 | 2026-05-20 | Initial data card; C1′ pipeline v1 |

**Citation (working):** *ReleaseTrain-TPSGDS-542* and *ReleaseTrain-TPSGDS-542-Trajectories* (ICMLA 2026 submission, SE4CPS).
