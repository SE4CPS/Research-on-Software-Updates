# Research Blueprint — ICMLA 2026

**Project:** `research-open-text-sentimental`  
**Status:** Direction locked — TPS/GDS classification + author–community divergence + trajectory support  
**Purpose:** Master roadmap for paper development (no implementation in this document)  
**Last updated:** 2026-05-20  

---

## Locked positioning

| In scope | Out of scope (paper narrative) |
|----------|--------------------------------|
| Applied NLP / discourse classification | Dashboard as contribution |
| TPS vs GDS benchmark | Netlify / live panel engineering |
| Author–community divergence (empirical) | “Update-risk proxy” without label alignment |
| Multi-level sentiment trajectories (method) | Usability-keyword exploratory study |
| ReleaseTrain as **data source** only | VADER as SOTA sentiment |

---

## 1. Final paper title options (ICMLA-style)

1. **Classifying Technical Problem-Solving vs. General Discontent in Software-Update Reddit Discussions**
2. **TPS vs. GDS: A Labeled Corpus and Benchmark for Software-Update Discourse on Reddit**
3. **Author–Community Sentiment Divergence and Discourse Type in Technical Reddit Threads**
4. **Beyond Title Sentiment: Discourse Classification and Emotional Divergence in Software-Update Forums**
5. **When the OP Disagrees with the Crowd: Discourse Classification and Sentiment Divergence in Update-Related Reddit Threads**
6. **A Multi-Level Trajectory Study of Technical vs. Emotional Discourse in Software Ecosystem Forums**
7. **Benchmarking Technical Problem-Solving vs. General Discontent with Author–Community Trajectory Analysis**
8. **Discourse-Aware Sentiment Analysis for Software-Update Communities: Classification, Divergence, and Trajectories**
9. **From Rants to Root Cause: Supervised Classification of Reddit Software-Update Discourse at Scale**
10. **ReleaseTrain Reddit Corpus: TPS/GDS Classification and Author–Community Sentiment Divergence**

**Recommended primary title:** **#1 or #7** (clear task + domain).  
**Recommended subtitle (if allowed):** *A 542-Post Labeled Benchmark with Trajectory and Divergence Analysis*

---

## 2. Final research questions

### RQ1 — Classification (primary, ML)
Can we reliably distinguish **Technical Problem-Solving (TPS)** from **General Discontent (GDS)** in software-update-related Reddit threads using supervised models on full-thread text, and how do classical vs. transformer models compare under class imbalance?

### RQ2 — Divergence (primary, empirical)
Do TPS and GDS threads exhibit **systematically different author–community sentiment divergence**, and is the OP more negative than the community more often in GDS than in TPS?

### RQ3 — Trajectories (supporting, methodological + empirical)
Do **comment-index sentiment trajectories** (author vs. community) differ by discourse class in slope, volatility, and early-vs-late shift—and does adding trajectory statistics improve classification beyond bag-of-words text?

### RQ4 — Generalization (secondary, applied)
How do results vary across **subreddits** and engagement levels, and what are the failure modes of lexicon-based (VADER) baselines for this domain?

**Excluded RQs (do not pursue in main paper):** dashboard usability; live API latency; minScore/maxScore as validated “risk” without `corrected_label` alignment.

---

## 3. Final contribution list (numbered)

**C1 — Task and labeled benchmark**  
We introduce the **TPS vs. GDS** binary discourse classification task for software-update-related Reddit content and release a **542-post expert-corrected corpus** (86 TPS, 456 GDS) with documented text construction (title + body + comments) and API provenance (ReleaseTrain).

**C2 — Classification benchmark and analysis**  
We report a **reproducible evaluation protocol** (stratified splits / cross-validation, imbalance-aware metrics) comparing **must-run baselines** (majority, TF-IDF+LR, TF-IDF+NB, rule+VADER) and **transformer fine-tuning** (RoBERTa), with ablations on text granularity and trajectory-derived features.

**C3 — Author–community divergence finding**  
On a **trajectory-eligible cohort** (C2, n=324 with engagement filters), we quantify **sentiment divergence** between OP and community and show **statistically significant association** with TPS/GDS where labels exist, plus cohort-wide descriptive laws (e.g., ~41% author-more-negative).

**C4 — Multi-level trajectory methodology**  
We define **role-separated, time-ordered sentiment trajectories** and demonstrate that trajectory statistics carry signal beyond title-only sentiment—motivating discourse-aware triage for update-intelligence systems.

---

## 4. Core thesis and novelty

### What the paper claims
Software-update discussions on Reddit are not monolithic: they split into **problem-solving discourse (TPS)** and **discontent-dominated discourse (GDS)**. This split is **learnable from text** with measurable accuracy, and it correlates with **how emotional alignment between OP and community evolves** over the thread. **Title-level sentiment is insufficient**; thread structure matters.

### Scientific contribution
- **Primary:** Supervised **discourse classification** benchmark in a defined applied domain (software updates / ecosystem forums).  
- **Secondary:** **Empirical discovery** about **author–community affective divergence** conditioned on discourse type.  
- **Tertiary:** **Operationalization** of multi-party sentiment trajectories for NLP evaluation.

### Novelty (honest framing)
| Claim | Strength |
|-------|----------|
| New **TPS/GDS** task + 542 corrected labels tied to ReleaseTrain | **Medium–high** (if guidelines + release) |
| Linking discourse class to **divergence/trajectories** | **Medium** (bridge cohort currently small—see §5) |
| VADER / Reddit sentiment trajectories alone | **Low** — must be subordinate |
| Live dashboard | **None** for ICMLA |

**Not claiming:** new transformer architecture; causal impact on update failures; human-level sentiment understanding.

---

## 5. Dataset definition

### C1 — Classification corpus (542 posts)
| Field | Value |
|-------|--------|
| Source file | `tps_gds_classification/data/updated_labeled_dataset_unique.csv` + `tps_gds_dataset.json` |
| Unit | One Reddit submission + included comments (per `text_raw` policy) |
| Labels | `corrected_label`: **1 = TPS**, **0 = GDS** |
| Size | **542** unique `reddit_id` |
| Balance | **86 TPS (15.9%)**, **456 GDS (84.1%)** — severe imbalance |
| API prior | `label_current` disagrees with `corrected_label` on **130/542 (24%)** — documents noise in candidate labeling |
| Annotation notes column | Empty today — **must be filled in guidelines retroactively** |
| Candidate provenance | TPS pool: `minScore=0.3`; GDS pool: `maxScore=0.5` (API filters)—**not** final labels |

**Paper name:** **ReleaseTrain-TPSGDS-542** (working).

### C2 — Trajectory cohort (324 posts)
| Field | Value |
|-------|--------|
| Source | `data/enhanced_automated_sentiment_results.json` → `all_analyzed_posts` |
| Inclusion | ≥10 comments, ≥3 author replies, ≥5 community comments, quality score ≥0.3 |
| Fetched pool | 3,519 posts → **324** analyzed (2025-10-07 snapshot) |
| Sentiment | VADER compound; thresholds ±0.05 |
| Metrics | `author_avg_sentiment`, `community_avg_sentiment`, `sentiment_divergence`, trends, volatility |

**Paper name:** **ReleaseTrain-Trajectory-324** (working).

### Overlap (critical — measured 2026-05-20)
| Set | Count |
|-----|-------|
| C1 ∩ C2 | **18 posts** (3.3% of C1, 5.6% of C2) |

**Implication:** Classification (C1) and divergence/trajectory (C2) are **mostly disjoint cohorts**. The paper must either:
- **(A)** Run trajectory extraction on **all 542 C1 posts** (recommended bridge), creating **C1′** with trajectories; or  
- **(B)** Present RQ2/RQ3 on C2 descriptively and RQ1 on C1 separately, with **18-post bridge** as pilot only (weak—avoid).

**Blueprint decision:** Plan for **C1′ = 542 labeled posts with computed trajectories** as the unified analysis set for RQ2–RQ3; retain C2 as **engagement-rich supplementary** cohort for replication.

### Subreddit filtering strategy
**Problem:** C1 and C2 over-represent **r/transformers** (toys), **r/Bitcoin**, **r/Android**; conflate “software updates” with general tech/hobby.

**Three-tier policy for paper:**
1. **Tier A (strict):** Subreddits clearly about **software/platforms/dev tools** (e.g., rust, linux, django, comfyui, neovim, Wordpress, kubernetes, …) — use allowlist derived from `metric.json` + manual list.  
2. **Tier B (full):** All subreddits in corpus — primary robustness table in appendix.  
3. **Tier C (exclude):** Obvious off-topic (e.g., transformers collectibles, anime) — sensitivity analysis.

**Report all main results on Tier A;** include Tier B in appendix.

### Class imbalance (C1)
- Training: **class weights**, **stratified k-fold**, report **PR-AUC** and **TPS-F1**.  
- Avoid claiming accuracy alone.  
- Optional **undersampling GDS** only inside CV folds for one supplementary table (already done in pilot with 175 GDS)—not the only protocol.

---

## 6. Proposed experimental structure

### Block A — Classification experiments (C1, primary)
| ID | Experiment | Purpose |
|----|------------|---------|
| A1 | **5× stratified CV** on 542 (full text) | Main benchmark |
| A2 | **Held-out test** (15%, stratified, seed=42) | Lock once; match current pilot for comparison |
| A3 | **Majority / random** baselines | Floor |
| A4 | **TF-IDF + Logistic Regression** | Strong classical |
| A5 | **TF-IDF + Multinomial NB** | Reproduce existing |
| A6 | **VADER + technical-cue rules** | Interpretable baseline (existing) |
| A7 | **RoBERTa-base fine-tune** | Transformer (replace under-tuned BERT as primary) |
| A8 | **Optional: DistilBERT** | Efficiency row only if RoBERTa marginal |

### Block B — Divergence experiments
| ID | Experiment | Cohort | Purpose |
|----|------------|--------|---------|
| B1 | Distribution of divergence (author vs community means) | C2 (+ C1′ when ready) | Descriptive |
| B2 | **Author-more-negative rate** by class | C1′ | GDS > TPS? |
| B3 | **Mann–Whitney U** (or Welch t-test) on divergence by TPS/GDS | C1′ | Statistical claim |
| B4 | Effect size (Cohen’s d) + **bootstrap 95% CI** on mean divergence | C1′ | Rigor |
| B5 | Subreddit mixed model / Kruskal-Wallis across top-k subs | C2 | Heterogeneity (RQ4) |

### Block C — Trajectory experiments
| ID | Experiment | Purpose |
|----|------------|---------|
| C1 | Author vs community **slope** (linear trend per comment index) by class | Trajectory difference |
| C2 | **Volatility** (mean |Δcompound|) by class | Emotional dynamics |
| C3 | Early vs late **sentiment shift** by class | Escalation / resolution proxy |
| C4 | **Trajectory feature ablation** for classifier (tabular: mean, slope, vol, divergence) | RQ3 |

### Block D — Ablations (classification)
| ID | Ablation | Features removed |
|----|----------|------------------|
| D1 | Title-only | body + comments |
| D2 | Title + body | comments |
| D3 | Full thread text | — (main) |
| D4 | Full text + trajectory stats | tests incremental value |
| D5 | Subreddit one-hot added | domain leakage check |

### Block E — Statistical validation & human evaluation
| ID | Activity | n | Purpose |
|----|----------|---|---------|
| E1 | **Human spot-check** TPS/GDS on 80 stratified samples | 80 | Precision of labels |
| E2 | **Human divergence rating** (aligned / OP more negative / community more negative) | 80 | Validate VADER divergence |
| E3 | **McNemar / bootstrap** model pair comparisons | — | Significance between best models |
| E4 | **Replicate** prior 25-post sentiment validation | 25 | Cite as motivation only |

---

## 7. Model roadmap

### MUST implement / report
| Model | Role |
|-------|------|
| Majority class | Floor |
| TF-IDF + **Logistic Regression** (balanced) | Primary classical |
| TF-IDF + **Multinomial NB** (balanced) | Existing reproducibility |
| **VADER rule baseline** (`technical_first`) | Interpretable + failure analysis |
| **RoBERTa-base** (fine-tuned, weighted loss) | Primary transformer |
| **5-fold stratified CV** | Main evaluation |

### OPTIONAL (one slot max each)
| Model | When |
|-------|------|
| DistilBERT | If page limit needs efficiency comparison |
| Linear SVM | If LR underperforms unexpectedly |
| XGBoost on trajectory tabular features | If D4 shows promise |

### UNNECESSARY / avoid for main paper
| Item | Why |
|------|-----|
| BERT-base (current) | RoBERTa supersedes; current run underperforms NB on n=40 |
| LSTM/GRU | Only if sequence classifier beats D4; high cost, low prior evidence |
| DeBERTa | Diminishing returns vs RoBERTa for 542 samples |
| Sentence-BERT clustering alone | Supplementary only |
| More VADER variants | One rule baseline sufficient |
| Ensemble stacking | Overkill for n=542; hurts interpretability |
| Live dashboard inference | Not research |

---

## 8. Evaluation roadmap

### Metrics (classification)
- **Primary:** Macro-F1, **TPS-class F1**, **PR-AUC** (TPS positive)  
- **Secondary:** Accuracy, per-class precision/recall, confusion matrix  
- **Calibration (optional):** Brier score on held-out if probability outputs used  

### CV strategy
- **Main:** 5-fold **stratified** CV on C1 (542), report mean ± std  
- **Locked test:** Single 15% stratified holdout (seed=42) — report once all models tuned on train+val only  
- **No repeated peeking** at test set during model selection  

### Confidence intervals
- **Bootstrap 95% CI** (1000 resamples) for: mean divergence, % author-more-negative, key F1 scores on holdout  
- Report CV mean ± std **and** bootstrap CI on holdout metrics  

### Statistical tests
| Comparison | Test |
|------------|------|
| Divergence TPS vs GDS | Mann–Whitney U (non-normal) + Welch t-test (supplementary) |
| Subreddit divergence | Kruskal-Wallis; post-hoc Dunn if significant |
| Model F1 differences | **McNemar** on holdout predictions (paired) |
| Multiple subreddits | Bonferroni or FDR for post-hoc (report conservatively) |

### Significance bar
- α = 0.05; report **p-values** and **effect sizes** (Cohen’s d, Cliff’s delta for Mann–Whitney)  

### Existing pilot (reference only — do not over-cite)
Single split, n_test=40, NB 70% acc / 0.67 macro-F1 — **insufficient for final claims**.

---

## 9. Figure and table roadmap

### Main paper (target 6 figures + 4 tables for 8-page IEEE)

| ID | Type | Content |
|----|------|---------|
| Fig. 1 | Pipeline diagram | Data: ReleaseTrain → text build → labels → models → divergence/trajectory |
| Table 1 | Corpus stats | C1/C2 sizes, class %, overlap, Tier A subreddit count |
| Fig. 2 | Class distribution | Bar: TPS vs GDS + subreddit breakdown (top 10) |
| Table 2 | **Main CV benchmark** | Models × macro-F1, TPS-F1, PR-AUC (mean±std) |
| Fig. 3 | **PR curve** | Best classical vs RoBERTa (holdout) |
| Fig. 4 | Confusion matrix | Best model (holdout) |
| Fig. 5 | **Divergence by class** | Violin/box: divergence TPS vs GDS (C1′) |
| Table 3 | Divergence stats | Means, CI, p-value, Cohen’s d |
| Fig. 6 | **Exemplar trajectories** | 2×2 panel: TPS vs GDS (author vs community lines) |
| Table 4 | Ablation | D1–D5 macro-F1 / TPS-F1 |

### Appendix
- Full subreddit table (Task 4 style)  
- Hyperparameters  
- 10 error examples (FP/FN)  
- Tier B sensitivity table  
- Optional UMAP of misclassified posts  

### Explicitly exclude from paper
- Netlify screenshots, live panel UI, 20 duplicate trajectory PNG grid  

---

## 10. Related work roadmap

### Must cite categories (aim 25–35 references)

1. **Sentiment analysis in SE / software engineering**  
   - Developer emotion in issues, chats, app reviews, release notes  

2. **Reddit / social platform NLP**  
   - Thread structure, conversational sentiment, social media mining  

3. **Technical Q&A and support discourse**  
   - Stack Overflow emotion, help-seeking, problem vs rant classification  

4. **Software update / release / ecosystem risk**  
   - Update failures, dependency breaking, ecosystem intelligence (ReleaseTrain-adjacent)  

5. **Text classification benchmarks & imbalance**  
   - Metrics for imbalanced classification, PR-AUC, class weights  

6. **Transformer fine-tuning for domain text**  
   - RoBERTa, domain adaptation, small-data fine-tuning pitfalls  

7. **Lexicon sentiment (VADER) limitations**  
   - Sarcasm, domain language, title vs body mismatch (supports your 25-post validation)  

8. **Author–audience / OP–community dynamics**  
   - Linguistic alignment, disagreement, conversation structure (if available)  

### Search keywords
`software engineering sentiment`, `reddit thread classification`, `technical support discourse`, `imbalanced text classification`, `release update community`, `author community alignment sentiment`

---

## 11. Threats to validity

| Threat | Description | Mitigation in paper |
|--------|-------------|---------------------|
| **Label subjectivity** | TPS/GDS boundary is interpretive; empty `notes` | Guidelines + E1 human spot-check; report disagreement rate |
| **Candidate label noise** | 24% API vs corrected mismatch | Use corrected only; analyze errors |
| **Class imbalance** | 15.9% TPS | PR-AUC, weighted loss, stratified CV |
| **C1∩C2 overlap tiny** | Only 18 posts | **Compute C1′ trajectories**; do not overclaim joint analysis |
| **Subreddit shift** | C1 ≠ C2 sub distribution | Tier A filter; report both tiers |
| **Off-topic subs** | r/transformers toys, Bitcoin price | Tier A exclude; discussion |
| **VADER construct validity** | 40–56% agreement w/ humans on sentiment | VADER for trajectories only; human E2 on divergence subset |
| **English-only** | Reddit skew | State limitation |
| **Temporal drift** | 2025 snapshot | Timestamp in corpus; no temporal generalization claim |
| **API selection bias** | ReleaseTrain filters | Document fetch URLs and filters |
| **Leakage** | Subreddit in features | D5 ablation; avoid subreddit in main model |
| **Single annotator risk** | If one expert corrected | Limitation + future IAA |
| **Autoreply / bot comments** | Noise in community track | Preprocessing rule (document) |

---

## 12. Final paper structure (ICMLA)

**Assumed length:** 8 pages IEEE + references (confirm ICMLA 2026 author kit).

| § | Section | Subsections | Pages (est.) |
|---|---------|-------------|--------------|
| 1 | **Introduction** | Motivation; RQs; contributions (C1–C4); paper organization | 1.0 |
| 2 | **Related Work** | SE sentiment; Reddit NLP; discourse/support; gap | 1.0 |
| 3 | **Task Definition** | TPS/GDS definitions; annotation; examples; ethics | 0.75 |
| 4 | **Corpus & Methodology** | C1, C2, C1′; text construction; trajectory & divergence metrics | 1.0 |
| 5 | **Experimental Setup** | Models; CV; metrics; statistical tests; human eval | 0.75 |
| 6 | **Results** | 6.1 Classification (A); 6.2 Divergence (B); 6.3 Trajectories (C); 6.4 Ablations (D) | 2.0 |
| 7 | **Discussion** | Implications for update intelligence; VADER limits; practitioner takeaway | 0.5 |
| 8 | **Threats to Validity** | Consolidated | 0.25 |
| 9 | **Conclusion & Future Work** | Dataset release; temporal models | 0.25 |
| — | **References** | | ~1.0 |
| App | Appendix | Hyperparams; subreddit table; extra figures | (optional) |

---

## Implementation priority (ICMLA impact)

| Rank | Task | Impact | Notes |
|------|------|--------|-------|
| 1 | **Annotation guidelines doc** + 20 gold examples | Critical | Enables C1 claim |
| 2 | **DATA_CARD** (provenance, splits, ethics) | Critical | Reviewer trust |
| 3 | **Unified eval harness** (5-fold CV, all baselines, one CLI) | Critical | Replaces ad-hoc scripts |
| 4 | **Trajectory compute on all 542 (C1′)** | Critical | Fixes 18-post overlap blocker |
| 5 | **RoBERTa + LR benchmark** | High | Main Table 2 |
| 6 | **Tier A subreddit allowlist** + sensitivity | High | Validity |
| 7 | **Divergence × TPS/GDS stats + bootstrap CI** | High | C3 claim |
| 8 | **Ablation D1–D5** | High | RQ3 |
| 9 | **Human eval E1–E2 (n=80)** | High | Labels + divergence |
| 10 | **Error analysis table + Fig 6 trajectories** | Medium | Clarity |
| 11 | **Related work bib (25+ cites)** | Medium | Parallel writing |
| 12 | **Holdout lock + McNemar** | Medium | Significance |
| 13 | DistilBERT | Low | Optional row |
| 14 | SHAP/LIME | Low | One figure max |
| 15 | t-SNE/UMAP | Low | Appendix |
| 16 | LSTM/GRU sequences | Avoid | Unless D4 fails |
| 17 | Dashboard/live features | **Avoid** | Zero ICMLA value |
| 18 | More endpoint minScore charts | **Avoid** | Unless tied to corrected_label |
| 19 | Usability-keyword study revival | **Avoid** | Weak construct |
| 20 | Expand to 1000+ labels before write | Medium-long | Do if time; not blocking first draft |

---

## Document control

- **Next author action:** Approve title #1 or #7; approve C1′ strategy; approve Tier A subreddit list.  
- **Next implementation action (when approved):** Items Rank 1–4 only.  
- **Do not start:** RoBERTa training until eval harness + DATA_CARD exist.

---

*This blueprint supersedes exploratory narratives in README/dashboard docs for publication purposes.*
