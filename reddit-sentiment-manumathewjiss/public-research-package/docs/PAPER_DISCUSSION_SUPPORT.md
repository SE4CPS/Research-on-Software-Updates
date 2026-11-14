# Paper Discussion Support (ICMLA 2026)

Evidence-based interpretation for Results and Discussion. **Do not overclaim** — all points below are tied to reported numbers in `evaluation/outputs/` and `analysis/outputs/`.

---

## 1. Title-only outperforms full thread (D1 vs D3)

**Finding:** D1 title-only macro-F1 **0.758** vs D3 full-thread **0.632** (full C1); Tier-A: **0.695** vs **0.622**.

**Plausible explanations (defensible):**

1. **Intent is front-loaded.** Reddit update posts often state the problem or stance in the title; comment threads add noise (jokes, tangents, duplicate advice).
2. **TF-IDF dilution.** Full-thread bags mix OP, commenters, and boilerplate; rare TPS cues (`error`, `crash`, `how do I`) get diluted by high-frequency community text.
3. **Label boundary aligns with title framing.** TPS/GDS distinction is partly *discourse goal* (fix vs vent), which authors signal early; body/comments may look “technical” even for GDS rants.
4. **Small-data regularization.** Simpler features (title) reduce overfitting vs high-dimensional sparse full text on n=542.

**What we do *not* claim:** titles alone are sufficient for all software discourse — only that, *for this corpus and task*, they are the strongest single signal.

---

## 2. TPS shows stronger author–community divergence than GDS

**Finding (C1′, VADER-based):**

| Cohort | n | Mean \|author − community\| | TPS − GDS Δ | MW p |
|--------|---|-----------------------------|-------------|------|
| Full   | 86 TPS / 456 GDS | 0.371 / 0.287 | **+0.084** | 0.009 |
| Tier-A | 53 TPS / 251 GDS | 0.412 / 0.307 | **+0.105** | 0.012 |

Also: **69.8%** (full) of TPS authors are more negative than their thread vs **53.7%** for GDS (p≈0.006).

**Interpretation:**

- **Troubleshooting posts** pair a frustrated OP with a more neutral/helpful comment stream → measurable sentiment gap.
- **General discontent** threads often share a negative tone across author and community (pile-on, agreement) → smaller gap.
- Effect **persists and slightly strengthens** on Tier-A → not an artifact of Bitcoin/transformers toy subs.

**Limitation:** Divergence uses **VADER** on short Reddit snippets; magnitude is a proxy, not ground-truth emotion. Human spot-check (`human_divergence_rating`) will validate buckets.

---

## 3. Trajectory features help TPS-F1 (D4) but not macro-F1

**Finding:** D4 full+trajectory TPS-F1 **0.423** vs D3 **0.352** (full); macro-F1 roughly flat (0.647 vs 0.632).

**Interpretation:**

- Trajectory captures **dynamic misalignment** (author trend vs community trend) that static bag-of-words misses — especially for borderline TPS.
- GDS majority class limits macro-F1 gains; TPS-F1 is the right metric to show trajectory value.
- On Tier-A, D4 TPS-F1 (0.337) ≈ D3 (0.348) — trajectory benefit may be subset-specific; report both.

---

## 4. Naive Bayes beats RoBERTa (and often LR) on TPS-F1

**Finding (5-fold OOF):**

| Model | Macro-F1 | TPS-F1 | TPS recall |
|-------|----------|--------|------------|
| Naive Bayes | **0.659** | **0.463** | **0.62** |
| Logistic Reg | 0.632 | 0.352 | 0.28 |
| RoBERTa | 0.591 | 0.307 | 0.36 |
| VADER rules | 0.409 | 0.330 | 0.90 |

**Error analysis:**

- **NB:** 91 FP / 33 FN — aggressive TPS recall; confuses GDS with technical title words (`update`, `error`, `fix`).
- **RoBERTa:** 57 FP / **55 FN** — fewer false alarms but **misses half of TPS**; errors on short, sarcastic, or news-style titles without clear “help” phrasing.

**Why NB wins here (honest):**

1. **n=86 TPS** is tiny for fine-tuning transformers; NB + TF-IDF exploits high-precision lexical cues.
2. **RoBERTa setup is minimal** (2 epochs, max_len 256) — not a SOTA transformer baseline.
3. **Class imbalance** — NB with class weighting favors TPS recall; RoBERTa under-predicts minority class.
4. **Domain mismatch** — pretrained on general text; Reddit update jargon (`comfyui`, `immich`, distro names) is sparse.

**Held-out test (n=82, seed=42):** LR macro-F1 **0.685** > NB **0.612** (McNemar p=0.027) — suggests CV advantage for NB is partly fold variance; report both CV (primary) and held-out (robustness appendix).

---

## 5. VADER limitations motivate richer analysis

**Finding:** VADER rules macro-F1 **0.409** with **90% TPS recall** but **20% precision** — labels almost everything upset as TPS-adjacent.

**Use in paper:**

- VADER is **inadequate for TPS/GDS classification** → justifies learned models and human correction of 24% of API labels.
- VADER remains **useful for coarse trajectory/divergence** where we compare *relative* author vs community tone within a thread, not absolute sentiment.
- Lexicon bias: misses sarcasm, memes, technical neutrality (“update broke X” reads negative but is factual).

---

## 6. Noisy Reddit discourse & subreddit skew

**Evidence:**

- Top FP subreddits (NB): **chrome** (23), Wordpress, immich — genuine software subs with ambiguous “update” titles.
- Top FN subreddits: **transformers** (6), **Bitcoin** (5), Android — fandom/crypto/news disguised as update talk.
- Tier-A filter **removes** much of this noise; NB macro-F1 rises **0.659 → 0.703**.

**Discussion angle:** Findings reflect **software-update-oriented Reddit**, not all social media. Tier-A analysis shows core patterns survive subreddit cleaning.

---

## 7. Connection to software-update communities

| Observation | Community behavior link |
|-------------|-------------------------|
| Title-only wins | Users encode update pain in headlines; communities respond asynchronously in comments. |
| TPS divergence | Help-seekers vent; commenters troubleshoot calmly → emotional misalignment. |
| GDS lower divergence | Shared outrage or announcement tone → aligned negativity or neutrality. |
| NB lexical bias | Update vocabulary is repetitive across subs (`broken`, `patch`, `rollback`). |
| Trajectory helps TPS | OP may soften or escalate across replies while crowd stays stable. |

---

## 8. Suggested Discussion paragraphs (adapt for paper)

**Classification.**  
Learned linear models outperform rule-based VADER on TPS/GDS discrimination, with Naive Bayes achieving the best trade-off between minority-class recall and macro-F1 under 5-fold cross-validation. Fine-tuned RoBERTa did not surpass simpler baselines, consistent with limited TPS examples and domain-specific phrasing. Ablation D1 shows that post titles alone carry most of the discriminative signal, suggesting that full-thread modeling introduces noise rather than complementary intent cues for this task.

**Divergence.**  
Using C1′ author trajectories, TPS threads exhibit significantly larger author–community sentiment separation than GDS threads (Δ≈0.08–0.11, p<0.02), including a higher rate of authors being more negative than their surrounding comment stream. This pattern aligns with troubleshooting behavior: frustrated posters seeking fixes amid comparatively neutral peer advice. The effect replicates on a Tier-A software-subreddit subset, arguing against confounding from unrelated communities.

**Limitations.**  
Labels reflect a single corrected annotation pass without full inter-annotator agreement on all 542 posts. VADER-based divergence is a proxy measure. Results are associative, not causal. Generalization beyond Reddit software-update discourse is untested.

---

## 9. Threats to Validity (bullet list)

- Subjective TPS/GDS boundary (rant vs help overlap).
- 15.9% TPS prevalence → high variance on minority metrics.
- Human validation (80 posts) pending — κ not yet computed.
- RoBERTa under-tuned; not an upper bound on transformer performance.
- ReleaseTrain/API sampling may skew toward volatile update periods.
- English-only, Reddit-specific norms (irony, memes).
