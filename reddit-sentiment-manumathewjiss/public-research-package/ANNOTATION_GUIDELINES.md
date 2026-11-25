# Annotation Guidelines — TPS vs. GDS (ICMLA 2026)

**Version:** 1.0  
**Task:** Binary discourse classification for software-update-related Reddit threads  
**Labels:** **TPS** (Technical Problem-Solving) = `1` · **GDS** (General Discontent) = `0`  
**Corpus:** ReleaseTrain-TPSGDS-542 (`updated_labeled_dataset_unique.csv`)

---

## 1. Purpose

Annotators assign each Reddit submission (with its discussion context) to one of two **discourse types**. The label describes the **dominant character of the thread text** used for modeling (title, post body, and included comments)—not the poster’s mood alone, and not ReleaseTrain’s separate “tag” taxonomy (Help Request, Discussion, etc.).

These guidelines support:

- Reproducible human review (spot-checks, future IAA studies)
- The ICMLA paper’s **Task Definition** section
- Alignment between `label_current` (API candidate) and `corrected_label` (gold)

---

## 2. Label definitions

### 2.1 TPS — Technical Problem-Solving (`corrected_label = 1`)

**Definition.** The thread is primarily oriented toward **identifying, diagnosing, or resolving a technical issue** related to software, configuration, dependencies, platforms, or updates. Typical signals:

- Concrete failure modes (errors, crashes, regressions after an update)
- Repro steps, logs, versions, environments
- Requests for fixes, workarounds, or “how do I…?” with technical substance
- Maintainer/author replies that troubleshoot rather than only vent

**Not required:** polite tone, successful resolution, or explicitly mentioning “update” (though many TPS posts do).

### 2.2 GDS — General Discontent (`corrected_label = 0`)

**Definition.** The thread is primarily **emotional, rhetorical, social, or general criticism** without sustained technical problem-solving as the main thrust. Typical signals:

- Rants, moral outrage, product hatred without actionable debugging
- Politics, finance hype, lifestyle, fandom unrelated to a fixable software task
- Vague complaints (“this app sucks”) with no technical thread of work
- Memes, jokes, or drama where technology is backdrop not problem domain

**Note:** A post can mention “bugs” in passing and still be GDS if the **dominant** discourse is discontent, not troubleshooting.

---

## 3. Decision rules (apply in order)

1. **Read** title, body/selftext, and top comments (not only the title).
2. If the **main ask** is “how do I fix / why does X fail / what changed after update Y” with technical detail → lean **TPS**.
3. If the **main thrust** is venting, judging, or off-topic debate with little actionable technical content → lean **GDS**.
4. If **mixed**, choose the label that a **triage engineer** would use to route the thread (support queue vs. ignore/monitor).
5. **Sarcasm / humor:** classify by underlying discourse (e.g., joke rant → GDS; joking but detailed bug report → TPS).
6. **API candidate mismatch:** `label_current` is from ReleaseTrain filters (`minScore` / `maxScore`); always record judgment in `corrected_label`. Do not copy API label blindly (24% of corpus disagrees with corrected labels).

---

## 4. Positive examples (TPS)

| ID | Subreddit | Title (abbrev.) | Why TPS |
|----|-----------|-----------------|---------|
| `1o0nzld` | comfyui | After update doesn't work FaceDetailer | Update regression; user seeks fix |
| `1r2myps` | immich | Error since the latest update… | Post-update failure; troubleshooting |
| `1nwebdk` | chrome | Latest Version … introduces bug to Auto Dark Mode… | Version-specific bug report |
| `1oetmbo` | linux | Ubuntu 25.10 Unattended Upgrades Broken Due To Rust Coreutils Bug | Concrete breakage + environment |
| `1ql2670` | comfyui | Re-installed ComfyUI which broke my workflow… how do i fix | Explicit repair request |
| `1nqixzh` | chrome | Chrome freezes … since I upgraded … to MacOS Tahoe | Upgrade-linked failure diagnosis |

---

## 5. Positive examples (GDS)

| ID | Subreddit | Title (abbrev.) | Why GDS |
|----|-----------|-----------------|---------|
| `1nze9rs` | neovim | Plugin store upgrade announcement thread | Product/news discourse; not OP troubleshooting |
| `1misnpf` | Nest | I hate Nest | Affective criticism, not structured debugging |
| `1mlixg3` | Bitcoin | Emergency fund (discussion) | Finance opinion, not software repair |
| `1mhlyz0` | transformers | Worst Transformer names | Fandom opinion, not software-update problem-solving |
| `1mul6k9` | union | Fight The Republican War On Workers | Off-domain politics |
| `1mid4hg` | Wordpress | Is blogging dead? | Strategy discussion, not defect resolution |

*(Titles taken from corpus; annotators should verify URL context when re-labeling.)*

---

## 6. Negative examples (common mistakes)

| Mistake | Wrong label | Correct | Reason |
|---------|-------------|---------|--------|
| Title contains “error” | TPS | GDS | “Domain error?” on GitHub was a neutral help question; word ≠ class |
| Title sounds angry | GDS | TPS | “I AM CRASHING OUT” on Godot may still be crash/debug context—read body |
| High upvotes / long thread | TPS | GDS | Engagement ≠ technical discourse |
| ReleaseTrain `label_current=1` | TPS | GDS | e.g. white borders / auto-update rant corrected to GDS in corpus |
| Pure release notes | TPS | GDS | Announcements without OP problem-solving → usually GDS |

---

## 7. Edge cases and ambiguous cases

| Situation | Guidance |
|-----------|----------|
| **Product review** (mixed praise/complaint) | TPS if focused on reproducible defect; GDS if general opinion |
| **“Is X dead?”** | Usually GDS (meta/discussion) unless technical migration failure |
| **Corporate / layoff / RTO** | GDS unless specific IT system failure dominates |
| **Dependency hell** | Usually **TPS** (technical constraints, versions, fixes) |
| **Crypto price / ideology** | **GDS** unless wallet/software bug is central |
| **Toy/collectible subreddits** | Often **GDS** for paper’s software-update focus; mark for Tier-A filtering |
| **Solved thread** | Still TPS if discourse was problem-solving throughout |
| **Only title available** | Defer annotation until body/comments fetched |

Record residual ambiguity in `notes` column (recommended for future passes; currently sparse in v1.0 export).

---

## 8. Labeling workflow

1. **Open** post URL from `updated_labeled_dataset_unique.csv`.
2. **Read** title → body → comments in chronological order (focus on OP and early replies).
3. **Assign** `corrected_label`: `1` = TPS, `0` = GDS.
4. **Optional:** Short `notes` (e.g., “sarcasm”, “title misleading”, “borderline TPS”).
5. **Do not change** `reddit_id`; keep one row per post.
6. **Spot-check batches:** 10% dual review when second annotator available (future IAA).
7. **Export** deduplicated CSV (`reddit_id` unique) before training/evaluation.

**Text used by models:**

- **Cleaned:** `text` in `tps_gds_dataset.json` (URLs stripped, lowercased) — default for TF-IDF models.
- **Raw:** `text_raw` (title + body + top comments by score) — VADER rule baseline.

---

## 9. Relationship to ReleaseTrain API candidates

| Pool | API filter | Candidate meaning |
|------|------------|-------------------|
| TPS candidates | `minScore=0.3` | Higher-scored threads (proxy, not gold) |
| GDS candidates | `maxScore=0.5` | Lower-scored threads (proxy, not gold) |

`label_current` in the CSV reflects API pool membership hints; **`corrected_label` is the gold label** for all paper experiments.

---

## 10. Limitations of manual labeling

- **Single annotator** for much of the corpus (inter-annotator agreement not yet reported).
- **Subjective boundary** between frustration (can be TPS) and rant (GDS).
- **Snapshot bias:** labels reflect one point in time; threads may evolve.
- **English-only** implicit assumption.
- **Subreddit prior:** annotators may recognize communities (r/rust vs r/Bitcoin)—mitigate by reading text each time.
- **Comment sampling:** dataset `text_raw` includes top comments by score, not full trees—labels are “thread sample” not guaranteed full-thread gold.

---

## 11. Version history

| Version | Date | Change |
|---------|------|--------|
| 1.0 | 2026-05-20 | Initial ICMLA infrastructure release; aligned with RESEARCH_BLUEPRINT_ICMLA2026.md |

**Maintainer contact:** SE4CPS / ReleaseTrain research team (internal).
