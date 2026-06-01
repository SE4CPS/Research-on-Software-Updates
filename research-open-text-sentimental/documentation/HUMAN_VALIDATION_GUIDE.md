# Human Validation Guide (80-post spot check)

**File to fill:** `data/human_eval/human_eval_template_80.csv`  
**After labeling:** `python3 scripts/analyze_human_eval.py`

---

## How to label (5 minutes per post)

1. Open the **url** in a browser.
2. Read **title + first ~5 comments** (30–60 seconds).
3. Fill:
   - **`human_label_tps_gds`:** `TPS`, `GDS`, or `Unclear`
   - **`human_divergence_rating`:** `aligned`, `author_more_negative`, `community_more_negative`, or `unclear`
   - **`human_notes`:** one short phrase if tricky

**Do not change** `gold_corrected_label` — that is the dataset gold for agreement.

---

## TPS vs GDS (quick rules)

| Choose **TPS** if… | Choose **GDS** if… |
|--------------------|---------------------|
| Main goal is fix a bug, error, install, config | Main goal is rant, opinion, news, meme |
| “How do I…?” with technical detail | “I hate…”, drama, politics, fandom |
| Update broke something and user wants help | Release announcement with no OP problem |

---

## Ambiguous posts in your sample (review carefully)

These rows from the template are **known edge cases** — good for Cohen’s κ:

| reddit_id | Why ambiguous |
|-----------|----------------|
| `1qhs7lf` | Transformers **toy** rant title (“SCRAP”) but gold **TPS** — likely API candidate noise |
| `1pp9s46` | Praise for WINE devs — positive news, gold **GDS** |
| `1pf32mo` | Microsoft update **news**, gold **TPS** — announcement vs troubleshooting |
| `1nqu3ag` | “Failed experiment” — technical blog tone, gold **TPS** |
| `1r7chxn` | Toy display cabinet — clearly **GDS** (non-software) |
| `1nuwiid` | Chrome playback bug — looks **TPS** but gold **GDS** (check body) |
| `1nznnb8` | ComfyUI weird image edit — help request vs casual, gold **GDS** |

If you disagree with gold, note it in `human_notes` — that is valuable for Threats to Validity.

---

## Divergence rating

Compare **OP mood** vs **commenters** (not title alone):

- **aligned** — similar tone overall  
- **author_more_negative** — OP more frustrated/upset than crowd  
- **community_more_negative** — crowd harsher (rare)  
- **unclear** — too short or sarcasm

Compare your rating to `vader_author_more_negative` and `vader_divergence` after running the analysis script.

---

## Target for the paper

- **Label agreement ≥ 80%** with gold → supports corpus quality  
- **κ ≥ 0.60** → “substantial” (Landis & Koch)  
- **Divergence agreement ≥ 65%** with VADER buckets → trajectory claims defensible  

If κ is lower, report honestly and cite ambiguous boundaries (TPS/GDS, sarcasm).
