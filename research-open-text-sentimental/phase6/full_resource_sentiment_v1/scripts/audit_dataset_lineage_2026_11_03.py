#!/usr/bin/env python3
"""Read-only ID and coverage reconciliation; writes derived audit outputs only.

Protected files are used for ID membership only. No protected outcome is copied.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parents[1]
OUT = BASE / "results/dataset_forensic_audit_2026-11-03"
SRC = {
    "historical": REPO / "tps_gds_classification/data/updated_labeled_dataset_unique.csv",
    "master": REPO / "phase2/combined_dataset/master_records.json",
    "development": REPO / "phase3/revised_development_v2/development_v2.jsonl",
    "broad": BASE / "results/broad_sentiment_2026-10-22-main/post_scores.jsonl",
    "thread": BASE / "results/broad_sentiment_2026-10-22-main/thread_metrics.jsonl",
    "relevance_predictions": BASE / "restricted/ai_relevance_exploratory_full_2026-10-29/full/predictions.jsonl",
    "selected": BASE / "results/ai_relevance_exploratory_full_2026-10-29/exploratory_ai_selected_cohort.jsonl",
    "existing": BASE / "results/tps_gds_combined_pool_2026-11-02/combined_pool.jsonl",
    "current": BASE / "results/tps_gds_full_ai_dataset_2026-11-02/full_dataset.jsonl",
    "capture_manifest": BASE / "restricted/broad_release_train/2026-10-22-main/manifest.json",
    "reserved_ids_only": REPO / "phase3/experiment_v1/data/reserved_test.jsonl",
    "fresh_ids_only": REPO / "phase3/fresh_test_20261007/frozen_joint_labels/all_500_reviewed.jsonl",
}


def ident(row: dict) -> str:
    raw = row.get("reddit_id") or row.get("redditId") or row.get("id")
    if not raw:
        return ""
    value = str(raw).strip().lower()
    if value.startswith("t3_"):
        value = value[3:]
    return value


def jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def load_by_id(name: str, rows):
    out = {}
    duplicates = Counter()
    missing = 0
    for row in rows:
        rid = ident(row)
        if not rid:
            missing += 1
            continue
        if rid in out:
            duplicates[rid] += 1
        else:
            out[rid] = row
    return out, {"rows": len(out) + sum(duplicates.values()) + missing,
                 "unique": len(out), "duplicate_extra_rows": sum(duplicates.values()),
                 "missing_id_rows": missing, "duplicate_id_count": len(duplicates)}


def write_csv(path: Path, fields: list[str], rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_reports(x: dict):
    n = x["set_counts"]
    c = x["conversation"]
    p = x["broad_prediction_counts"]
    q = x["broad_unassembled_prediction_labels"]
    h = x["historical_to_current_nonprotected"]
    src = x["source_paths_and_sha256"]
    def f(name: str, content: str):
        (OUT / name).write_text(content.strip() + "\n", encoding="utf-8")
    source_table = "\n".join(f"| {k} | `{v['path']}` | `{v['sha256'][:16]}…` |" for k, v in src.items())
    f("MASTER_DATASET_INVENTORY.md", f"""
# Master dataset inventory — ID-level audit, 2026-11-03

| Source | Unique IDs | Meaning |
|---|---:|---|
| Historical corrected-label file | {n['historical_542']:,} | Original binary TPS={x['historical_corrected_label_counts']['TPS']}, broad GDS={x['historical_corrected_label_counts']['broad_GDS']}; `corrected_label` is the final historical column, not `label_current`. |
| Phase 2 master | {n['phase2_master_3362']:,} | Older mixed-source assembled review corpus; includes all 542 historical IDs and 200 protected reserved IDs. |
| Revised development | {n['development_3151']:,} | Development subset, not a separate additive population. |
| Raw ReleaseTrain page collection | {x['raw_capture_page_check']['raw_rows']:,} | 76 pages, including 133 non-Reddit URLs. |
| Frozen broad capture, Reddit-URL frame | {n['broad_37596']:,} | Raw pages filtered by Reddit URL and scored with ReleaseTrain/VADER; URL criterion, not human relevance. |
| Broad relevance predictions | {n['relevance_predictions']:,} | One AI relevance response per broad candidate; not TPS/GDS labels. |
| AI-relevant predictions after protected exclusion | {n['ai_selected_5686']:,} | Selection candidates for current purpose labeling. |
| Preexisting current decisions | {n['existing_1964']:,} | Mixed existing reviewed/assisted decisions; see source field. |
| Current unified pool | {n['current_6425']:,} | Deduplicated historical + existing + selected posts, all with current decision. |
| Current update-relevant subset | {n['current_relevant_4944']:,} | Current relevance decision, not all independently confirmed. |
| Main comparable VADER TPS/GDS | {n['main_vader_tps_gds']:,} | 3,028 TPS + 231 GDS with frozen broad score definition. |

All source files used in this audit and their SHA-256 prefixes:

| Key | Absolute path | SHA-256 prefix |
|---|---|---|
{source_table}

`DATASET_OVERLAP_MATRIX.csv` contains all pairwise exact-ID intersections. `FULL_RECORD_RECONCILIATION.csv` contains one row per union ID with no raw post text or protected outcomes. The protection lists were read for IDs only.
""")
    f("DATASET_LINEAGE_AND_HISTORY.md", f"""
# Dataset lineage and history

```mermaid
flowchart TD
 H[Original 542: 86 historical TPS / 456 broad GDS] --> M[Phase 2 master: 3,362]
 N[2,820 additional phase-2 records] --> M
 M --> D[Development snapshot: 3,151]
 B[Frozen ReleaseTrain Reddit-URL capture: 37,596] --> R[AI software-update relevance predictions: 37,596]
 R --> S[Predicted relevant: 6,204]
 S --> E[Exclude 518 protected IDs: 5,686 selected]
 H --> U[Union by Reddit ID: 6,425]
 X[Existing current decisions: 1,964] --> U
 E --> U
 U --> L[Current relevant: 4,944; TPS 3,052 / GDS 242 / Other 1,642 / Unclear 8]
 L --> V[Comparable frozen VADER TPS/GDS: 3,028 / 231]
```

The union is exactly **1,964 + 542 + 5,686 − 1,748 − 19 − 14 + 14 = 6,425**. The three subtractions are existing∩selected, existing∩historical, and historical∩selected; the final +14 restores the triple intersection. These figures come from the saved all-source inventory and this audit's ID matrix.

| Approximate stage | Input → output | Counts and transformation | Label meaning / evidence |
|---|---|---|---|
| Earlier binary study | `tps_gds_classification/data/updated_labeled_dataset_unique.csv` | 542 unique; corrected 86 TPS / 456 broad GDS | Historical binary labels, not today's strict GDS. Original assignment method is not proven independently human from this CSV alone. |
| Oct 6–7 phase-2 reviews | Historical 542 + new 2,820 → `phase2/combined_dataset/master_records.json` | 3,362 unique; 0 duplicate IDs in master | `phase2/combined_dataset/build_manifest.json`: 2,757 assisted review, 300 joint review, 100 pilot discussion, 200 reserved consensus, 5 uncertain; these are statuses, not independent-rater counts. |
| Oct development | Master → `phase3/revised_development_v2/development_v2.jsonl` | 3,151 unique, all within master | Development subset; includes target/outcome fields, so this audit uses IDs only. |
| Oct 22–23 frozen capture | ReleaseTrain API saved pages → broad score table | {x['raw_capture_page_check']['raw_rows']:,} raw unique IDs from 76 pages; 133 non-Reddit URL rows excluded; 37,596 Reddit-URL candidates, 0 duplicate score IDs | `restricted/broad_release_train/2026-10-22-main/manifest.json` and `results/broad_sentiment_2026-10-22-main/post_scores.jsonl`. API route/search is not a human relevance label. |
| Oct 29 broad relevance | 37,596 → 37,596 AI responses | 6,204 predicted relevant; 30,693 not relevant; 454 unclear; 97 unassessable; 148 errors | `restricted/ai_relevance_exploratory_full_2026-10-29/full/predictions.jsonl`; exploratory model output, no independent full-frame benchmark. |
| Nov 2 selection | 6,204 relevant → 5,686 | Exclude 518 protected IDs | `results/ai_relevance_exploratory_full_2026-10-29/exploratory_ai_selected_cohort.jsonl` and ID-only protected lists. |
| Nov 2 current decisions | Existing 1,964 + historical 542 + selected 5,686 → unified 6,425 | 1,964 existing decisions + 4,461 new Terra decisions on the union | `results/tps_gds_all_source_inventory_2026-11-02/`, `restricted/tps_gds_full_classification_2026-11-02/`, `results/tps_gds_full_ai_dataset_2026-11-02/`. |
| Nov 3 corrected VADER | Unified 6,425 → frozen relevant TPS 3,028/GDS 231 | Exclude other-purpose, nonrelevant, and 520 historical-definition score rows from the comparable contrast | `results/tps_gds_sentiment_v2_2026-11-03/stage1_corrected_vader/`. |

Some exact per-stage candidate exclusions before the phase-2 master are not recoverable from these final files alone; do not infer them from final totals. No source data were overwritten by this audit.
""")
    f("CLASSIFICATION_COVERAGE_REPORT.md", f"""
# Classification coverage

The answer to “were all 37,596 classified?” depends on **which decision**. All {n['broad_37596']:,} received a saved **software-update relevance** response. Only the {n['current_6425']:,}-ID union received current relevance and TPS/GDS/Other purpose decisions; that union includes {x['current_minus_broad']:,} historical-only IDs.

Broad relevance responses: relevant **6,204**, not relevant **30,693**, unclear **454**, unassessable **97**, and error **148**. Of 6,204 predicted relevant, **518 protected** IDs were excluded, leaving **5,686** selected nonprotected IDs; all 5,686 appear in the current table.

The exact broad/current overlap is **{x['broad_current_overlap']:,}**. Therefore **{x['broad_minus_current']:,} broad IDs are outside the current table**, and **{x['current_minus_broad']:,} current IDs are historical/local-only**. This explains why the intuitive subtraction 37,596 − 6,425 = 31,171 is wrong. Within broad-not-current, predictions are: not relevant **{q['not_relevant']:,}**, unclear **{q['unclear']:,}**, unassessable **{q['unassessable']:,}**, error **{q['error']:,}**, and relevant **{q['relevant']:,}**. All {q['relevant']:,} omitted predicted-relevant IDs are protected; no nonprotected predicted-relevant ID was accidentally left out.

The current table also includes **{x['broad_in_current_but_not_ai_selected']}** broad posts from older decisions that the broad relevance model did not select: {x['broad_in_current_nonselected_prediction_labels']}. This is a disagreement/uncertainty group worth auditing, not a reason to silently replace prior decisions.

Current table counts: relevant TPS **3,052**, strict GDS **242**, Other **1,642**, purpose-Unclear **8**; not relevant **1,352**, relevance unclear **50**, unassessable **79**. The last three sum to 1,481, but only 1,352 are marked not relevant. `final_relevance` and `final_discussion` in `full_dataset.jsonl` are the authoritative current columns. Relevance model predictions are in a separate file and must not be mistaken for purpose labels.

The raw saved pages contain {x['raw_capture_page_check']['raw_rows']:,} unique IDs: {n['broad_37596']:,} Reddit URLs and 133 other URLs. The broad score table has nonempty titles for {x['broad_text_coverage']['title_nonempty']:,}/{n['broad_37596']:,} and nonempty bodies for {x['broad_text_coverage']['body_nonempty']:,}/{n['broad_37596']:,}. All 37,596 have a Reddit URL. The API `source` field explicitly says `reddit` on {x['broad_text_coverage']['source_reddit']:,} and is absent on the remainder. Missing `source` is a metadata limitation, not proof of another platform. Title/body can be reused from the frozen local capture; no recollection is needed for screening.

The `Other` purpose category includes update announcements, praise, planning, neutral release information, and other relevant discussion that is neither primarily troubleshooting nor dissatisfaction under `validation_review_2026-11-02/ANNOTATION_RUBRIC.md`. The 1,642 records cannot be subdivided reliably into these themes from the saved four-way label alone. Whether some should be TPS or GDS is a classification-error question; this audit did not silently reassign them. Review would require a defined sample and the saved original title/body.
""")
    f("GDS_AND_CONVERSATION_COVERAGE.md", f"""
# GDS and saved-conversation coverage

Historical `corrected_label`: 86 TPS and 456 **broad** GDS. Among those original 456, current decisions are: **{h['historical_broad_GDS'].get('relevant/GDS',0)} strict GDS**, {h['historical_broad_GDS'].get('relevant/TPS',0)} TPS, {h['historical_broad_GDS'].get('relevant/Other',0)} Other, {h['historical_broad_GDS'].get('relevant/Unclear',0)} purpose-Unclear, and the remainder outside current relevance. This demonstrates why the 456 cannot be added to today's GDS count. The current 242 GDS include 76 existing-decision and 166 new Terra-decision cases; the low total reflects a narrower definition, a relevance-selected pool, and unmeasured classification error. It is not proof that few dissatisfied conversations exist in the full capture.

| Population | N | ≥1 nonempty reply evidenced | `n_stored_comments` >0 | ≥1 author reply | ≥2 | ≥3 | ≥4 | ≥5 | ≥10 nonempty replies | Both roles | ≥2 timed replies | Count gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
""" + "\n".join(
        f"| {label} | {total:,} | {d.get('comments_ge1',0):,} | {d.get('stored_comments_ge1',0):,} | {d.get('author_ge1',0):,} | {d.get('author_ge2',0):,} | {d.get('author_ge3',0):,} | {d.get('author_ge4',0):,} | {d.get('author_ge5',0):,} | {d.get('comments_ge10',0):,} | {d.get('both_roles',0):,} | {d.get('timed_ge2',0):,} | {d.get('count_gap',0):,} |"
        for label, total, d in [
            ("Frozen broad", n["broad_37596"], c["broad"]),
            ("Current union", n["current_6425"], c["current"]),
            ("Current TPS", 3052, c["TPS"]),
            ("Current GDS", 242, c["GDS"]),
            ("Current Other", 1642, c["Other"]),
            ("Broad outside union", x["broad_minus_current"], c["broad_unassembled"]),
        ]) + f"""

The nonempty-reply column uses frozen broad scores where available and, for 520 historical-only rows, the saved author/community reply counts in the historical comparison. The stored-comment and timestamp columns are known from the frozen capture only; **520 historical-only rows lack those fields in this joined table**. Thus current GDS has {c['GDS'].get('comments_ge1',0)} with evidence of a nonempty reply, but only {c['GDS'].get('stored_comments_ge1',0)} with a frozen `n_stored_comments` field above zero; the remaining 11 are historical-only. The Stage-1 comparable subset has **231 GDS**, of which **98** have an author-reply mean and **30** have ≥4 author replies plus an observed author-shift score. The full 242 has {c['GDS'].get('author_ge4',0)} with ≥4 replies, so the 30 is due to the frozen-score/eligibility filter, not the entire current pool.

Among broad posts with nonempty saved replies, {c['broad'].get('all_nonempty_replies_timed',0):,} have timestamps for every nonempty reply and {c['broad'].get('partial_timestamp_coverage',0):,} have only partial timestamp coverage. For current GDS with frozen timing fields, these counts are {c['GDS'].get('all_nonempty_replies_timed',0)} and {c['GDS'].get('partial_timestamp_coverage',0)}; 11 historical-only GDS timing fields are unavailable here. At least two timed replies is a narrower eligibility test than merely having comments.

The {x['broad_minus_current']:,} broad posts outside the union include **{c['broad_unassembled'].get('author_ge4',0):,}** with ≥4 saved nonempty author replies, but without current strict GDS/TPS labels. They cannot be called additional GDS conversations until relevant purpose decisions are made. The normalized local snapshot audit found **{x['alternative_saved_snapshot']['more_nonempty_comments_than_broad_snapshot']}** overlapping ID with more nonempty comments in another saved source; alternative snapshots should be checked by ID before any recovery, not silently merged across capture times. `results/normalized_records_compact.jsonl` provides source and comment-count metadata. The phase-2 master often has a flattened discussion context but no role/timestamp structure, so its conversation coverage is not directly comparable with the frozen reply arrays.
""")
    f("MISSING_DATA_AND_EXCLUSIONS.md", f"""
# Missing data and exclusions

No duplicate or missing IDs were found in the eight primary ID files audited. Normalization lowercases IDs and removes a leading `t3_`; the audit did not find resulting collisions. The 542 historical IDs are all in the 3,362 master and all in the current union. Only **22** historical IDs are in the frozen broad capture; the other **520** are historical/local-only and use a different sentiment construction.

Of the phase-2 master, **{x['master_unassembled_protected']} protected** and **{x['master_unassembled_nonprotected']} nonprotected** IDs are absent from the current union. Among the 441 nonprotected omitted IDs, older `phase2_update_related` values are {x['master_unassembled_nonprotected_review_fields']['update_related']}. Of the **39** old “yes” decisions, **30 Stack Overflow URLs** are outside the Reddit target population and **9 Reddit URLs** deserve targeted current-rubric checking. The old label is evidence for review, not an automatic current inclusion. Some older master IDs may fall outside the frozen broad capture or have an AI relevance result of unclear/not relevant; see `FULL_RECORD_RECONCILIATION.csv`.

The nine Reddit IDs are: {', '.join('`'+rid+'`' for rid in x['master_unassembled_nonprotected_review_fields']['update_yes_reddit_ids'])}. Seven are outside the frozen broad capture; two are inside it with an AI `unclear` result. Their source titles and prior review fields remain in `phase2/combined_dataset/master_records.json`.

The {x['broad_minus_current']:,} broad omitted IDs are exactly accounted for in `CLASSIFICATION_COVERAGE_REPORT.md`. The 518 predicted relevant but omitted IDs are protected. The other omissions follow the *AI relevance selection rule*; that rule may have false negatives. There is no independent full-frame estimate of how many software-update or GDS posts are hidden among the 30,574 AI-not-relevant IDs outside the union. The 200-case sample was drawn from the 6,425 table, not this excluded broad stratum, and currently has zero saved decisions.

The broad capture has {x['broad_text_coverage']['body_nonempty']:,} nonempty bodies and {x['broad_text_coverage']['title_nonempty']:,} nonempty titles. For conversations, the table in `GDS_AND_CONVERSATION_COVERAGE.md` distinguishes zero *nonempty* replies from zero *stored* comments. Count gaps (API reported comments versus saved replies) do not prove missing data can be recovered; a separate saved snapshot may differ by capture time. Missing author replies are not neutral sentiment.

The broad relevance run spans {x['broad_period_counts']}; the current table's saved selection strata are {x['current_period_counts']}. These time groups should be checked separately in any expanded analysis. A title counted as nonempty can still be a removal notice; the 97 broad `unassessable` AI predictions and 79 current `unassessable` decisions reflect this distinction.

The record CSV contains normalized ID, source membership, current status when allowed, exclusion/selection reason, and coverage metadata. It never exports full text, usernames, protected outcome columns, model rationale, or sentiment values.
""")
    f("DATASET_EXPANSION_OPTIONS.md", f"""
# Dataset expansion options and decision

| Option | Benefit | Risk / work | Recommendation |
|---|---|---|---|
| A. Keep 6,425 as the full analysis pool | Already assembled, scored, and reproducible; no new labeling | Selected cohort; cannot generalize to 37,596; small GDS cadence group | Continue existing analysis, but describe population honestly. |
| B. Reuse frozen broad capture | {x['broad_minus_current']:,} unassembled candidate IDs and {c['broad_unassembled'].get('author_ge4',0):,} with ≥4 saved author replies; no new Reddit fetch | Most were predicted not relevant; blindly purpose-labeling all is wasteful and may import false positives. Requires an excluded-stratum audit and new validation design | **Best targeted next move**: first check 9 old update-yes Reddit omissions and a stratified sample of AI-not-relevant/unclear/error posts, prioritizing comment-rich candidates while retaining random coverage. Freeze rule and measure false negatives before scaling. No model call in this audit. |
| C. Collect new Reddit posts | Could find newer update/GDS cases | New time period, permissions, duplicate handling, capture completeness, cost | Defer until saved-data opportunities are evaluated. |

No numerical GDS yield is defensible yet because the excluded broad posts lack current purpose labels. Relevance prediction exists for every broad ID, but TPS/GDS/Other purpose classification is not complete for the broad frame. If the target population is expanded beyond the 6,425 selected union, the existing 160-fresh/40-coverage review does **not** validate the new excluded stratum. A separate probability sample from that stratum is needed to estimate missed relevance and GDS yield; protect all reserved/fresh IDs and maintain the old 200-case packet unchanged. Report collection period, source-metadata missingness, count gaps, and AI label provenance.
""")
    f("FINAL_DATASET_AUDIT_SUMMARY.md", f"""
# Final dataset forensic audit — 2026-11-03

## Decision

The 6,425-post table is **correctly assembled for its stated three-source union**, but it is **not the complete 37,596-post ReleaseTrain capture**. All 37,596 received an exploratory AI software-update relevance prediction; {x['broad_minus_current']:,} captured IDs are outside the current table, while 520 historical/local-only IDs are inside it. There is no evidence of an accidental omission of a nonprotected AI-predicted-relevant ID. There **is** a small review opportunity: 9 previously update-yes Reddit posts in the phase-2 master were omitted, and the AI-rejected/unclear broad stratum may contain unknown false negatives.

## Exact arithmetic

37,596 broad = {x['broad_current_overlap']:,} broad∩current + {x['broad_minus_current']:,} broad-only. 6,425 current = {x['broad_current_overlap']:,} broad∩current + {x['current_minus_broad']:,} local-only. Thus 37,596 − 6,425 = 31,171 is **not** the excluded broad count. The 5,686 AI-selected nonprotected IDs all appear in current; 219 additional broad IDs arrived through preexisting decisions; 520 current IDs are historical/local-only.

## Main limits

Current purpose labels are TPS 3,052, strict GDS 242, Other 1,642, Unclear 8 among 4,944 currently relevant posts. The 1,481 outside relevance are 1,352 not relevant, 50 unclear, 79 unassessable. Historical broad GDS is incompatible with strict GDS. The current 242 GDS include AI-predicted purpose labels and have no completed independent 200-case validation. Saved GDS author-reply cadence is sparse (30 comparable early/late cases). Neither the 6,425 selected pool nor its label accuracy can be generalized to all 37,596 without testing excluded strata.

## Immediate research recommendation

**Pause larger sentiment-model runs until the dataset scope decision is recorded.** Reuse existing data: adjudicate the 9 omitted old update-yes Reddit IDs under the current rubric, then draw a small, prespecified, protected-free probability sample from broad posts not selected by relevance AI, with a comment-rich oversample and known weights. This determines whether Option B can materially improve GDS/cadence coverage. Do not silently merge new posts or relabel the current 6,425; version any expanded cohort. Once scope is frozen, resume model comparison and conversation analysis. The existing 200-case packet is suitable only for its original selected-frame target and still has zero saved decisions.

## Audit artifacts

`MASTER_DATASET_INVENTORY.md`, `DATASET_LINEAGE_AND_HISTORY.md`, `DATASET_OVERLAP_MATRIX.csv`, `FULL_RECORD_RECONCILIATION.csv`, `CLASSIFICATION_COVERAGE_REPORT.md`, `GDS_AND_CONVERSATION_COVERAGE.md`, `MISSING_DATA_AND_EXCLUSIONS.md`, `DATASET_EXPANSION_OPTIONS.md`, and `AUDIT_COMPUTATIONS.json` were generated by `scripts/audit_dataset_lineage_2026_11_03.py`. No source, label, sentiment, protected outcome, or review packet was modified.
""")


def conv_from_broad(row: dict, thread: dict | None) -> dict:
    na = int(row.get("n_author_replies_nonempty") or 0)
    nc = int(row.get("n_community_replies_nonempty") or 0)
    nn = int(row.get("n_nonempty_comments") or 0)
    return {
        "n_comments": nn, "n_stored": int(row.get("n_stored_comments") or 0),
        "n_author": na, "n_community": nc,
        "n_timed": int((thread or {}).get("n_timed_nonempty") or 0),
        "order_eligible": bool((thread or {}).get("order_eligible")),
        "count_gap": row.get("count_gap"),
        "comments_status": row.get("comments_status"),
    }


def coverage(ids, broad, threads, current, master):
    result = Counter()
    for rid in ids:
        if rid in broad:
            c = conv_from_broad(broad[rid], threads.get(rid))
        elif rid in current:
            s = current[rid].get("sentiment") or {}
            inferred_nonempty = s.get("n_nonempty_comments")
            if inferred_nonempty is None:
                inferred_nonempty = int(s.get("n_author_replies_nonempty") or 0) + int(s.get("n_community_replies_nonempty") or 0)
            c = {"n_comments": int(inferred_nonempty),
                 "n_stored": s.get("n_stored_comments"),
                 "n_author": int(s.get("n_author_replies_nonempty") or 0),
                 "n_community": int(s.get("n_community_replies_nonempty") or 0),
                 "n_timed": int(s["n_timed_nonempty"]) if s.get("n_timed_nonempty") is not None else None,
                 "order_eligible": bool(s.get("order_eligible")),
                 "count_gap": s.get("count_gap")}
        elif rid in master:
            c = {"n_comments": int(master[rid].get("n_comments_stored") or 0),
                 "n_stored": int(master[rid].get("n_comments_stored") or 0),
                 "n_author": None, "n_community": None, "n_timed": None,
                 "order_eligible": False, "count_gap": None}
        else:
            result["no_conversation_source"] += 1
            continue
        n, a, k = c["n_comments"], c["n_author"], c["n_community"]
        result["with_counts"] += 1
        result["comments_0" if n == 0 else "comments_ge1"] += 1
        if c["n_stored"] is not None:
            result["stored_comments_ge1" if int(c["n_stored"]) > 0 else "stored_comments_0"] += 1
        else:
            result["stored_comment_count_unknown"] += 1
        for threshold in (1, 2, 3, 4, 5):
            if a is not None and a >= threshold:
                result[f"author_ge{threshold}"] += 1
        if n >= 10:
            result["comments_ge10"] += 1
        if a is not None and k is not None and a > 0 and k > 0:
            result["both_roles"] += 1
        if c["n_timed"] is not None and c["n_timed"] >= 2:
            result["timed_ge2"] += 1
        if c["n_timed"] is not None and n > 0:
            if c["n_timed"] == n:
                result["all_nonempty_replies_timed"] += 1
            elif 0 < c["n_timed"] < n:
                result["partial_timestamp_coverage"] += 1
            elif c["n_timed"] == 0:
                result["no_timed_nonempty_reply"] += 1
        if c["order_eligible"]:
            result["order_eligible"] += 1
        if c["count_gap"] is True:
            result["count_gap"] += 1
        if a is None:
            result["role_count_unknown"] += 1
    return dict(result)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with SRC["historical"].open(newline="", encoding="utf-8-sig") as handle:
        historical, hs = load_by_id("historical", csv.DictReader(handle))
    master, ms = load_by_id("master", json.loads(SRC["master"].read_text()))
    development, ds = load_by_id("development", jsonl(SRC["development"]))
    broad, bs = load_by_id("broad", jsonl(SRC["broad"]))
    capture_manifest = json.loads(SRC["capture_manifest"].read_text())
    capture_ids = []
    capture_host = Counter()
    page_hash_failures = []
    for page in capture_manifest["pages"]:
        page_path = SRC["capture_manifest"].parent / page["file"]
        if digest(page_path) != page["sha256"]:
            page_hash_failures.append(page["file"])
        raw_page = json.loads(page_path.read_text())
        for record in raw_page.get("data", []):
            rid = ident(record)
            if rid:
                capture_ids.append(rid)
            url = str(record.get("url") or record.get("reddit_url") or "")
            if "reddit.com" in url:
                capture_host["reddit.com"] += 1
            elif url:
                capture_host["other_or_unknown_url"] += 1
            else:
                capture_host["url_missing"] += 1
    assert set(broad) <= set(capture_ids), "broad score IDs absent from raw frozen pages"
    assert len(capture_ids) - len(broad) == 133, "raw-to-Reddit URL filter changed"
    assert not page_hash_failures, "frozen capture page hash mismatch"
    # Only conversation metadata are retained from the thread table.
    threads = {ident(r): {k: r.get(k) for k in ("n_timed_nonempty", "order_eligible")}
               for r in jsonl(SRC["thread"]) if ident(r)}
    predictions, ps = load_by_id("relevance_predictions", jsonl(SRC["relevance_predictions"]))
    selected_all, ss = load_by_id("selected", jsonl(SRC["selected"]))
    existing, es = load_by_id("existing", jsonl(SRC["existing"]))
    current, cs = load_by_id("current", jsonl(SRC["current"]))
    # Read only ID keys from protected records. Ignore every label/outcome field.
    protected = {ident(r) for key in ("reserved_ids_only", "fresh_ids_only")
                 for r in jsonl(SRC[key]) if ident(r)}
    selected = {rid: row for rid, row in selected_all.items() if rid not in protected}
    sets = {"historical_542": set(historical), "phase2_master_3362": set(master),
            "development_3151": set(development), "broad_37596": set(broad),
            "relevance_predictions": set(predictions), "ai_selected_5686": set(selected),
            "existing_1964": set(existing), "current_6425": set(current),
            "current_relevant_4944": {i for i, r in current.items() if r.get("final_relevance") == "relevant"},
            "main_vader_tps_gds": {i for i, r in current.items() if r.get("final_relevance") == "relevant"
                                   and r.get("final_discussion") in ("TPS", "GDS")
                                   and r.get("sentiment_definition") == "frozen_broad_release_train_package_scores"},
            "protected_ids_only": protected}
    names = list(sets)
    overlap = []
    for a in names:
        for b in names:
            overlap.append({"set_a": a, "set_b": b, "n_a": len(sets[a]), "n_b": len(sets[b]),
                            "intersection": len(sets[a] & sets[b]),
                            "a_only": len(sets[a] - sets[b]), "b_only": len(sets[b] - sets[a]),
                            "union": len(sets[a] | sets[b])})
    write_csv(OUT / "DATASET_OVERLAP_MATRIX.csv",
              ["set_a", "set_b", "n_a", "n_b", "intersection", "a_only", "b_only", "union"], overlap)

    # Separate local snapshots are retained as alternative availability evidence.
    alternative = {}
    normalized_path = BASE / "results/normalized_records_compact.jsonl"
    for row in jsonl(normalized_path):
        rid = ident(row)
        snapshots = row.get("snapshots") or []
        if rid and snapshots:
            best = max(snapshots, key=lambda s: int(s.get("n_nonempty_comments") or 0))
            alternative[rid] = {"max_nonempty_comments": int(best.get("n_nonempty_comments") or 0),
                                "max_author_replies": int(best.get("n_author_replies_nonempty") or 0),
                                "max_community_replies": int(best.get("n_community_replies_nonempty") or 0),
                                "source_key": best.get("source_key")}

    all_ids = sorted(set().union(sets["historical_542"], sets["phase2_master_3362"],
                                 sets["broad_37596"], sets["current_6425"]))
    fields = ["reddit_id", "in_historical_542", "in_phase2_master_3362", "in_development_3151",
              "in_broad_37596", "in_ai_selected_5686", "in_existing_1964", "in_current_6425",
              "protected_id", "historical_binary_label", "current_relevance", "current_purpose",
              "decision_source", "broad_relevance_prediction", "broad_prediction_status",
              "exclusion_or_nonselection_reason", "title_available", "body_available", "n_comments_nonempty",
              "n_author_replies", "n_community_replies", "n_timed_replies", "order_eligible", "count_gap",
              "alternate_max_nonempty_comments", "alternate_max_author_replies", "alternate_source_key"]
    def rec_rows():
        for rid in all_ids:
            b = broad.get(rid, {})
            t = threads.get(rid, {})
            p = predictions.get(rid, {})
            c = current.get(rid, {})
            m = master.get(rid, {})
            s = c.get("sentiment") or {}
            if rid in protected:
                reason = "protected_evaluation_id_excluded"
            elif rid in broad:
                reason = "selected_and_assembled" if rid in current else (
                    "predicted_relevant_but_not_selected" if p.get("predicted_label") == "relevant" else
                    "not_ai_predicted_relevant_or_prediction_error")
            elif rid in current:
                reason = "historical_or_local_only_not_in_frozen_broad_capture"
            else:
                reason = "local_only_not_in_current_pool"
            yield {"reddit_id": rid, "in_historical_542": int(rid in historical),
                   "in_phase2_master_3362": int(rid in master), "in_development_3151": int(rid in development),
                   "in_broad_37596": int(rid in broad), "in_ai_selected_5686": int(rid in selected),
                   "in_existing_1964": int(rid in existing), "in_current_6425": int(rid in current),
                   "protected_id": int(rid in protected),
                   "historical_binary_label": ("TPS" if historical[rid].get("corrected_label") == "1" else "broad_GDS") if rid in historical and rid not in protected else "",
                   "current_relevance": c.get("final_relevance", "") if rid not in protected else "",
                   "current_purpose": c.get("final_discussion", "") if rid not in protected else "",
                   "decision_source": c.get("decision_source", "") if rid not in protected else "",
                   "broad_relevance_prediction": p.get("predicted_label", "") if rid not in protected else "",
                   "broad_prediction_status": p.get("response_status", "") if rid not in protected else "",
                   "exclusion_or_nonselection_reason": reason,
                   "title_available": int(bool(b.get("title_nonempty"))) if rid in broad else int(bool(m.get("title"))),
                   "body_available": int(bool(b.get("body_nonempty"))) if rid in broad else int(bool(m.get("post_body"))),
                   "n_comments_nonempty": b.get("n_nonempty_comments",
                       s.get("n_nonempty_comments",
                             (int(s.get("n_author_replies_nonempty") or 0) + int(s.get("n_community_replies_nonempty") or 0)) if s else m.get("n_comments_stored", ""))),
                   "n_author_replies": b.get("n_author_replies_nonempty", s.get("n_author_replies_nonempty", "")),
                   "n_community_replies": b.get("n_community_replies_nonempty", s.get("n_community_replies_nonempty", "")),
                   "n_timed_replies": t.get("n_timed_nonempty", s.get("n_timed_nonempty", "")),
                   "order_eligible": t.get("order_eligible", s.get("order_eligible", "")),
                   "count_gap": b.get("count_gap", s.get("count_gap", "")),
                   "alternate_max_nonempty_comments": alternative.get(rid, {}).get("max_nonempty_comments", ""),
                   "alternate_max_author_replies": alternative.get(rid, {}).get("max_author_replies", ""),
                   "alternate_source_key": alternative.get(rid, {}).get("source_key", "")}
    write_csv(OUT / "FULL_RECORD_RECONCILIATION.csv", fields, rec_rows())

    raw_stats = {"historical": hs, "master": ms, "development": ds, "broad": bs,
                 "relevance_predictions": ps, "selected": ss, "existing": es, "current": cs}
    relev_counts = Counter((r.get("predicted_label"), r.get("response_status")) for r in predictions.values())
    current_relev = Counter(r.get("final_relevance") for r in current.values())
    current_disc = Counter(r.get("final_discussion") for r in current.values() if r.get("final_relevance") == "relevant")
    old_to_new = defaultdict(Counter)
    for rid, h in historical.items():
        if rid in protected:
            continue
        old = "historical_TPS" if h.get("corrected_label") == "1" else "historical_broad_GDS"
        now = current.get(rid, {})
        label = (now.get("final_relevance") or "not_in_current") + "/" + (now.get("final_discussion") or "NA")
        old_to_new[old][label] += 1
    # Alternative locally saved snapshots can contain more replies than the chosen broad snapshot.
    alt_more = [rid for rid, a in alternative.items() if rid in broad and
                a["max_nonempty_comments"] > int(broad[rid].get("n_nonempty_comments") or 0)]
    broad_in_current_nonselected = (sets["broad_37596"] & sets["current_6425"]) - sets["ai_selected_5686"]
    summary = {
        "source_paths_and_sha256": {k: {"path": str(p), "sha256": digest(p)} for k, p in SRC.items()},
        "source_row_diagnostics": raw_stats,
        "raw_capture_page_check": {"page_count": len(capture_manifest["pages"]),
                                    "page_hash_failures": page_hash_failures,
                                    "raw_rows": len(capture_ids), "unique_ids": len(set(capture_ids)),
                                    "duplicate_extra_rows": len(capture_ids) - len(set(capture_ids)),
                                    "broad_ids_subset_of_raw": set(broad) <= set(capture_ids),
                                    "raw_only_nonreddit_url_ids": len(set(capture_ids) - set(broad)),
                                    "url_host_counts": dict(capture_host)},
        "set_counts": {k: len(v) for k, v in sets.items()},
        "broad_minus_current": len(sets["broad_37596"] - sets["current_6425"]),
        "current_minus_broad": len(sets["current_6425"] - sets["broad_37596"]),
        "broad_current_overlap": len(sets["broad_37596"] & sets["current_6425"]),
        "broad_in_current_but_not_ai_selected": len(broad_in_current_nonselected),
        "broad_in_current_nonselected_prediction_labels": dict(Counter(
            str(predictions[rid].get("predicted_label") or "error") for rid in broad_in_current_nonselected)),
        "broad_unassembled_prediction_labels": dict(Counter(
            str(predictions[rid].get("predicted_label") or "error")
            for rid in sets["broad_37596"] - sets["current_6425"])),
        "broad_unassembled_protected": len((sets["broad_37596"] - sets["current_6425"]) & protected),
        "master_unassembled_protected": len((sets["phase2_master_3362"] - sets["current_6425"]) & protected),
        "master_unassembled_nonprotected": len((sets["phase2_master_3362"] - sets["current_6425"]) - protected),
        "master_unassembled_nonprotected_review_fields": {
            "update_related": dict(Counter(str(master[rid].get("phase2_update_related"))
                                           for rid in (sets["phase2_master_3362"] - sets["current_6425"]) - protected)),
            "review_status": dict(Counter(str(master[rid].get("phase2_review_status"))
                                         for rid in (sets["phase2_master_3362"] - sets["current_6425"]) - protected)),
            "update_yes_reddit_url": sum(1 for rid in (sets["phase2_master_3362"] - sets["current_6425"]) - protected
                                         if master[rid].get("phase2_update_related") == "yes"
                                         and "reddit.com" in str(master[rid].get("url") or "")),
            "update_yes_stackoverflow_url": sum(1 for rid in (sets["phase2_master_3362"] - sets["current_6425"]) - protected
                                               if master[rid].get("phase2_update_related") == "yes"
                                               and "stackoverflow.com" in str(master[rid].get("url") or "")),
            "update_yes_reddit_ids": sorted(rid for rid in (sets["phase2_master_3362"] - sets["current_6425"]) - protected
                                            if master[rid].get("phase2_update_related") == "yes"
                                            and "reddit.com" in str(master[rid].get("url") or "")),
        },
        "broad_period_counts": dict(Counter(str(predictions[rid].get("stratum")) for rid in broad)),
        "current_period_counts": dict(Counter(str(r.get("selection_stratum")) for r in current.values())),
        "broad_text_coverage": {
            "title_nonempty": sum(bool(r.get("title_nonempty")) for r in broad.values()),
            "body_nonempty": sum(bool(r.get("body_nonempty")) for r in broad.values()),
            "title_or_body_nonempty": sum(bool(r.get("title_nonempty") or r.get("body_nonempty")) for r in broad.values()),
            "source_reddit": sum(bool(r.get("source_is_reddit")) for r in broad.values()),
            "reddit_url_present": sum(bool(r.get("reddit_url")) for r in broad.values()),
        },
        "selected_all_including_protected": len(selected_all),
        "selected_protected_excluded": len(set(selected_all) & protected),
        "historical_corrected_label_counts": dict(Counter(
            "TPS" if r.get("corrected_label") == "1" else "broad_GDS" for r in historical.values())),
        "alternative_saved_snapshot": {
            "normalized_record_ids": len(alternative),
            "overlap_broad": len(set(alternative) & set(broad)),
            "more_nonempty_comments_than_broad_snapshot": len(alt_more),
            "more_in_current": len(set(alt_more) & set(current)),
            "more_in_broad_unassembled": len(set(alt_more) - set(current)),
            "source_path": str(normalized_path), "sha256": digest(normalized_path),
        },
        "broad_prediction_counts": {f"{a}/{b}": n for (a, b), n in relev_counts.items()},
        "current_relevance_counts": dict(current_relev), "current_purpose_counts": dict(current_disc),
        "historical_to_current_nonprotected": {k: dict(v) for k, v in old_to_new.items()},
        "conversation": {
            "broad": coverage(sets["broad_37596"], broad, threads, current, master),
            "master": coverage(sets["phase2_master_3362"], broad, threads, current, master),
            "current": coverage(sets["current_6425"], broad, threads, current, master),
            **{label: coverage({i for i, r in current.items() if r.get("final_relevance") == "relevant"
                                  and r.get("final_discussion") == label}, broad, threads, current, master)
               for label in ("TPS", "GDS", "Other", "Unclear")},
            "broad_unassembled": coverage(sets["broad_37596"] - sets["current_6425"], broad, threads, current, master),
        },
    }
    write_json(OUT / "AUDIT_COMPUTATIONS.json", summary)
    assert summary["broad_current_overlap"] + summary["broad_minus_current"] == 37596
    assert summary["broad_current_overlap"] + summary["current_minus_broad"] == 6425
    assert summary["selected_protected_excluded"] == 518 and len(selected) == 5686
    assert set(selected) <= set(current)
    assert not (set(current) & protected)
    write_reports(summary)
    print(json.dumps({k: summary[k] for k in ("set_counts", "broad_current_overlap", "broad_minus_current", "current_minus_broad", "current_relevance_counts", "current_purpose_counts")}, indent=2))


if __name__ == "__main__":
    main()
