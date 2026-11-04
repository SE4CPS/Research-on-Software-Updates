"""Stage 1: removal correction, duplication audit, frozen v2 candidate dataset."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any

from full_resource_sentiment_v1.correction_rq_cardiff_2026_11_03 import DATASET_VERSION
from full_resource_sentiment_v1.correction_rq_cardiff_2026_11_03.common import (
    CAPTURE_DIR,
    CONTINUATION_OUT,
    DATASET_JSONL,
    OUT_ROOT,
    POST_SCORES,
    PRIOR_SCREEN,
    RESTRICTED_ROOT,
    TEXT_MANIFEST,
    load_jsonl,
    protected_ids_only,
    sha256_file,
    sha256_text,
    utcnow,
    write_json,
    write_jsonl,
)
from full_resource_sentiment_v1.correction_rq_cardiff_2026_11_03.removal_rule import (
    deterministic_unassessable,
)
from full_resource_sentiment_v1.master_research_corrections_2026_11_03.near_dup import (
    build_components,
    normalize_ws,
)


def _load_texts(ids: set[str]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    remaining = set(ids)
    for path in sorted(CAPTURE_DIR.glob("page_*.json")):
        if not remaining:
            break
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("data") or []:
            rid = str(row.get("redditId") or "")
            if rid not in remaining:
                continue
            title = "" if row.get("title") is None else str(row.get("title"))
            body = "" if row.get("author_description") is None else str(row.get("author_description"))
            out[rid] = {"title": title, "body": body}
            remaining.discard(rid)
    return out


def run_stage1() -> dict[str, Any]:
    dest = OUT_ROOT / "stage1_dataset_correction"
    dest.mkdir(parents=True, exist_ok=True)
    prot = protected_ids_only()
    prior = load_jsonl(PRIOR_SCREEN)
    assert len(prior) == 30998
    ids = {str(r["reddit_id"]) for r in prior}
    assert not (ids & prot)

    print("  loading capture texts...", flush=True)
    texts = _load_texts(ids)
    print(f"  loaded {len(texts)}", flush=True)

    corrected = []
    changed_audit = []
    override_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    purpose_counts: Counter[str] = Counter()

    for row in prior:
        rid = str(row["reddit_id"])
        doc = texts.get(rid) or {"title": "", "body": ""}
        title, body = doc.get("title") or "", doc.get("body") or ""
        basis = deterministic_unassessable(title, body)
        out = dict(row)
        out["title_hash"] = sha256_text(title)
        out["body_hash"] = sha256_text(body)
        out["norm_title_body_hash"] = sha256_text(normalize_ws(f"{title}\n{body}"))
        out["original_model_recovery_status"] = row.get("recovery_status")
        out["original_model_stageA"] = row.get("stageA_relevance_pred")
        out["original_model_stageB"] = row.get("stageB_purpose_pred")
        out["original_model_stageA_prob"] = row.get("stageA_prob_relevant")
        out["original_model_stageB_prob"] = row.get("stageB_purpose_prob")

        if basis:
            out["effective_status"] = "unassessable"
            out["effective_relevance"] = "unassessable"
            out["effective_purpose"] = None
            out["override_reason"] = basis
            out["recovery_status_effective"] = "unassessable_removed_or_empty"
            if row.get("recovery_status") != "unassessable_empty_text" and row.get("recovery_status") != "unassessable":
                changed_audit.append(
                    {
                        "reddit_id": rid,
                        "prior_recovery_status": row.get("recovery_status"),
                        "prior_purpose": row.get("stageB_purpose_pred"),
                        "prior_stageA_prob": row.get("stageA_prob_relevant"),
                        "override_reason": basis,
                        "title_hash": out["title_hash"],
                        "body_hash": out["body_hash"],
                        "in_prior_9882_screen": row.get("in_prior_9882_screen"),
                        "in_fixed_pilot_240": row.get("in_fixed_pilot_240"),
                        "text_excerpt_redacted": "[REMOVAL_OR_EMPTY_NOTICE]",
                    }
                )
                override_counts[basis] += 1
        else:
            out["effective_status"] = row.get("recovery_status")
            out["effective_relevance"] = row.get("stageA_relevance_pred")
            out["effective_purpose"] = row.get("stageB_purpose_pred")
            out["override_reason"] = None
            out["recovery_status_effective"] = row.get("recovery_status")

        # Effective candidate flag
        is_cand = (
            out["recovery_status_effective"] == "candidate_ai_only"
            and out.get("effective_purpose") in {"TPS", "GDS"}
        )
        out["effective_candidate_tps_gds"] = bool(is_cand)
        status_counts[out["recovery_status_effective"]] += 1
        if is_cand:
            purpose_counts[out["effective_purpose"]] += 1
        corrected.append(out)

    write_jsonl(RESTRICTED_ROOT / "full_frame_screen_30998_corrected.jsonl", corrected)
    write_jsonl(RESTRICTED_ROOT / "removal_override_audit.jsonl", changed_audit)

    n_cand = sum(1 for r in corrected if r["effective_candidate_tps_gds"])
    n_tps = purpose_counts.get("TPS", 0)
    n_gds = purpose_counts.get("GDS", 0)
    n_unassess = sum(1 for r in corrected if r["effective_status"] == "unassessable")
    n_other = sum(1 for r in corrected if r["recovery_status_effective"] == "candidate_relevant_other_only")
    n_not = sum(1 for r in corrected if r["recovery_status_effective"] == "not_candidate")

    # --- Training ↔ screen duplication ---
    frame_text = {str(r["reddit_id"]): r for r in load_jsonl(TEXT_MANIFEST)}
    train_rows = []
    for r in load_jsonl(DATASET_JSONL):
        rid = str(r["reddit_id"])
        if rid in prot:
            continue
        doc = frame_text.get(rid) or {"title": "", "body": ""}
        title, body = doc.get("title") or "", doc.get("body") or ""
        text = f"{title}\n{body}".strip()
        if not text:
            continue
        train_rows.append(
            {
                "reddit_id": rid,
                "title": title,
                "body": body,
                "norm_hash": sha256_text(normalize_ws(text)),
                "label": r.get("final_discussion"),
                "relevance": r.get("final_relevance"),
            }
        )
    train_hash_to_ids: dict[str, list[str]] = defaultdict(list)
    for t in train_rows:
        train_hash_to_ids[t["norm_hash"]].append(t["reddit_id"])

    exact_dup_candidates = []
    for r in corrected:
        if not r["effective_candidate_tps_gds"]:
            continue
        h = r["norm_title_body_hash"]
        if h in train_hash_to_ids:
            exact_dup_candidates.append(
                {
                    "screen_reddit_id": r["reddit_id"],
                    "train_reddit_ids": train_hash_to_ids[h],
                    "purpose": r["effective_purpose"],
                    "norm_hash": h,
                }
            )

    # Near-dup via connected components on combined train+candidate texts (predeclared Jaccard)
    cand_for_nd = [r for r in corrected if r["effective_candidate_tps_gds"]]
    combo_ids = [t["reddit_id"] for t in train_rows] + [r["reddit_id"] for r in cand_for_nd]
    combo_titles = [t["title"] for t in train_rows] + [
        (texts.get(r["reddit_id"]) or {}).get("title") or "" for r in cand_for_nd
    ]
    combo_bodies = [t["body"] for t in train_rows] + [
        (texts.get(r["reddit_id"]) or {}).get("body") or "" for r in cand_for_nd
    ]
    train_id_set = {t["reddit_id"] for t in train_rows}
    cand_id_set = {r["reddit_id"] for r in cand_for_nd}
    cand_purpose = {r["reddit_id"]: r["effective_purpose"] for r in cand_for_nd}
    comp = build_components(combo_ids, combo_titles, combo_bodies)
    # Map component -> members
    from collections import defaultdict as _dd

    members: dict[Any, list[str]] = _dd(list)
    for rid, cid in zip(combo_ids, comp["component_ids"]):
        members[cid].append(rid)
    near_dup = []
    near_dup_ids: set[str] = set()
    for cid, group in members.items():
        train_hits = [x for x in group if x in train_id_set]
        cand_hits = [x for x in group if x in cand_id_set]
        if train_hits and cand_hits:
            for sid in cand_hits:
                near_dup.append(
                    {
                        "screen_reddit_id": sid,
                        "train_reddit_ids_in_component": train_hits[:20],
                        "component_size": len(group),
                        "purpose": cand_purpose.get(sid),
                    }
                )
                near_dup_ids.add(sid)

    write_jsonl(RESTRICTED_ROOT / "train_screen_exact_dup_candidates.jsonl", exact_dup_candidates)
    write_jsonl(RESTRICTED_ROOT / "train_screen_near_dup_candidates.jsonl", near_dup)

    dup_report = {
        "utc": utcnow(),
        "exact_normalized_title_body_hash_dups": {
            "n_candidate_rows": len(exact_dup_candidates),
            "n_tps": sum(1 for x in exact_dup_candidates if x["purpose"] == "TPS"),
            "n_gds": sum(1 for x in exact_dup_candidates if x["purpose"] == "GDS"),
        },
        "near_dup_method": "build_components exact_full_plus_title_body_jaccard (same as RQ4)",
        "near_dup_candidates": {
            "n_candidate_rows_flagged": len(near_dup_ids),
            "n_tps": sum(1 for x in near_dup if x["purpose"] == "TPS"),
            "n_gds": sum(1 for x in near_dup if x["purpose"] == "GDS"),
        },
        "sensitivity_note": (
            "If material, exclude these IDs in RQ sensitivity; do not retrain on full-frame outputs"
        ),
        "model_not_tuned_on_full_frame_outputs": True,
    }
    write_json(dest / "TRAIN_SCREEN_DUPLICATION_AUDIT.json", dup_report)

    # --- New versioned union ---
    original = load_jsonl(DATASET_JSONL)
    assert len(original) == 6425
    orig_ids = {str(r["reddit_id"]) for r in original}
    post_meta = {str(r["reddit_id"]): r for r in load_jsonl(POST_SCORES)}

    union = []
    for r in original:
        rid = str(r["reddit_id"])
        union.append(
            {
                "dataset_version": DATASET_VERSION,
                "record_origin": "original_6425_preserved",
                "reddit_id": rid,
                "final_relevance": r.get("final_relevance"),
                "final_discussion": r.get("final_discussion"),
                "decision_source": r.get("decision_source"),
                "decision_confidence": r.get("decision_confidence"),
                "selection_stratum": r.get("selection_stratum"),
                "sentiment_definition": r.get("sentiment_definition"),
                "ai_assisted_candidate": False,
                "human_validated": None,
                "confirmed": False,
                "label_overwritten": False,
                "n_author": (r.get("sentiment") or {}).get("n_author_replies_nonempty"),
                "n_community": (r.get("sentiment") or {}).get("n_community_replies_nonempty"),
                "capture_source": (post_meta.get(rid) or {}).get("source"),
                "created_month": (post_meta.get(rid) or {}).get("created_month"),
            }
        )

    exact_dup_ids = {x["screen_reddit_id"] for x in exact_dup_candidates}
    near_dup_ids = {x["screen_reddit_id"] for x in near_dup}
    selected = [r for r in corrected if r["effective_candidate_tps_gds"]]
    for r in selected:
        rid = r["reddit_id"]
        assert rid not in orig_ids
        assert rid not in prot
        union.append(
            {
                "dataset_version": DATASET_VERSION,
                "record_origin": "ai_assisted_screen_candidate_v2",
                "reddit_id": rid,
                "final_relevance": "relevant",
                "final_discussion": r["effective_purpose"],
                "decision_source": "tfidf_two_stage_full_frame_v1_seed_20261103",
                "decision_confidence": r.get("original_model_stageB_prob"),
                "decision_status": "ai_candidate_unconfirmed",
                "selection_stratum": f"excluded_screen_{r.get('author_depth_bucket')}",
                "ai_assisted_candidate": True,
                "human_validated": False,
                "confirmed": False,
                "label_overwritten": False,
                "override_reason": r.get("override_reason"),
                "stageA_prob_relevant": r.get("original_model_stageA_prob"),
                "n_author": r.get("n_author"),
                "n_community": r.get("n_community"),
                "capture_source": r.get("source"),
                "created_month": r.get("created_month"),
                "exact_dup_with_train": rid in exact_dup_ids,
                "near_dup_with_train": rid in near_dup_ids,
                "note": "AI candidate only — NOT human-confirmed",
            }
        )

    write_jsonl(RESTRICTED_ROOT / f"{DATASET_VERSION}.jsonl", union)

    summary = {
        "utc": utcnow(),
        "preserves_continuation_tree": str(CONTINUATION_OUT),
        "n_screened": len(corrected),
        "n_override_to_unassessable": len(changed_audit),
        "override_reason_counts": dict(override_counts),
        "by_recovery_status_effective": dict(status_counts),
        "n_unassessable": n_unassess,
        "n_candidate_tps_gds_ai_only_corrected": n_cand,
        "n_candidate_tps": n_tps,
        "n_candidate_gds": n_gds,
        "n_candidate_relevant_other": n_other,
        "n_not_candidate": n_not,
        "n_confirmed": 0,
        "expected_min_overrides_codex": 65,
        "overrides_ge_65": len(changed_audit) >= 65,
        "dataset_version": DATASET_VERSION,
        "n_union_rows": len(union),
        "n_original_preserved": 6425,
        "n_ai_assisted_added": len(selected),
        "overlap_original_ai": 0,
        "protected_in_union": sum(1 for r in union if r["reddit_id"] in prot),
        "unique_ids": len({r["reddit_id"] for r in union}),
        "duplication": dup_report,
        "prior_screen_sha256": sha256_file(PRIOR_SCREEN),
        "scientific_status": "AI candidates corrected for unassessable removals; not human-confirmed",
    }
    write_json(dest / "DATASET_CORRECTION_SUMMARY.json", summary)

    # Correction checkpoint packet for Codex
    (dest / "CORRECTION_CHECKPOINT_FOR_CODEX.md").write_text(
        f"""# Correction checkpoint — dataset v2 (removals)

**UTC:** {utcnow()}  
**Sealed prior tree preserved:** `{CONTINUATION_OUT}`  
**Original 6425 untouched.**

## Corrected counts (AI-only; not confirmed)
- Overrides to unassessable: **{len(changed_audit)}** (Codex known ≥65; ge_65={len(changed_audit) >= 65})
- Effective AI TPS/GDS candidates: **{n_cand}** (TPS {n_tps}, GDS {n_gds})
- Unassessable: **{n_unassess}**
- Relevant/Other: **{n_other}**; not candidate: **{n_not}**
- Union v2 rows: **{len(union)}** (6425 + {len(selected)})
- Exact train↔screen norm-hash dups among candidates: **{len(exact_dup_candidates)}**
- Near-dup (RQ4 component grouping) candidates flagged: **{len(near_dup_ids)}**

## Files
- `DATASET_CORRECTION_SUMMARY.json`
- restricted: `full_frame_screen_30998_corrected.jsonl`, `removal_override_audit.jsonl`, `{DATASET_VERSION}.jsonl`

Main RQ/Cardiff analyses proceed on this frozen v2 after this checkpoint is written.
""",
        encoding="utf-8",
    )
    return summary
