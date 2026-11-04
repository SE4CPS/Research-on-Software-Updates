"""Stage 1: reply-depth inventory, TF-IDF audit, two-stage candidate recovery (no merge)."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

from full_resource_sentiment_v1.strict_continuation_corrections_2026_11_03 import SEED
from full_resource_sentiment_v1.strict_continuation_corrections_2026_11_03.common import (
    CAPTURE_DIR,
    DATASET_JSONL,
    LUNA_PREDS,
    ORIG_STRICT,
    ORIG_STRICT_RESTRICTED,
    OUT_ROOT,
    PILOT_IDS,
    POST_SCORES,
    RESTRICTED_ROOT,
    load_jsonl,
    protected_ids_only,
    sha256_file,
    sha256_text,
    utcnow,
    write_json,
    write_jsonl,
)

UPDATE_CUES = re.compile(
    r"\b(update|upgrade|patch|rollback|firmware|version|changelog|release notes?|kernel)\b", re.I
)
TROUBLE_CUES = re.compile(
    r"\b(error|crash|bug|fail(?:ed|ure)?|broken|fix|workaround|stacktrace|regression|not working)\b",
    re.I,
)
DISCONTENT_CUES = re.compile(
    r"\b(hate|ruined|awful|terrible|garbage|worst|disappointed|frustrated|angry|sucks)\b", re.I
)


def reply_depth_inventory() -> dict[str, Any]:
    dest = OUT_ROOT / "stage1_reply_depth_inventory"
    dest.mkdir(parents=True, exist_ok=True)
    prot = protected_ids_only()
    frame = {str(r["reddit_id"]) for r in load_jsonl(DATASET_JSONL)}
    counts = Counter()
    # Overlap matrix for reporting without double-counting in headlines
    for row in load_jsonl(POST_SCORES):
        rid = str(row["reddit_id"])
        if rid in frame or rid in prot:
            continue
        counts["nonprotected_excluded"] += 1
        na = int(row.get("n_author_replies_nonempty") or 0)
        nc = int(row.get("n_community_replies_nonempty") or 0)
        nt = int(row.get("n_nonempty_comments") or 0)
        if na >= 1:
            counts["author_ge1"] += 1
        if na >= 2:
            counts["author_ge2"] += 1
        if na >= 4:
            counts["author_ge4"] += 1
        if nc >= 1:
            counts["community_ge1"] += 1
        if nt >= 2:
            counts["total_replies_ge2"] += 1
        # exclusive-ish slices for clarity
        if na == 1:
            counts["author_exact_1"] += 1
        if na >= 2 and na < 4:
            counts["author_exact_2_or_3"] += 1

    report = {
        "utc": utcnow(),
        "counts_separate_overlapping_groups": dict(counts),
        "priority_note": (
            "Prioritize author>=2 for conversation-rich TPS/GDS recovery candidates; "
            "retain author==1 for first-author-reply sentiment; author>=4 for long trajectories only; "
            "retain fixed 240 probability sample for selection-bias / missed-relevance estimation."
        ),
        "expected_match": {
            "nonprotected_excluded": 30998,
            "author_ge1": 14918,
            "author_ge2": 8147,
            "author_ge4": 2847,
            "community_ge1": 24268,
        },
        "match": counts["nonprotected_excluded"] == 30998
        and counts["author_ge1"] == 14918
        and counts["author_ge2"] == 8147
        and counts["author_ge4"] == 2847,
        "protected_n": len(prot),
        "pilot_sha256": sha256_file(PILOT_IDS),
    }
    write_json(dest / "REPLY_DEPTH_COUNTS.json", report)
    return report


def audit_tfidf_model() -> dict[str, Any]:
    dest = OUT_ROOT / "stage1_tfidf_audit"
    dest.mkdir(parents=True, exist_ok=True)
    card_path = ORIG_STRICT / "phase2_excluded_screening/TFIDF_RELEVANCE_MODEL_CARD.json"
    card = json.loads(card_path.read_text(encoding="utf-8")) if card_path.exists() else {}
    preds = load_jsonl(ORIG_STRICT_RESTRICTED / "tfidf_predictions_pilot240.jsonl")
    n_rel = sum(1 for p in preds if p.get("tfidf_relevance") == "relevant")
    # FP-prone patterns: short title with update cue only
    patterns = Counter()
    for p in preds:
        if p.get("tfidf_relevance") != "relevant":
            continue
        ex = (p.get("evidence_quote") or p.get("text_excerpt") or "").lower()
        if "update" in ex and len(ex) < 80:
            patterns["short_title_with_update_token"] += 1
        if "version" in ex and "error" not in ex and "bug" not in ex:
            patterns["version_without_trouble_cues_in_excerpt"] += 1
    report = {
        "utc": utcnow(),
        "model_card": card,
        "train_positive_prevalence": {
            "reported_train_positive_relevant": card.get("training_positive_relevant"),
            "reported_training_n": card.get("training_n"),
            "approx_rate": (
                (card.get("training_positive_relevant") or 0) / (card.get("training_n") or 1)
            ),
            "note": "High positive prevalence + threshold 0.35 → many pilot relevant flags; not verified recoveries",
        },
        "pilot_tfidf_relevant_rate": {"n_relevant": n_rel, "n_pilot": len(preds), "rate": n_rel / max(len(preds), 1)},
        "threshold": card.get("threshold_selected_on_val"),
        "false_positive_prone_patterns_heuristic": dict(patterns),
        "rename": "evidence_quote → treat as text_excerpt (not verified update-evidence span)",
        "scientific_status": "TF-IDF relevant ≠ verified software-update relevance or TPS/GDS",
    }
    write_json(dest / "TFIDF_AUDIT.json", report)
    return report


def _load_capture_texts(ids: set[str]) -> dict[str, dict[str, str]]:
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


def _purpose_heuristic(title: str, body: str) -> dict[str, Any]:
    text = f"{title}\n{body}"
    if not text.strip():
        return {"purpose_candidate": "unassessable", "confidence": "high", "stage": "heuristic_v1"}
    if not UPDATE_CUES.search(text):
        return {
            "purpose_candidate": None,
            "relevance_gate": "not_update_related_by_cues",
            "confidence": "medium",
            "stage": "heuristic_v1",
        }
    trouble, discontent = bool(TROUBLE_CUES.search(text)), bool(DISCONTENT_CUES.search(text))
    if trouble and not discontent:
        purpose = "TPS"
    elif discontent and not trouble:
        purpose = "GDS"
    elif trouble and discontent:
        purpose = "TPS"
    else:
        purpose = "Other"
    return {
        "purpose_candidate": purpose,
        "relevance_gate": "update_cue_present",
        "confidence": "low",
        "stage": "heuristic_v1",
        "cues": {"trouble": trouble, "discontent": discontent},
    }


def two_stage_candidate_recovery() -> dict[str, Any]:
    """Bounded diagnostic: Luna vs TF-IDF + purpose heuristic/candidates. No merge. No full-frame run."""
    dest = OUT_ROOT / "stage1_candidate_recovery"
    dest.mkdir(parents=True, exist_ok=True)
    prot = protected_ids_only()

    # Spec for full-frame two-stage pipeline (code+checkpoints, not executed)
    pipeline_spec = {
        "stages": [
            {
                "name": "A_software_update_relevance",
                "outputs": ["relevant", "not_relevant", "unclear", "unassessable"],
                "methods_allowed_after_approval": ["tfidf_v1", "embedding_minilm", "paid_llm"],
            },
            {
                "name": "B_purpose_conditional_on_relevance",
                "outputs": ["TPS", "GDS", "Other", "Unclear"],
                "rule": "Never assign purpose if stage A ≠ relevant",
            },
        ],
        "checkpointing": "per-1000 IDs jsonl shards under restricted/",
        "idempotent": True,
        "full_frame_run": "NOT_EXECUTED — awaiting approval",
        "n_ids_if_approved": 30998,
    }
    write_json(dest / "TWO_STAGE_PIPELINE_SPEC.json", pipeline_spec)

    # Load existing TF-IDF preds for pilot+targeted
    pilot_pred = {
        str(r["reddit_id"]): r for r in load_jsonl(ORIG_STRICT_RESTRICTED / "tfidf_predictions_pilot240.jsonl")
    }
    targ_pred = {
        str(r["reddit_id"]): r for r in load_jsonl(ORIG_STRICT_RESTRICTED / "tfidf_predictions_targeted200.jsonl")
    }
    with PILOT_IDS.open(encoding="utf-8", newline="") as handle:
        pilot_rows = list(csv.DictReader(handle))
    pilot_ids = {str(r["reddit_id"]) for r in pilot_rows}
    targeted_path = ORIG_STRICT / "phase2_excluded_screening/targeted_200_public.jsonl"
    targeted = load_jsonl(targeted_path) if targeted_path.exists() else []

    # Author-rich diagnostic: sample from author>=2 Luna not_relevant vs TF-IDF would need scoring —
    # Here we use targeted (already author-enriched) + pilot disagreements only (bounded).
    need_ids = set(pilot_pred) | set(targ_pred)
    texts = _load_capture_texts(need_ids)

    # Train purpose TF-IDF on frame relevant TPS/GDS/Other only (blocked: prot/pilot/targeted)
    blocked = set(prot) | pilot_ids | {str(r["reddit_id"]) for r in targeted}
    purpose_rows = []
    for r in load_jsonl(DATASET_JSONL):
        rid = str(r["reddit_id"])
        if rid in blocked:
            continue
        if r.get("final_relevance") != "relevant":
            continue
        if r.get("final_discussion") not in {"TPS", "GDS", "Other"}:
            continue
        purpose_rows.append(r)
    # texts for frame from restricted manifest if available
    man = Path(
        "/Users/manumathewjiss/Documents/Research-on-Software-Updates/research-open-text-sentimental/"
        "phase6/full_resource_sentiment_v1/restricted/master_research_2026-11-03/text_thread_manifest_restricted.jsonl"
    )
    frame_text = {str(r["reddit_id"]): r for r in load_jsonl(man)} if man.exists() else {}

    X, y, groups = [], [], []
    label_to_i = {"TPS": 0, "GDS": 1, "Other": 2}
    for r in purpose_rows:
        rid = str(r["reddit_id"])
        doc = frame_text.get(rid) or {"title": "", "body": ""}
        text = f"{doc.get('title') or ''}\n{doc.get('body') or ''}".strip()
        if not text:
            continue
        X.append(text)
        y.append(label_to_i[r["final_discussion"]])
        groups.append(sha256_text((doc.get("title") or "")[:60])[:12])
    y_arr = np.asarray(y, dtype=int)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    tr, va = next(gss.split(X, y_arr, groups))
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=30000, sublinear_tf=True)
    Xtr = vec.fit_transform([X[i] for i in tr])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=SEED)
    clf.fit(Xtr, y_arr[tr])
    inv = {v: k for k, v in label_to_i.items()}

    post_scores = {str(r["reddit_id"]): r for r in load_jsonl(POST_SCORES)}
    luna = {}
    for row in load_jsonl(LUNA_PREDS):
        rid = str(row["reddit_id"])
        if rid in need_ids:
            luna[rid] = row.get("predicted_label")

    candidates = []
    for sample_name, pred_map in (("fixed_pilot_240", pilot_pred), ("targeted_200", targ_pred)):
        for rid, pref in pred_map.items():
            doc = texts.get(rid) or {"title": "", "body": ""}
            title, body = doc.get("title") or "", doc.get("body") or ""
            text = f"{title}\n{body}".strip()
            na = int((post_scores.get(rid) or {}).get("n_author_replies_nonempty") or 0)
            tfidf_rel = pref.get("tfidf_relevance")
            luna_lab = luna.get(rid)
            heur = _purpose_heuristic(title, body)
            purpose_model = None
            purpose_prob = None
            if tfidf_rel == "relevant" and text:
                proba = clf.predict_proba(vec.transform([text]))[0]
                idx = int(np.argmax(proba))
                purpose_model = inv[idx]
                purpose_prob = float(proba[idx])
            # Candidate recovery only if stage-A relevant (TF-IDF) AND purpose in TPS/GDS
            is_candidate = tfidf_rel == "relevant" and purpose_model in {"TPS", "GDS"}
            status = "candidate_ai_only" if is_candidate else "not_candidate"
            if tfidf_rel == "relevant" and purpose_model == "Other":
                status = "candidate_relevant_other_only"
            candidates.append(
                {
                    "reddit_id": rid,
                    "sample": sample_name,
                    "n_author": na,
                    "author_depth_bucket": (
                        "ge4" if na >= 4 else ("ge2" if na >= 2 else ("ge1" if na >= 1 else "ge0"))
                    ),
                    "luna_relevance": luna_lab,
                    "tfidf_relevance": tfidf_rel,
                    "tfidf_prob_relevant": pref.get("tfidf_prob_relevant"),
                    "purpose_tfidf_candidate": purpose_model,
                    "purpose_tfidf_prob": purpose_prob,
                    "purpose_heuristic": heur.get("purpose_candidate"),
                    "text_excerpt": (title or text)[:160],
                    "recovery_status": status,
                    "confirmed": False,
                    "human_validated": False,
                    "note": "AI agreement ≠ ground truth; not merged into 6425",
                }
            )

    write_jsonl(RESTRICTED_ROOT / "candidate_recoveries_pilot_targeted.jsonl", candidates)

    by_depth = defaultdict(Counter)
    for c in candidates:
        if c["recovery_status"] == "candidate_ai_only":
            by_depth[c["author_depth_bucket"]][c["purpose_tfidf_candidate"]] += 1
    summary = {
        "utc": utcnow(),
        "n_scored": len(candidates),
        "n_candidate_tps_gds_ai_only": sum(1 for c in candidates if c["recovery_status"] == "candidate_ai_only"),
        "n_candidate_tps": sum(
            1 for c in candidates if c["recovery_status"] == "candidate_ai_only" and c["purpose_tfidf_candidate"] == "TPS"
        ),
        "n_candidate_gds": sum(
            1 for c in candidates if c["recovery_status"] == "candidate_ai_only" and c["purpose_tfidf_candidate"] == "GDS"
        ),
        "n_confirmed": 0,
        "confirmed_definition": "Requires independent human reference — none in this run",
        "by_author_depth_candidates": {k: dict(v) for k, v in by_depth.items()},
        "disagreement_luna_not_relevant_tfidf_relevant": sum(
            1
            for c in candidates
            if c.get("luna_relevance") == "not_relevant" and c.get("tfidf_relevance") == "relevant"
        ),
        "scientific_status": "CANDIDATES only; not confirmed recoveries; not auto-merged",
        "purpose_model": {
            "id": "tfidf_purpose_ovr_v1",
            "train_n": int(len(tr)),
            "labels": ["TPS", "GDS", "Other"],
            "blocked_from_training": ["protected_693", "pilot_240", "targeted_200"],
        },
    }
    write_json(dest / "CANDIDATE_RECOVERY_SUMMARY.json", summary)

    # Resource proposal for full 30998
    proposal = {
        "action": "Run two-stage local TF-IDF relevance+purpose (or approved embedding/LLM) on 30,998 nonprotected excluded IDs",
        "n_ids": 30998,
        "priority_order": ["author_ge2", "author_ge1", "author_ge4_trajectory", "probability_240_stratum"],
        "estimated_time_cpu": "15–45 min TF-IDF both stages; hours if transformer without GPU",
        "estimated_disk": "<1 GiB outputs",
        "estimated_money": 0,
        "privacy": "restricted predictions; no protected IDs",
        "checkpoints": "shards of 1000",
        "expected_files": ["relevance_30998.jsonl", "purpose_among_relevant.jsonl", "disagreement_queues.json"],
        "rollback": "delete new tree; never merge into full_dataset without approval",
        "limitations": "Still AI labels; needs human sample for confirmation rates",
        "status": "AWAITING_APPROVAL",
    }
    write_json(dest / "FULL_FRAME_CLASSIFICATION_PROPOSAL.json", proposal)
    return summary


def run_stage1() -> dict[str, Any]:
    depths = reply_depth_inventory()
    audit = audit_tfidf_model()
    cand = two_stage_candidate_recovery()
    return {"depths": depths, "tfidf_audit": audit, "candidates": cand}
