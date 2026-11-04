"""Priority + broad-sample two-stage AI screening of excluded posts (candidates only; no merge).

Full 30,998 classification remains an approval-gated proposal (Codex gate).
This run screens: all author>=2 (8,147) + stratified samples of author 0/1 for FN coverage.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

from full_resource_sentiment_v1.exact_join_repair_2026_11_03 import SEED
from full_resource_sentiment_v1.exact_join_repair_2026_11_03.common import (
    CAPTURE_DIR,
    DATASET_JSONL,
    LUNA_PREDS,
    OUT_ROOT,
    PILOT_IDS,
    POST_SCORES,
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

UPDATE_CUES = re.compile(
    r"\b(update|upgrade|patch|rollback|firmware|version|changelog|release notes?|kernel)\b", re.I
)


def _load_texts_for_ids(ids: set[str]) -> dict[str, dict[str, str]]:
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


def run_recovery_screen() -> dict[str, Any]:
    dest = OUT_ROOT / "phase2_recovery_screen"
    dest.mkdir(parents=True, exist_ok=True)
    prot = protected_ids_only()
    # Anything already in the saved full_dataset is not part of the excluded recovery pool
    frame = {str(r["reddit_id"]) for r in load_jsonl(DATASET_JSONL)}

    with PILOT_IDS.open(encoding="utf-8", newline="") as handle:
        pilot_ids = {str(r["reddit_id"]) for r in csv.DictReader(handle)}

    # Inventory excluded nonprotected
    excluded: list[dict[str, Any]] = []
    for row in load_jsonl(POST_SCORES):
        rid = str(row["reddit_id"])
        if rid in frame or rid in prot:
            continue
        na = int(row.get("n_author_replies_nonempty") or 0)
        excluded.append({"reddit_id": rid, "n_author": na, "capture_page": row.get("capture_page")})

    by_depth = Counter()
    for e in excluded:
        na = e["n_author"]
        by_depth["total"] += 1
        if na >= 2:
            by_depth["author_ge2"] += 1
        elif na == 1:
            by_depth["author_1"] += 1
        else:
            by_depth["author_0"] += 1

    # Screening universe: all author>=2 + samples of 0/1 + always include pilot 240 if excluded
    rng = np.random.default_rng(SEED + 77)
    ge2 = [e for e in excluded if e["n_author"] >= 2]
    a1 = [e for e in excluded if e["n_author"] == 1]
    a0 = [e for e in excluded if e["n_author"] == 0]
    n_broad_each = 800  # broad FN/coverage sample; not full frame
    sample_a1 = list(rng.choice(len(a1), size=min(n_broad_each, len(a1)), replace=False)) if a1 else []
    sample_a0 = list(rng.choice(len(a0), size=min(n_broad_each, len(a0)), replace=False)) if a0 else []
    screen_ids = {e["reddit_id"] for e in ge2}
    screen_ids |= {a1[i]["reddit_id"] for i in sample_a1}
    screen_ids |= {a0[i]["reddit_id"] for i in sample_a0}
    screen_ids |= {rid for rid in pilot_ids if rid not in frame and rid not in prot}
    # never include protected
    screen_ids -= prot

    texts = _load_texts_for_ids(screen_ids)

    # Train relevance binary: frame relevant vs sample of non-frame excluded Luna-not_relevant? 
    # Use dataset: relevant vs not from saved labels in full_dataset where available,
    # plus treat frozen TPS/GDS/Other relevant as positive.
    frame_text = {str(r["reddit_id"]): r for r in load_jsonl(TEXT_MANIFEST)}
    rel_X, rel_y, rel_g = [], [], []
    for r in load_jsonl(DATASET_JSONL):
        rid = str(r["reddit_id"])
        if rid in prot or rid in pilot_ids:
            continue
        doc = frame_text.get(rid)
        if not doc:
            continue
        text = f"{doc.get('title') or ''}\n{doc.get('body') or ''}".strip()
        if not text:
            continue
        y = 1 if r.get("final_relevance") == "relevant" else 0
        # Only use rows with explicit relevance decisions
        if r.get("final_relevance") not in {"relevant", "not_relevant"}:
            continue
        rel_X.append(text)
        rel_y.append(y)
        rel_g.append(sha256_text(text[:60])[:12])

    # Purpose model on relevant TPS/GDS/Other
    pur_X, pur_y, pur_g = [], [], []
    label_to_i = {"TPS": 0, "GDS": 1, "Other": 2}
    for r in load_jsonl(DATASET_JSONL):
        rid = str(r["reddit_id"])
        if rid in prot or rid in pilot_ids:
            continue
        if r.get("final_relevance") != "relevant":
            continue
        if r.get("final_discussion") not in label_to_i:
            continue
        doc = frame_text.get(rid) or {"title": "", "body": ""}
        text = f"{doc.get('title') or ''}\n{doc.get('body') or ''}".strip()
        if not text:
            continue
        pur_X.append(text)
        pur_y.append(label_to_i[r["final_discussion"]])
        pur_g.append(sha256_text(text[:60])[:12])

    rel_y_arr = np.asarray(rel_y, dtype=int)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    if len(rel_X) >= 50 and len(set(rel_y)) > 1:
        tr, _va = next(gss.split(rel_X, rel_y_arr, rel_g))
        rel_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=40000, sublinear_tf=True)
        rel_clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=SEED)
        rel_clf.fit(rel_vec.fit_transform([rel_X[i] for i in tr]), rel_y_arr[tr])
        rel_threshold = 0.35  # match prior card
    else:
        # Fallback: keyword gate only
        rel_vec = rel_clf = None
        rel_threshold = None

    pur_y_arr = np.asarray(pur_y, dtype=int)
    pur_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=30000, sublinear_tf=True)
    pur_clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=SEED)
    trp, _ = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED).split(pur_X, pur_y_arr, pur_g))
    pur_clf.fit(pur_vec.fit_transform([pur_X[i] for i in trp]), pur_y_arr[trp])
    inv = {v: k for k, v in label_to_i.items()}

    luna = {}
    for row in load_jsonl(LUNA_PREDS):
        rid = str(row["reddit_id"])
        if rid in screen_ids:
            luna[rid] = row.get("predicted_label")

    na_map = {e["reddit_id"]: e["n_author"] for e in excluded}
    candidates = []
    for rid in sorted(screen_ids):
        doc = texts.get(rid) or {"title": "", "body": ""}
        title, body = doc.get("title") or "", doc.get("body") or ""
        text = f"{title}\n{body}".strip()
        na = int(na_map.get(rid, -1))
        if rel_clf is not None and text:
            prob = float(rel_clf.predict_proba(rel_vec.transform([text]))[0][1])
            tfidf_rel = "relevant" if prob >= rel_threshold else "not_relevant"
        else:
            prob = None
            tfidf_rel = "relevant" if UPDATE_CUES.search(text) else "not_relevant"
        purpose_model = purpose_prob = None
        if tfidf_rel == "relevant" and text:
            proba = pur_clf.predict_proba(pur_vec.transform([text]))[0]
            idx = int(np.argmax(proba))
            purpose_model = inv[idx]
            purpose_prob = float(proba[idx])
        is_cand = tfidf_rel == "relevant" and purpose_model in {"TPS", "GDS"}
        status = "candidate_ai_only" if is_cand else "not_candidate"
        if tfidf_rel == "relevant" and purpose_model == "Other":
            status = "candidate_relevant_other_only"
        bucket = "ge2" if na >= 2 else ("ge1" if na == 1 else ("ge0" if na == 0 else "unknown"))
        cohort = "priority_author_ge2" if na >= 2 else ("broad_sample_author_1" if na == 1 else "broad_sample_author_0")
        if rid in pilot_ids:
            cohort = "fixed_pilot_240_overlap"
        candidates.append(
            {
                "reddit_id": rid,
                "cohort": cohort,
                "n_author": na,
                "author_depth_bucket": bucket,
                "luna_relevance": luna.get(rid),
                "stageA_relevance_pred": tfidf_rel,
                "stageA_prob_relevant": prob,
                "stageB_purpose_pred": purpose_model,
                "stageB_purpose_prob": purpose_prob,
                "recovery_status": status,
                "confirmed": False,
                "human_validated": False,
                "text_excerpt": (title or text)[:160],
                "note": "AI candidate ≠ confirmed; not merged",
            }
        )

    write_jsonl(RESTRICTED_ROOT / "recovery_candidates_priority_broad.jsonl", candidates)

    by_cohort = defaultdict(Counter)
    by_bucket = defaultdict(Counter)
    for c in candidates:
        if c["recovery_status"] == "candidate_ai_only":
            by_cohort[c["cohort"]][c["stageB_purpose_pred"]] += 1
            by_bucket[c["author_depth_bucket"]][c["stageB_purpose_pred"]] += 1

    n_cand = sum(1 for c in candidates if c["recovery_status"] == "candidate_ai_only")
    n_tps = sum(1 for c in candidates if c["recovery_status"] == "candidate_ai_only" and c["stageB_purpose_pred"] == "TPS")
    n_gds = sum(1 for c in candidates if c["recovery_status"] == "candidate_ai_only" and c["stageB_purpose_pred"] == "GDS")

    summary = {
        "utc": utcnow(),
        "excluded_inventory": dict(by_depth),
        "expected_excluded_total": 30998,
        "inventory_match": by_depth["total"] == 30998 and by_depth["author_ge2"] == 8147,
        "screened_n": len(candidates),
        "screen_design": {
            "priority_author_ge2_all": by_depth["author_ge2"],
            "broad_sample_author_1": len(sample_a1),
            "broad_sample_author_0": len(sample_a0),
            "pilot_overlap_included": True,
            "full_30998": False,
            "rationale": "Codex gate blocks full-frame; user priority is author>=2 with broad FN coverage samples",
        },
        "n_candidate_tps_gds_ai_only": n_cand,
        "n_candidate_tps": n_tps,
        "n_candidate_gds": n_gds,
        "n_confirmed": 0,
        "by_cohort_candidates": {k: dict(v) for k, v in by_cohort.items()},
        "by_author_depth_candidates": {k: dict(v) for k, v in by_bucket.items()},
        "protected_excluded_from_screen": len(prot),
        "pilot_ids_sha256": sha256_file(PILOT_IDS),
        "scientific_status": "CANDIDATES only; zero confirmed; not merged into 6,425",
        "selection_limitations": [
            "Stage-A/B trained on mixed-source saved labels",
            "Broad samples of author 0/1 are not a complete screen of those strata",
            "No independent human FP/FN rates in this run",
            "Threshold 0.35 inherited from prior TF-IDF card when classifier available",
        ],
        "merge_status": "BLOCKED pending dataset version audit + approval",
    }
    write_json(dest / "RECOVERY_SCREEN_SUMMARY.json", summary)

    # Full-frame remains proposal
    write_json(
        dest / "FULL_FRAME_30998_PROPOSAL.json",
        {
            "utc": utcnow(),
            "action": "Extend two-stage local TF-IDF to all 30,998 nonprotected excluded IDs",
            "n_ids": 30998,
            "already_screened_this_run": len(candidates),
            "remaining_if_approved": 30998 - len({c["reddit_id"] for c in candidates}),
            "est_time_cpu_min": "15–45",
            "money": 0,
            "status": "AWAITING_APPROVAL",
            "merge": "still blocked after classification until human audit approval",
        },
    )
    return summary
