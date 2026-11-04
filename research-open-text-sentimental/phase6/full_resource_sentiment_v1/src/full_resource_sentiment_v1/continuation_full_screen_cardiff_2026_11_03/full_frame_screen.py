"""Complete two-stage local screen of all 30,998 excluded nonprotected IDs."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

from full_resource_sentiment_v1.continuation_full_screen_cardiff_2026_11_03 import SEED
from full_resource_sentiment_v1.continuation_full_screen_cardiff_2026_11_03.common import (
    CAPTURE_DIR,
    DATASET_JSONL,
    LUNA_PREDS,
    OUT_ROOT,
    PILOT_IDS,
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

REL_THRESHOLD = 0.35
SHARD_SIZE = 1000


def _load_capture_texts(ids: set[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    remaining = set(ids)
    for path in sorted(CAPTURE_DIR.glob("page_*.json")):
        if not remaining:
            break
        page = int(path.stem.split("_")[1])
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("data") or []:
            rid = str(row.get("redditId") or "")
            if rid not in remaining:
                continue
            title = "" if row.get("title") is None else str(row.get("title"))
            body = "" if row.get("author_description") is None else str(row.get("author_description"))
            out[rid] = {
                "title": title,
                "body": body,
                "capture_page": page,
                "created_utc": row.get("created_utc"),
            }
            remaining.discard(rid)
    return out


def run_full_frame_screen() -> dict[str, Any]:
    dest = OUT_ROOT / "phase1_full_frame_screen"
    dest.mkdir(parents=True, exist_ok=True)
    shard_dir = RESTRICTED_ROOT / "full_frame_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    prot = protected_ids_only()
    frame = {str(r["reddit_id"]) for r in load_jsonl(DATASET_JSONL)}
    with PILOT_IDS.open(encoding="utf-8", newline="") as handle:
        pilot_ids = {str(r["reddit_id"]) for r in csv.DictReader(handle)}
    prior_rows = {str(r["reddit_id"]): r for r in load_jsonl(PRIOR_SCREEN)} if PRIOR_SCREEN.exists() else {}

    # Excluded universe
    excluded_meta: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(POST_SCORES):
        rid = str(row["reddit_id"])
        if rid in frame or rid in prot:
            continue
        excluded_meta[rid] = {
            "n_author": int(row.get("n_author_replies_nonempty") or 0),
            "n_community": int(row.get("n_community_replies_nonempty") or 0),
            "capture_page": row.get("capture_page"),
            "source": row.get("source"),
            "created_month": row.get("created_month"),
            "date_band": row.get("date_band"),
            "capture_id": row.get("capture_id"),
        }
    assert len(excluded_meta) == 30998, f"expected 30998 got {len(excluded_meta)}"

    # Train frozen models on frame labels excluding prot/pilot
    frame_text = {str(r["reddit_id"]): r for r in load_jsonl(TEXT_MANIFEST)}
    blocked = set(prot) | pilot_ids
    rel_X, rel_y, rel_g, rel_ids = [], [], [], []
    source_comp: Counter[str] = Counter()
    for r in load_jsonl(DATASET_JSONL):
        rid = str(r["reddit_id"])
        if rid in blocked:
            continue
        if r.get("final_relevance") not in {"relevant", "not_relevant"}:
            continue
        doc = frame_text.get(rid)
        if not doc:
            continue
        text = f"{doc.get('title') or ''}\n{doc.get('body') or ''}".strip()
        if not text:
            continue
        rel_X.append(text)
        rel_y.append(1 if r["final_relevance"] == "relevant" else 0)
        rel_g.append(sha256_text(text[:60])[:12])
        rel_ids.append(rid)
        source_comp[str(r.get("decision_source") or "unknown")] += 1

    pur_X, pur_y, pur_g, pur_ids = [], [], [], []
    label_to_i = {"TPS": 0, "GDS": 1, "Other": 2}
    pur_source: Counter[str] = Counter()
    pur_class: Counter[str] = Counter()
    for r in load_jsonl(DATASET_JSONL):
        rid = str(r["reddit_id"])
        if rid in blocked:
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
        pur_ids.append(rid)
        pur_source[str(r.get("decision_source") or "unknown")] += 1
        pur_class[r["final_discussion"]] += 1

    rel_y_arr = np.asarray(rel_y, dtype=int)
    pur_y_arr = np.asarray(pur_y, dtype=int)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    tr_rel, te_rel = next(gss.split(rel_X, rel_y_arr, rel_g))
    tr_pur, te_pur = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED).split(pur_X, pur_y_arr, pur_g))

    rel_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=40000, sublinear_tf=True)
    rel_clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=SEED)
    rel_clf.fit(rel_vec.fit_transform([rel_X[i] for i in tr_rel]), rel_y_arr[tr_rel])

    pur_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=30000, sublinear_tf=True)
    pur_clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=SEED)
    pur_clf.fit(pur_vec.fit_transform([pur_X[i] for i in tr_pur]), pur_y_arr[tr_pur])
    inv = {v: k for k, v in label_to_i.items()}

    model_card = {
        "utc": utcnow(),
        "seed": SEED,
        "rel_threshold": REL_THRESHOLD,
        "blocked_from_training": {"protected_n": len(prot), "pilot_n": len(pilot_ids)},
        "relevance_train": {
            "n_train": int(len(tr_rel)),
            "n_heldout_unused_for_screen": int(len(te_rel)),
            "train_ids_sha256": sha256_text("\n".join(sorted(rel_ids[i] for i in tr_rel))),
            "source_composition_universe": dict(source_comp),
        },
        "purpose_train": {
            "n_train": int(len(tr_pur)),
            "n_heldout_unused_for_screen": int(len(te_pur)),
            "train_ids_sha256": sha256_text("\n".join(sorted(pur_ids[i] for i in tr_pur))),
            "class_counts_universe": dict(pur_class),
            "source_composition_universe": dict(pur_source),
        },
        "group_isolation": "GroupShuffleSplit on title/body prefix hash; prot+pilot blocked",
        "rubric": "strict TPS/GDS/Other; stageB only if stageA relevant",
        "scientific_status": "AI predictions on mixed-source saved labels — not human-confirmed",
    }
    write_json(dest / "FROZEN_SCREEN_MODEL_CARD.json", model_card)

    # Load all texts
    print("  loading capture texts for 30998...", flush=True)
    texts = _load_capture_texts(set(excluded_meta))
    print(f"  loaded {len(texts)} texts", flush=True)

    luna = {}
    for row in load_jsonl(LUNA_PREDS):
        rid = str(row["reddit_id"])
        if rid in excluded_meta:
            luna[rid] = row.get("predicted_label")

    # Prior sample inclusion probabilities (author 0/1 strata)
    n_a0 = sum(1 for m in excluded_meta.values() if m["n_author"] == 0)
    n_a1 = sum(1 for m in excluded_meta.values() if m["n_author"] == 1)
    prior_inclusion = {
        "author_0_stratum_N": n_a0,
        "author_1_stratum_N": n_a1,
        "author_ge2_stratum_N": sum(1 for m in excluded_meta.values() if m["n_author"] >= 2),
        "prior_broad_sample_n_each": 800,
        "prior_inclusion_prob_author_0": 800 / n_a0 if n_a0 else None,
        "prior_inclusion_prob_author_1": 800 / n_a1 if n_a1 else None,
        "prior_author_ge2_prob": 1.0,
        "note": "Full-frame now p=1 for all; prior probs retained for weighted analyses of prior subsample only",
    }
    write_json(dest / "PRIOR_SAMPLE_INCLUSION_PROBS.json", prior_inclusion)

    all_ids = sorted(excluded_meta.keys())
    results: list[dict[str, Any]] = []
    shard_rows: list[dict[str, Any]] = []
    shard_i = 0

    def flush_shard() -> None:
        nonlocal shard_i, shard_rows
        if not shard_rows:
            return
        path = shard_dir / f"shard_{shard_i:04d}.jsonl"
        write_jsonl(path, shard_rows)
        shard_i += 1
        shard_rows = []

    batch_size = 500
    for start in range(0, len(all_ids), batch_size):
        batch_ids = all_ids[start : start + batch_size]
        batch_meta = []
        batch_texts = []
        for rid in batch_ids:
            meta = excluded_meta[rid]
            doc = texts.get(rid) or {"title": "", "body": "", "capture_page": meta.get("capture_page")}
            title, body = doc.get("title") or "", doc.get("body") or ""
            text = f"{title}\n{body}".strip()
            batch_meta.append((rid, meta, doc, title, text))
            batch_texts.append(text)

        nonempty_idx = [i for i, t in enumerate(batch_texts) if t.strip()]
        rel_probs = [None] * len(batch_texts)
        if nonempty_idx:
            X = rel_vec.transform([batch_texts[i] for i in nonempty_idx])
            probs = rel_clf.predict_proba(X)[:, 1]
            for j, i in enumerate(nonempty_idx):
                rel_probs[i] = float(probs[j])

        relevant_idx = [i for i in nonempty_idx if rel_probs[i] is not None and rel_probs[i] >= REL_THRESHOLD]
        purpose_preds: dict[int, tuple[str, float]] = {}
        if relevant_idx:
            Xp = pur_vec.transform([batch_texts[i] for i in relevant_idx])
            proba = pur_clf.predict_proba(Xp)
            for j, i in enumerate(relevant_idx):
                idx = int(np.argmax(proba[j]))
                purpose_preds[i] = (inv[idx], float(proba[j][idx]))

        for i, (rid, meta, doc, title, text) in enumerate(batch_meta):
            na = meta["n_author"]
            bucket = "ge4" if na >= 4 else ("ge2" if na >= 2 else ("ge1" if na == 1 else "ge0"))
            err = None
            if not text.strip():
                tfidf_rel, prob = "unassessable", None
                purpose_model = purpose_prob = None
                status = "unassessable_empty_text"
            else:
                prob = rel_probs[i]
                tfidf_rel = "relevant" if prob is not None and prob >= REL_THRESHOLD else "not_relevant"
                purpose_model = purpose_prob = None
                if tfidf_rel == "relevant" and i in purpose_preds:
                    purpose_model, purpose_prob = purpose_preds[i]
                if tfidf_rel == "relevant" and purpose_model in {"TPS", "GDS"}:
                    status = "candidate_ai_only"
                elif tfidf_rel == "relevant" and purpose_model == "Other":
                    status = "candidate_relevant_other_only"
                else:
                    status = "not_candidate"

            row = {
                "reddit_id": rid,
                "n_author": na,
                "n_community": meta["n_community"],
                "author_depth_bucket": bucket,
                "capture_page": doc.get("capture_page")
                if doc.get("capture_page") is not None
                else meta.get("capture_page"),
                "capture_id": meta.get("capture_id"),
                "source": meta.get("source"),
                "created_month": meta.get("created_month"),
                "date_band": meta.get("date_band"),
                "created_utc": doc.get("created_utc"),
                "luna_relevance": luna.get(rid),
                "stageA_relevance_pred": tfidf_rel,
                "stageA_prob_relevant": prob,
                "stageB_purpose_pred": purpose_model,
                "stageB_purpose_prob": purpose_prob,
                "recovery_status": status,
                "error": err,
                "confirmed": False,
                "human_validated": False,
                "in_prior_9882_screen": rid in prior_rows,
                "in_fixed_pilot_240": rid in pilot_ids,
                "prior_recovery_status": (prior_rows.get(rid) or {}).get("recovery_status"),
                "prior_stageB_purpose_pred": (prior_rows.get(rid) or {}).get("stageB_purpose_pred"),
                "text_excerpt": (title or text)[:160],
                "text_chars": len(text),
                "screen_version": "full_frame_tfidf_v1_seed_20261103",
                "note": "AI candidate ≠ confirmed; not human-validated; not overwrite of 6425",
            }
            results.append(row)
            shard_rows.append(row)
            if len(shard_rows) >= SHARD_SIZE:
                flush_shard()
        if (start + len(batch_ids)) % 5000 == 0 or start + len(batch_ids) == len(all_ids):
            print(f"  scored {start + len(batch_ids)}/{len(all_ids)}", flush=True)
    flush_shard()

    # Combined restricted table
    write_jsonl(RESTRICTED_ROOT / "full_frame_screen_30998.jsonl", results)

    by_depth = defaultdict(Counter)
    by_month = Counter()
    by_source = Counter()
    by_status = Counter()
    by_purpose = Counter()
    for r in results:
        by_status[r["recovery_status"]] += 1
        by_month[str(r.get("created_month") or "unknown")] += 1
        by_source[str(r.get("source") or "unknown")] += 1
        if r["recovery_status"] == "candidate_ai_only":
            by_depth[r["author_depth_bucket"]][r["stageB_purpose_pred"]] += 1
            by_purpose[r["stageB_purpose_pred"]] += 1

    n_cand = by_status.get("candidate_ai_only", 0)
    n_tps = by_purpose.get("TPS", 0)
    n_gds = by_purpose.get("GDS", 0)

    # Agreement with prior screen on overlap
    overlap = [r for r in results if r["in_prior_9882_screen"]]
    agree_status = sum(1 for r in overlap if r["recovery_status"] == r.get("prior_recovery_status"))
    agree_purpose = sum(
        1
        for r in overlap
        if r["recovery_status"] == "candidate_ai_only"
        and r.get("stageB_purpose_pred") == r.get("prior_stageB_purpose_pred")
    )

    summary = {
        "utc": utcnow(),
        "n_screened": len(results),
        "expected": 30998,
        "match": len(results) == 30998,
        "n_texts_loaded": len(texts),
        "n_missing_text_in_capture": len(excluded_meta) - len(texts),
        "by_recovery_status": dict(by_status),
        "n_candidate_tps_gds_ai_only": n_cand,
        "n_candidate_tps": n_tps,
        "n_candidate_gds": n_gds,
        "n_confirmed": 0,
        "by_author_depth_candidates": {k: dict(v) for k, v in by_depth.items()},
        "by_created_month_all_screened": dict(by_month),
        "by_capture_source_all_screened": dict(by_source),
        "prior_9882": {
            "n_identifiable_in_full_frame": len(overlap),
            "status_agreement_with_prior_row": agree_status,
            "candidate_purpose_agreement": agree_purpose,
            "not_independent_validation": True,
        },
        "fixed_pilot_240": {
            "n_in_excluded_and_screened": sum(1 for r in results if r["in_fixed_pilot_240"]),
            "not_independent_validation": True,
            "pilot_ids_sha256": sha256_file(PILOT_IDS),
        },
        "protected_excluded": len(prot),
        "model_card": "FROZEN_SCREEN_MODEL_CARD.json",
        "shards": shard_i,
        "scientific_status": "ALL outputs AI-only candidates; zero human-confirmed; not merged into original 6425",
        "post_scores_sha256": sha256_file(POST_SCORES),
        "dataset_jsonl_sha256": sha256_file(DATASET_JSONL),
    }
    write_json(dest / "FULL_FRAME_SCREEN_SUMMARY.json", summary)
    return summary
