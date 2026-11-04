"""Phase 2: excluded-frame inventory + bounded local TF-IDF relevance screening."""

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

from full_resource_sentiment_v1.strict_continuation_2026_11_03 import SEED
from full_resource_sentiment_v1.strict_continuation_2026_11_03.common import (
    CAPTURE_DIR,
    DATASET_JSONL,
    LUNA_PREDS,
    OUT_ROOT,
    PILOT_IDS,
    PILOT_STRATA,
    POST_SCORES,
    RESTRICTED_ROOT,
    load_jsonl,
    protected_ids,
    sha256_file,
    sha256_text,
    utcnow,
    write_json,
    write_jsonl,
)

EXPECTED_EXCLUDED = {
    "nonprotected_excluded": 30998,
    "author_ge1": 14918,
    "author_ge2": 8147,
    "author_ge4": 2847,
    "community_ge1": 24268,
}


def _load_pilot() -> list[dict[str, Any]]:
    with PILOT_IDS.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _capture_texts(ids: set[str]) -> dict[str, dict[str, str]]:
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
            prev = out.get(rid)
            if prev is None or len(body) > len(prev.get("body") or ""):
                out[rid] = {"title": title, "body": body}
            remaining.discard(rid)
    return out


def _select_targeted_200(
    pool: list[dict[str, Any]],
    *,
    exclude_ids: set[str],
    seed: int = SEED,
    n: int = 200,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fixed-seed targeted diagnostic sample enriched for author-rich strata.

    Scheme (predeclared):
    - Stratum A: author_ge4 (quota 100)
    - Stratum B: author_ge2 & <4 (quota 50)
    - Stratum C: author_ge1 & <2 (quota 30)
    - Stratum D: author_ge0 with body available (quota 20)
    Within stratum: RNG sample without replacement; selection probability = quota/stratum_N
    (or 1 if N<=quota). Not a population probability sample of all 30,998.
    """
    rng = np.random.default_rng(seed)
    eligible = [r for r in pool if r["reddit_id"] not in exclude_ids]
    quotas = {"A_author_ge4": 100, "B_author_ge2": 50, "C_author_ge1": 30, "D_author_0_with_body": 20}
    buckets: dict[str, list[dict[str, Any]]] = {k: [] for k in quotas}
    for r in eligible:
        na = int(r["n_author"])
        if na >= 4:
            buckets["A_author_ge4"].append(r)
        elif na >= 2:
            buckets["B_author_ge2"].append(r)
        elif na >= 1:
            buckets["C_author_ge1"].append(r)
        elif r.get("body_available"):
            buckets["D_author_0_with_body"].append(r)

    selected: list[dict[str, Any]] = []
    design = {"seed": seed, "quotas": quotas, "stratum_N": {}, "selection_probs": {}}
    for name, quota in quotas.items():
        bucket = buckets[name]
        design["stratum_N"][name] = len(bucket)
        take = min(quota, len(bucket))
        if take == 0:
            design["selection_probs"][name] = 0.0
            continue
        idx = rng.choice(len(bucket), size=take, replace=False)
        prob = take / len(bucket)
        design["selection_probs"][name] = float(prob)
        for i in idx:
            row = dict(bucket[int(i)])
            row["target_stratum"] = name
            row["selection_probability"] = float(prob)
            row["sample_weight"] = float(1.0 / prob) if prob > 0 else None
            row["sample_type"] = "targeted_diagnostic_200"
            selected.append(row)
    design["n_selected"] = len(selected)
    design["note"] = (
        "Targeted diagnostic enrichment for author-rich strata; not interchangeable with the fixed 240 IPW sample."
    )
    return selected, design


def _train_tfidf_relevance(
    frame_rows: list[dict[str, Any]],
    texts: dict[str, dict[str, str]],
    *,
    blocked_ids: set[str],
) -> tuple[Any, Any, dict[str, Any]]:
    """Train-only TF-IDF relevance classifier on frame labels; exclude blocked IDs from training."""
    X_text = []
    y = []
    groups = []
    train_ids = []
    label_map = {"relevant": 1, "not_relevant": 0}
    provenance = Counter()
    skipped = Counter()
    for r in frame_rows:
        rid = str(r["reddit_id"])
        if rid in blocked_ids:
            skipped["blocked_id"] += 1
            continue
        rel = r.get("final_relevance")
        if rel not in label_map:
            skipped[f"skip_relevance_{rel}"] += 1
            continue
        doc = texts.get(rid) or {"title": "", "body": ""}
        text = f"{doc.get('title') or ''}\n{doc.get('body') or ''}".strip()
        if not text:
            skipped["empty_text"] += 1
            continue
        X_text.append(text)
        y.append(label_map[rel])
        groups.append(sha256_text((doc.get("title") or "")[:80])[:12])
        train_ids.append(rid)
        provenance[str(r.get("decision_source"))] += 1

    y_arr = np.asarray(y, dtype=int)
    # Hold out a small grouped val inside training for threshold only (not the excluded screening set)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    tr_idx, va_idx = next(gss.split(X_text, y_arr, groups))
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3, max_features=40000, sublinear_tf=True)
    Xtr = vec.fit_transform([X_text[i] for i in tr_idx])
    Xva = vec.transform([X_text[i] for i in va_idx])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=SEED)
    clf.fit(Xtr, y_arr[tr_idx])
    # threshold on val for macro relevance F1 proxy (binary)
    from sklearn.metrics import f1_score

    prob = clf.predict_proba(Xva)[:, 1]
    best_t, best_f = 0.5, -1.0
    for t in np.linspace(0.1, 0.9, 17):
        f = f1_score(y_arr[va_idx], (prob >= t).astype(int), average="binary", zero_division=0)
        if f > best_f:
            best_f, best_t = float(f), float(t)
    meta = {
        "model_id": "tfidf_logreg_relevance_v1_strict_continuation",
        "version": "2026-11-03",
        "features": "title+body TF-IDF word/bigram; no labels/sentiment/replies in features",
        "training_n": int(len(tr_idx)),
        "val_n": int(len(va_idx)),
        "training_positive_relevant": int(y_arr[tr_idx].sum()),
        "threshold_selected_on_val": best_t,
        "val_pos_f1_at_threshold": best_f,
        "decision_source_counts_in_usable_pool": dict(provenance),
        "skipped": dict(skipped),
        "blocked_ids_excluded_from_training": sorted(blocked_ids)[:20],
        "n_blocked_excluded": len(blocked_ids),
        "leakage_controls": {
            "pilot_240_not_in_training": True,
            "protected_not_in_training": True,
            "screening_targets_not_in_training": True,
            "note": "Frame mixed-source labels; agreement ≠ accuracy.",
        },
        "seed": SEED,
    }
    return vec, clf, meta


def run_phase2() -> dict[str, Any]:
    dest = OUT_ROOT / "phase2_excluded_screening"
    dest.mkdir(parents=True, exist_ok=True)
    prot = protected_ids()
    frame_rows = load_jsonl(DATASET_JSONL)
    frame_ids = {str(r["reddit_id"]) for r in frame_rows}
    post_scores = {str(r["reddit_id"]): r for r in load_jsonl(POST_SCORES)}
    pilot = _load_pilot()
    pilot_ids = {str(r["reddit_id"]) for r in pilot}
    assert len(pilot_ids) == 240
    pilot_hash = sha256_file(PILOT_IDS)
    strata_hash = sha256_file(PILOT_STRATA)

    # Inventory all nonprotected excluded
    inventory = []
    labels = Counter()
    author_ge = Counter()
    for row in load_jsonl(LUNA_PREDS):
        rid = str(row["reddit_id"])
        if rid in frame_ids or rid in prot:
            continue
        ps = post_scores.get(rid) or {}
        na = int(ps.get("n_author_replies_nonempty") or 0)
        nc = int(ps.get("n_community_replies_nonempty") or 0)
        lab = row.get("predicted_label")
        if lab is None and row.get("response_status") == "error":
            lab = "error"
        labels[str(lab)] += 1
        for thr in (1, 2, 4):
            if na >= thr:
                author_ge[f"author_ge{thr}"] += 1
        if nc >= 1:
            author_ge["community_ge1"] += 1
        period = "october" if str(ps.get("created_month") or "").startswith("2026-10") else (
            "older" if ps.get("created_month") else "unknown"
        )
        # date_band / selection from expansion: use created_month heuristic; also check luna created_month
        if row.get("created_month"):
            period = "october" if str(row["created_month"]).startswith("2026-10") else "older"
        inventory.append(
            {
                "reddit_id": rid,
                "luna_predicted_label": lab,
                "luna_model_id": row.get("model_id") or row.get("resolved_model"),
                "luna_prompt_version": row.get("prompt_version"),
                "n_author": na,
                "n_community": nc,
                "body_available": bool(ps.get("body_nonempty")),
                "title_available": bool(ps.get("title_nonempty")),
                "count_gap": ps.get("count_gap"),
                "created_month": ps.get("created_month") or row.get("created_month"),
                "period": period,
                "in_fixed_pilot_240": rid in pilot_ids,
            }
        )

    recon = {
        "n_inventory": len(inventory),
        "expected": EXPECTED_EXCLUDED,
        "observed": {
            "nonprotected_excluded": len(inventory),
            "author_ge1": author_ge["author_ge1"],
            "author_ge2": author_ge["author_ge2"],
            "author_ge4": author_ge["author_ge4"],
            "community_ge1": author_ge["community_ge1"],
        },
        "luna_label_counts": dict(labels),
        "protected_overlap_in_inventory": 0,  # constructed by exclusion
        "pilot_ids_sha256": pilot_hash,
        "pilot_strata_sha256": strata_hash,
        "pilot_preserved": True,
        "match_expected": len(inventory) == EXPECTED_EXCLUDED["nonprotected_excluded"]
        and author_ge["author_ge1"] == EXPECTED_EXCLUDED["author_ge1"]
        and author_ge["author_ge2"] == EXPECTED_EXCLUDED["author_ge2"]
        and author_ge["author_ge4"] == EXPECTED_EXCLUDED["author_ge4"],
    }
    write_json(dest / "EXCLUDED_INVENTORY_SUMMARY.json", recon)

    # Public aggregate only (no full ID dump in public); restricted ID list optional thin
    write_jsonl(
        RESTRICTED_ROOT / "excluded_30998_inventory_thin.jsonl",
        (
            {
                "reddit_id": r["reddit_id"],
                "luna_predicted_label": r["luna_predicted_label"],
                "n_author": r["n_author"],
                "n_community": r["n_community"],
                "period": r["period"],
                "in_fixed_pilot_240": r["in_fixed_pilot_240"],
            }
            for r in inventory
        ),
    )

    # Targeted 200 (exclude pilot + protected; protected already out)
    targeted, design = _select_targeted_200(inventory, exclude_ids=pilot_ids | prot, seed=SEED, n=200)
    write_json(dest / "TARGETED_200_DESIGN.json", design)
    write_jsonl(
        dest / "targeted_200_public.jsonl",
        (
            {
                "reddit_id": r["reddit_id"],
                "target_stratum": r["target_stratum"],
                "selection_probability": r["selection_probability"],
                "sample_weight": r["sample_weight"],
                "n_author": r["n_author"],
                "luna_predicted_label": r["luna_predicted_label"],
                "period": r["period"],
            }
            for r in targeted
        ),
    )

    # Training exposure audit
    exposure = {
        "pilot_240_in_frame": sorted(pilot_ids & frame_ids),
        "n_pilot_in_frame": len(pilot_ids & frame_ids),
        "protected_in_frame": len(prot & frame_ids),
        "targeted_in_frame": len({r["reddit_id"] for r in targeted} & frame_ids),
        "training_label_source": "full_dataset final_relevance among non-blocked frame IDs",
        "note": "Pilot/targeted/protected blocked from TF-IDF training. Labels are mixed-source.",
    }
    write_json(dest / "TRAINING_EXPOSURE_AUDIT.json", exposure)

    # Texts for training + screening
    need_ids = frame_ids | pilot_ids | {r["reddit_id"] for r in targeted}
    texts = _capture_texts(need_ids)
    # Also merge frame restricted manifest if present for frame texts
    man_path = Path(
        "/Users/manumathewjiss/Documents/Research-on-Software-Updates/research-open-text-sentimental/"
        "phase6/full_resource_sentiment_v1/restricted/master_research_2026-11-03/text_thread_manifest_restricted.jsonl"
    )
    if man_path.exists():
        for r in load_jsonl(man_path):
            rid = str(r["reddit_id"])
            if rid in frame_ids and rid not in texts:
                texts[rid] = {"title": r.get("title") or "", "body": r.get("body") or ""}

    blocked = set(pilot_ids) | set(prot) | {r["reddit_id"] for r in targeted}
    vec, clf, model_meta = _train_tfidf_relevance(frame_rows, texts, blocked_ids=blocked)
    write_json(dest / "TFIDF_RELEVANCE_MODEL_CARD.json", model_meta)

    def predict_batch(rows: list[dict[str, Any]], sample_tag: str) -> list[dict[str, Any]]:
        out = []
        for r in rows:
            rid = str(r["reddit_id"] if "reddit_id" in r else r.get("reddit_id"))
            # pilot rows from CSV
            if "reddit_id" not in r and "prediction_stratum" in r:
                rid = str(r["reddit_id"])
            doc = texts.get(rid) or {"title": "", "body": ""}
            text = f"{doc.get('title') or ''}\n{doc.get('body') or ''}".strip()
            if not text:
                out.append(
                    {
                        "reddit_id": rid,
                        "sample": sample_tag,
                        "tfidf_relevance": "unassessable",
                        "tfidf_prob_relevant": None,
                        "error": "empty_text",
                        "model_id": model_meta["model_id"],
                        "model_version": model_meta["version"],
                    }
                )
                continue
            X = vec.transform([text])
            prob = float(clf.predict_proba(X)[0, 1])
            thr = float(model_meta["threshold_selected_on_val"])
            pred = "relevant" if prob >= thr else "not_relevant"
            # uncertainty: distance to threshold
            unc = abs(prob - thr)
            out.append(
                {
                    "reddit_id": rid,
                    "sample": sample_tag,
                    "tfidf_relevance": pred,
                    "tfidf_prob_relevant": prob,
                    "threshold": thr,
                    "uncertainty_margin": unc,
                    "low_confidence": unc < 0.1,
                    "evidence_quote": (doc.get("title") or text)[:160],
                    "error": None,
                    "model_id": model_meta["model_id"],
                    "model_version": model_meta["version"],
                    "n_author": r.get("n_author") or r.get("author_replies_ge4"),
                }
            )
        return out

    # Pilot predictions (preserve Luna join)
    luna_by = {str(r["reddit_id"]): r for r in load_jsonl(LUNA_PREDS) if str(r["reddit_id"]) in pilot_ids}
    pilot_rows = []
    for p in pilot:
        rid = str(p["reddit_id"])
        meta = next((x for x in inventory if x["reddit_id"] == rid), None)
        pilot_rows.append(
            {
                "reddit_id": rid,
                "n_author": (meta or {}).get("n_author"),
                "luna_predicted_label": (luna_by.get(rid) or {}).get("predicted_label")
                or (meta or {}).get("luna_predicted_label"),
            }
        )
    tfidf_pilot = predict_batch(pilot_rows, "fixed_pilot_240")
    tfidf_target = predict_batch(targeted, "targeted_diagnostic_200")

    # Join disagreements
    def join_disagreements(preds: list[dict[str, Any]], luna_map: dict[str, Any]) -> tuple[list[dict], dict]:
        disagree = []
        agree = 0
        compared = 0
        for p in preds:
            rid = p["reddit_id"]
            luna = luna_map.get(rid) or {}
            luna_lab = luna.get("predicted_label") or luna.get("luna_predicted_label")
            if luna_lab is None:
                continue
            compared += 1
            # Map luna error/unclear/unassessable as distinct
            same = str(luna_lab) == str(p["tfidf_relevance"])
            if same:
                agree += 1
            else:
                disagree.append(
                    {
                        "reddit_id": rid,
                        "luna_predicted_label": luna_lab,
                        "tfidf_relevance": p["tfidf_relevance"],
                        "tfidf_prob_relevant": p["tfidf_prob_relevant"],
                        "sample": p["sample"],
                        "n_author": p.get("n_author"),
                        "low_confidence": p.get("low_confidence"),
                    }
                )
        return disagree, {"compared": compared, "agree": agree, "disagree": len(disagree)}

    luna_pilot_map = {r["reddit_id"]: r for r in pilot_rows}
    luna_target_map = {r["reddit_id"]: r for r in targeted}
    d_pilot, s_pilot = join_disagreements(tfidf_pilot, luna_pilot_map)
    d_targ, s_targ = join_disagreements(tfidf_target, luna_target_map)

    write_jsonl(RESTRICTED_ROOT / "tfidf_predictions_pilot240.jsonl", tfidf_pilot)
    write_jsonl(RESTRICTED_ROOT / "tfidf_predictions_targeted200.jsonl", tfidf_target)
    write_jsonl(dest / "disagreements_pilot240.jsonl", d_pilot)
    write_jsonl(dest / "disagreements_targeted200.jsonl", d_targ)

    # Review-priority queues (public: IDs + flags only)
    queues = {"possible_relevant_author_rich": [], "luna_not_relevant_tfidf_relevant": [], "both_relevant": []}
    for p in tfidf_pilot + tfidf_target:
        rid = p["reddit_id"]
        luna_lab = (luna_pilot_map.get(rid) or luna_target_map.get(rid) or {}).get("luna_predicted_label") or (
            luna_pilot_map.get(rid) or luna_target_map.get(rid) or {}
        ).get("predicted_label")
        na = int(p.get("n_author") or 0)
        item = {
            "reddit_id": rid,
            "sample": p["sample"],
            "tfidf_relevance": p["tfidf_relevance"],
            "luna_predicted_label": luna_lab,
            "n_author": na,
            "tfidf_prob_relevant": p.get("tfidf_prob_relevant"),
            "priority_note": "Relevance disagreement/queue only — not purpose (TPS/GDS/Other) assertion.",
        }
        if p["tfidf_relevance"] == "relevant" and na >= 1:
            queues["possible_relevant_author_rich"].append(item)
        if luna_lab == "not_relevant" and p["tfidf_relevance"] == "relevant":
            queues["luna_not_relevant_tfidf_relevant"].append(item)
        if luna_lab == "relevant" and p["tfidf_relevance"] == "relevant":
            queues["both_relevant"].append(item)
    # sort author-rich by prob desc
    queues["possible_relevant_author_rich"].sort(key=lambda x: (-(x.get("n_author") or 0), -(x.get("tfidf_prob_relevant") or 0)))
    write_json(dest / "REVIEW_PRIORITY_QUEUES.json", {k: v[:100] for k, v in queues.items()})

    # NLI/transformer blocker note
    nli_blocker = {
        "status": "not_run",
        "reason": "torch/transformers not installed; large download/install requires approval",
        "cardiff_cache_partial": True,
        "bertweet_cache_partial": True,
    }

    report = {
        "utc": utcnow(),
        "inventory": recon,
        "targeted_200_design": design,
        "training_exposure": exposure,
        "model": model_meta,
        "agreement_pilot": s_pilot,
        "agreement_targeted": s_targ,
        "scientific_status": (
            "TF-IDF vs Luna agreement is NOT accuracy, NOT FN rate, NOT verified GDS yield. "
            "No purpose labels asserted from relevance alone. Fixed 240 packet not rewritten."
        ),
        "nli_alternative": nli_blocker,
        "full_frame_new_model": "blocked_awaiting_approval",
        "paid_api": False,
    }
    write_json(dest / "PHASE2_SUMMARY.json", report)
    (dest / "FINDINGS.md").write_text(
        f"""# Phase 2 — excluded inventory & bounded screening

**UTC:** {report['utc']}  
Inventory match expected: **{recon['match_expected']}** (n={recon['n_inventory']}).  
Fixed 240 preserved (sha `{pilot_hash}`). Targeted diagnostic n={design['n_selected']}.

## Local method
`{model_meta['model_id']}` — train-only TF-IDF on frame relevance labels; pilot/protected/targeted blocked from training.

## Agreement (not accuracy)
- Pilot 240: compared={s_pilot['compared']} agree={s_pilot['agree']} disagree={s_pilot['disagree']}
- Targeted 200: compared={s_targ['compared']} agree={s_targ['agree']} disagree={s_targ['disagree']}

## Queues
See `REVIEW_PRIORITY_QUEUES.json` — relevance/disagreement priorities only; do not treat as TPS/GDS labels.

## Blocked
Full-frame reclassification; paid API; NLI/transformer without approved install.
""",
        encoding="utf-8",
    )
    (dest / "LIMITATIONS.md").write_text(
        """# Phase 2 limitations

- Frame training labels are mixed human/AI provenance.
- TF-IDF is lexical; not a contextual LLM.
- Targeted 200 is stratum-enriched, not a redesign of the 240 IPW sample.
- Agreement ≠ truth; no auto-merge into 6,425.
""",
        encoding="utf-8",
    )
    return report
