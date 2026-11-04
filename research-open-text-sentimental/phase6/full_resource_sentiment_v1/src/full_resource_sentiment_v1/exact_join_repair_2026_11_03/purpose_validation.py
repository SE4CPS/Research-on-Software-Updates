"""Purpose-model validation documentation + predeclared FN/FP reference plan."""

from __future__ import annotations

import csv
from collections import Counter
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit

from full_resource_sentiment_v1.exact_join_repair_2026_11_03 import SEED
from full_resource_sentiment_v1.exact_join_repair_2026_11_03.common import (
    DATASET_JSONL,
    ORIG_STRICT,
    ORIG_STRICT_RESTRICTED,
    OUT_ROOT,
    PILOT_IDS,
    TEXT_MANIFEST,
    load_jsonl,
    protected_ids_only,
    sha256_file,
    sha256_text,
    utcnow,
    write_json,
)


def run_purpose_validation() -> dict[str, Any]:
    dest = OUT_ROOT / "phase2_purpose_validation"
    dest.mkdir(parents=True, exist_ok=True)
    prot = protected_ids_only()

    with PILOT_IDS.open(encoding="utf-8", newline="") as handle:
        pilot_ids = {str(r["reddit_id"]) for r in csv.DictReader(handle)}
    targeted_path = ORIG_STRICT / "phase2_excluded_screening/targeted_200_public.jsonl"
    targeted = load_jsonl(targeted_path) if targeted_path.exists() else []
    targeted_ids = {str(r["reddit_id"]) for r in targeted}
    blocked = set(prot) | pilot_ids | targeted_ids

    frame_text = {str(r["reddit_id"]): r for r in load_jsonl(TEXT_MANIFEST)}
    source_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    rows = []
    for r in load_jsonl(DATASET_JSONL):
        rid = str(r["reddit_id"])
        if rid in blocked:
            continue
        if r.get("final_relevance") != "relevant":
            continue
        if r.get("final_discussion") not in {"TPS", "GDS", "Other"}:
            continue
        doc = frame_text.get(rid) or {"title": "", "body": ""}
        text = f"{doc.get('title') or ''}\n{doc.get('body') or ''}".strip()
        if not text:
            continue
        lab = r["final_discussion"]
        class_counts[lab] += 1
        source_counts[str(r.get("decision_source") or "unknown")] += 1
        rows.append({"reddit_id": rid, "label": lab, "text": text, "source": r.get("decision_source")})

    label_to_i = {"TPS": 0, "GDS": 1, "Other": 2}
    X = [r["text"] for r in rows]
    y = np.asarray([label_to_i[r["label"]] for r in rows], dtype=int)
    groups = [sha256_text(r["text"][:60])[:12] for r in rows]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED + 11)
    tr, te = next(gss.split(X, y, groups))
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=30000, sublinear_tf=True)
    Xtr = vec.fit_transform([X[i] for i in tr])
    Xte = vec.transform([X[i] for i in te])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=SEED)
    clf.fit(Xtr, y[tr])
    pred = clf.predict(Xte)
    p, rcall, f, _ = precision_recall_fscore_support(y[te], pred, labels=[0, 1, 2], zero_division=0)
    # GDS = class 1
    gds_precision = float(p[1])
    gds_recall = float(rcall[1])

    # Fixed samples preserved
    pilot_sha = sha256_file(PILOT_IDS)
    prior_cand = ORIG_STRICT_RESTRICTED / "tfidf_predictions_pilot240.jsonl"

    plan = {
        "name": "minimal_reference_evaluation_v1",
        "goal": "Estimate FP among AI-relevant/TPS-GDS candidates and FN among AI-not-relevant without large review",
        "samples": [
            {
                "name": "FP_queue",
                "n": 40,
                "draw_from": "AI stage-A relevant ∩ purpose∈{TPS,GDS} among excluded screened",
                "stratify": "purpose × author_depth_bucket",
                "seed": SEED + 101,
                "outcome": "human relevant? human purpose?",
            },
            {
                "name": "FN_queue",
                "n": 40,
                "draw_from": "AI stage-A not_relevant among excluded; oversample author>=2 + update-cue titles",
                "seed": SEED + 102,
                "outcome": "human relevant? if yes purpose",
            },
            {
                "name": "fixed_probability_240",
                "n": 240,
                "status": "IDs frozen; labels incomplete — retain for selection-bias estimation when labeled",
                "sha256": pilot_sha,
            },
            {
                "name": "targeted_200",
                "n": 200,
                "status": "retained; not independent human accuracy alone",
            },
        ],
        "not_allowed": [
            "Treat 18 prior candidate GDS as confirmed",
            "Use protected outcomes for model selection",
            "Merge candidates into frozen frame without audit approval",
        ],
        "status": "PREDECLARED — not executed as human review in this run",
    }

    report = {
        "utc": utcnow(),
        "scientific_status": (
            "Held-out metrics are saved-label agreement on mixed-source AI/heuristic labels — "
            "NOT independent human accuracy"
        ),
        "split_isolation": {
            "blocked_from_train_and_heldout_construction": [
                "protected_693_id_only",
                "fixed_pilot_240",
                "targeted_200",
            ],
            "protected_n": len(prot),
            "pilot_n": len(pilot_ids),
            "targeted_n": len(targeted_ids),
            "pilot_ids_sha256": pilot_sha,
        },
        "class_counts_train_universe": dict(class_counts),
        "source_label_composition": dict(source_counts),
        "heldout_saved_label_metrics": {
            "n_train": int(len(tr)),
            "n_heldout": int(len(te)),
            "macro_f1": float(np.mean(f)),
            "per_class": {
                "TPS": {"precision": float(p[0]), "recall": float(rcall[0]), "f1": float(f[0])},
                "GDS": {"precision": gds_precision, "recall": gds_recall, "f1": float(f[1])},
                "Other": {"precision": float(p[2]), "recall": float(rcall[2]), "f1": float(f[2])},
            },
            "gds_precision": gds_precision,
            "gds_recall": gds_recall,
        },
        "prior_18_gds_candidates": {
            "status": "AI_ONLY_CANDIDATES — not confirmed",
            "do_not_cite_as_yield": True,
        },
        "reference_evaluation_plan": plan,
        "full_frame_30998": "NOT_RUN in this package (Codex gate); see priority screening separately",
    }
    write_json(dest / "PURPOSE_MODEL_VALIDATION.json", report)
    write_json(dest / "REFERENCE_EVALUATION_PLAN.json", plan)
    return report
