"""Phase 3: predictive class-balance experiment on fixed RQ4 v2 test IDs."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

from full_resource_sentiment_v1.master_research_corrections_2026_11_03.near_dup import build_components
from full_resource_sentiment_v1.strict_continuation_2026_11_03 import SEED
from full_resource_sentiment_v1.strict_continuation_2026_11_03.common import (
    DATASET_JSONL,
    FROZEN_DEF,
    OUT_ROOT,
    RQ4_V2,
    TB_SCORES,
    TEXT_MANIFEST,
    load_jsonl,
    utcnow,
    write_json,
)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    return {
        "n": int(len(y_true)),
        "n_gds": int((y_true == 1).sum()),
        "predicted_gds_rate": float((y_pred == 1).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "tps_precision": float(p[0]),
        "tps_recall": float(r[0]),
        "tps_f1": float(f[0]),
        "gds_precision": float(p[1]),
        "gds_recall": float(r[1]),
        "gds_f1": float(f[1]),
        "confusion_matrix_labels_[TPS,GDS]": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def run_phase3() -> dict[str, Any]:
    dest = OUT_ROOT / "phase3_class_balance"
    dest.mkdir(parents=True, exist_ok=True)

    texts = {str(r["reddit_id"]): r for r in load_jsonl(TEXT_MANIFEST)}
    tb = {str(r["reddit_id"]): r for r in load_jsonl(TB_SCORES)} if TB_SCORES.exists() else {}
    rows = []
    for r in load_jsonl(DATASET_JSONL):
        if r.get("final_relevance") != "relevant":
            continue
        if r.get("final_discussion") not in {"TPS", "GDS"}:
            continue
        if r.get("sentiment_definition") != FROZEN_DEF:
            continue
        rid = str(r["reddit_id"])
        doc = texts.get(rid) or {"title": "", "body": ""}
        title, body = doc.get("title") or "", doc.get("body") or ""
        sent = r.get("sentiment") or {}
        tbr = tb.get(rid) or {}
        rows.append(
            {
                "reddit_id": rid,
                "y": 1 if r["final_discussion"] == "GDS" else 0,
                "text": f"{title}\n{body}".strip(),
                "title": title,
                "body": body,
                "vader_combined": sent.get("combined_compound"),
                "vader_title": sent.get("title_compound"),
                "vader_body": sent.get("body_compound"),
            }
        )

    # Rebuild same connected-component groups and reproduce split with same seed
    comp = build_components(
        [r["reddit_id"] for r in rows],
        [r["title"] for r in rows],
        [r["body"] for r in rows],
    )
    groups = np.asarray(comp["component_ids"])
    y = np.asarray([r["y"] for r in rows], dtype=int)
    texts_list = [r["text"] for r in rows]
    ids = [r["reddit_id"] for r in rows]

    from sklearn.model_selection import GroupShuffleSplit

    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    train_idx, test_idx = next(gss.split(texts_list, y, groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED + 1)
    tr2, val_idx = next(gss2.split([texts_list[i] for i in train_idx], y[train_idx], groups[train_idx]))
    train_core = train_idx[tr2]
    val_core = train_idx[val_idx]

    class_counts = {
        "train": {"TPS": int((y[train_core] == 0).sum()), "GDS": int((y[train_core] == 1).sum())},
        "val": {"TPS": int((y[val_core] == 0).sum()), "GDS": int((y[val_core] == 1).sum())},
        "test": {"TPS": int((y[test_idx] == 0).sum()), "GDS": int((y[test_idx] == 1).sum())},
    }
    n_tr = len(train_core)
    n_gds_tr = class_counts["train"]["GDS"]
    n_tps_tr = class_counts["train"]["TPS"]
    # sklearn balanced weight = n_samples / (n_classes * n_c)
    w_tps = n_tr / (2.0 * n_tps_tr)
    w_gds = n_tr / (2.0 * n_gds_tr)

    test_ids_sha = hashlib.sha256("\n".join(sorted(ids[i] for i in test_idx)).encode()).hexdigest()
    v2 = json.loads(Path(RQ4_V2).read_text(encoding="utf-8")) if RQ4_V2.exists() else {}
    expected_test_sha = (v2.get("split") or {}).get("test_ids_sha256")

    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3, max_features=50000, sublinear_tf=True)
    Xtr = vec.fit_transform([texts_list[i] for i in train_core])
    Xva = vec.transform([texts_list[i] for i in val_core])
    Xte = vec.transform([texts_list[i] for i in test_idx])

    def choose_thr(prob_val: np.ndarray, y_val: np.ndarray) -> tuple[float, float]:
        best_t, best_f = 0.5, -1.0
        for t in np.linspace(0.05, 0.95, 19):
            f = f1_score(y_val, (prob_val >= t).astype(int), average="macro", zero_division=0)
            if f > best_f:
                best_f, best_t = float(f), float(t)
        return best_t, best_f

    results = {}

    # 1) Unweighted logistic + val threshold
    clf_u = LogisticRegression(max_iter=2000, class_weight=None, solver="liblinear", random_state=SEED)
    clf_u.fit(Xtr, y[train_core])
    thr_u, _ = choose_thr(clf_u.predict_proba(Xva)[:, 1], y[val_core])
    pu = clf_u.predict_proba(Xte)[:, 1]
    results["unweighted_val_threshold"] = {
        **_metrics(y[test_idx], (pu >= thr_u).astype(int)),
        "threshold": thr_u,
        "class_weight": None,
    }

    # 2) Balanced logistic + val threshold (matches prior design)
    clf_b = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=SEED)
    clf_b.fit(Xtr, y[train_core])
    thr_b, _ = choose_thr(clf_b.predict_proba(Xva)[:, 1], y[val_core])
    pb = clf_b.predict_proba(Xte)[:, 1]
    results["balanced_val_threshold"] = {
        **_metrics(y[test_idx], (pb >= thr_b).astype(int)),
        "threshold": thr_b,
        "class_weight": "balanced",
        "effective_train_weights": {"TPS": w_tps, "GDS": w_gds},
    }

    # 3) Balanced + default 0.5 threshold (no val tuning)
    results["balanced_threshold_0_5"] = {
        **_metrics(y[test_idx], (pb >= 0.5).astype(int)),
        "threshold": 0.5,
        "class_weight": "balanced",
    }

    # 4) Training-only random oversample of minority to 1:1 then unweighted + val thr
    rng = np.random.default_rng(SEED)
    ytr = y[train_core]
    pos_idx = np.where(ytr == 1)[0]
    neg_idx = np.where(ytr == 0)[0]
    pos_up = rng.choice(pos_idx, size=len(neg_idx), replace=True)
    mix = np.concatenate([neg_idx, pos_up])
    rng.shuffle(mix)
    Xtr_rs = Xtr[mix]
    ytr_rs = ytr[mix]
    clf_rs = LogisticRegression(max_iter=2000, class_weight=None, solver="liblinear", random_state=SEED)
    clf_rs.fit(Xtr_rs, ytr_rs)
    # val not resampled
    thr_rs, _ = choose_thr(clf_rs.predict_proba(Xva)[:, 1], y[val_core])
    prs = clf_rs.predict_proba(Xte)[:, 1]
    results["train_oversample_1to1_unweighted_val_threshold"] = {
        **_metrics(y[test_idx], (prs >= thr_rs).astype(int)),
        "threshold": thr_rs,
        "class_weight": None,
        "resampling": "random_oversample_GDS_to_TPS_count_training_only",
        "train_rows_after_resample": int(len(mix)),
    }

    # Paired bootstrap balanced vs unweighted macro-F1
    def paired(prob_a, thr_a, prob_b, thr_b):
        rngb = np.random.default_rng(SEED)
        yt = y[test_idx]
        deltas = []
        for _ in range(2000):
            ix = rngb.integers(0, len(yt), size=len(yt))
            fa = f1_score(yt[ix], (prob_a[ix] >= thr_a).astype(int), average="macro", zero_division=0)
            fb = f1_score(yt[ix], (prob_b[ix] >= thr_b).astype(int), average="macro", zero_division=0)
            deltas.append(fa - fb)
        deltas.sort()
        return {"mean_delta_macro_f1": float(np.mean(deltas)), "ci95": [float(deltas[50]), float(deltas[1949])]}

    paired_boot = {
        "balanced_minus_unweighted": paired(pb, thr_b, pu, thr_u),
        "oversample_minus_unweighted": paired(prs, thr_rs, pu, thr_u),
    }

    report = {
        "utc": utcnow(),
        "scientific_status": (
            "Agreement with saved mixed-provenance labels — NOT independent purpose truth. "
            "Descriptive sentiment population was not rebalanced."
        ),
        "class_counts": class_counts,
        "balanced_weights_formula": {
            "TPS": w_tps,
            "GDS": w_gds,
            "expected_approx": {"TPS": 0.5363, "GDS": 7.3788},
        },
        "test_ids_sha256": test_ids_sha,
        "test_ids_match_rq4_v2": test_ids_sha == expected_test_sha,
        "rq4_v2_expected_test_sha": expected_test_sha,
        "no_resampling_in_baseline_design": True,
        "models": results,
        "paired_bootstrap": paired_boot,
        "test_gds_prevalence": float(y[test_idx].mean()),
    }
    write_json(dest / "CLASS_BALANCE_RESULTS.json", report)

    with (dest / "model_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "macro_f1",
                "gds_precision",
                "gds_recall",
                "gds_f1",
                "predicted_gds_rate",
                "threshold",
            ],
        )
        w.writeheader()
        for name, m in results.items():
            w.writerow(
                {
                    "model": name,
                    "macro_f1": m["macro_f1"],
                    "gds_precision": m["gds_precision"],
                    "gds_recall": m["gds_recall"],
                    "gds_f1": m["gds_f1"],
                    "predicted_gds_rate": m["predicted_gds_rate"],
                    "threshold": m.get("threshold"),
                }
            )

    (dest / "FINDINGS.md").write_text(
        f"""# Phase 3 — class-balance experiment

**UTC:** {report['utc']}  
Test IDs match RQ4 v2: **{report['test_ids_match_rq4_v2']}**  
Train counts TPS/GDS: {class_counts['train']}; balanced weights ≈ TPS {w_tps:.4f} / GDS {w_gds:.4f}

See `model_comparison.csv`. {report['scientific_status']}
""",
        encoding="utf-8",
    )
    return report
