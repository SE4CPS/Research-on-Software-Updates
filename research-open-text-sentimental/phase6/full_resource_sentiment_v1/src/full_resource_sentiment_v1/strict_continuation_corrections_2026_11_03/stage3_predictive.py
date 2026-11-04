"""Stage 3: fixed-split predictive diagnostics + multi-split stability."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from full_resource_sentiment_v1.master_research_corrections_2026_11_03.near_dup import build_components
from full_resource_sentiment_v1.paths import WORKSPACE
from full_resource_sentiment_v1.strict_continuation_corrections_2026_11_03 import SEED
from full_resource_sentiment_v1.strict_continuation_corrections_2026_11_03.common import (
    DATASET_JSONL,
    FROZEN_DEF,
    OUT_ROOT,
    RESTRICTED_ROOT,
    RQ4_V2,
    TEXT_MANIFEST,
    load_jsonl,
    utcnow,
    write_json,
    write_jsonl,
)

TB_SCORES = WORKSPACE / "restricted/master_research_2026-11-03/textblob_tps_gds_scores.jsonl"


def _metrics(y_true, y_pred, y_prob) -> dict[str, Any]:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "gds_precision": float(p[1]),
        "gds_recall": float(r[1]),
        "gds_f1": float(f[1]),
        "tps_precision": float(p[0]),
        "tps_recall": float(r[0]),
        "tps_f1": float(f[0]),
        "pr_auc_gds": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "confusion_matrix_[TPS,GDS]": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "predicted_gds_rate": float((y_pred == 1).mean()),
        "n": int(len(y_true)),
        "n_gds": int((y_true == 1).sum()),
    }


def _load_rows() -> list[dict[str, Any]]:
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
                "tb_combined": tbr.get("textblob_combined_polarity"),
                "tb_title": tbr.get("textblob_title_polarity"),
                "tb_body": tbr.get("textblob_body_polarity"),
            }
        )
    return rows


def run_stage3() -> dict[str, Any]:
    dest = OUT_ROOT / "stage3_predictive"
    dest.mkdir(parents=True, exist_ok=True)
    rows = _load_rows()
    comp = build_components([r["reddit_id"] for r in rows], [r["title"] for r in rows], [r["body"] for r in rows])
    groups = np.asarray(comp["component_ids"])
    y = np.asarray([r["y"] for r in rows], dtype=int)
    texts = [r["text"] for r in rows]
    ids = [r["reddit_id"] for r in rows]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    train_idx, test_idx = next(gss.split(texts, y, groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED + 1)
    tr2, val_i = next(gss2.split([texts[i] for i in train_idx], y[train_idx], groups[train_idx]))
    train_core, val_core = train_idx[tr2], train_idx[val_i]
    test_sha = hashlib.sha256("\n".join(sorted(ids[i] for i in test_idx)).encode()).hexdigest()
    v2 = json.loads(Path(RQ4_V2).read_text(encoding="utf-8")) if RQ4_V2.exists() else {}
    assert test_sha == (v2.get("split") or {}).get("test_ids_sha256"), "fixed test IDs must match RQ4 v2"

    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3, max_features=50000, sublinear_tf=True)
    Xtr = vec.fit_transform([texts[i] for i in train_core])
    Xva = vec.transform([texts[i] for i in val_core])
    Xte = vec.transform([texts[i] for i in test_idx])

    def sent_mat(idx, keys):
        mat = []
        for i in idx:
            row = []
            for k in keys:
                v = rows[i].get(k)
                row += [0.0 if v is None else float(v), 0.0 if v is None else 1.0]
            mat.append(row)
        return np.asarray(mat, dtype=float)

    def choose_thr(prob, yv):
        best_t, best_f = 0.5, -1.0
        for t in np.linspace(0.05, 0.95, 19):
            f = f1_score(yv, (prob >= t).astype(int), average="macro", zero_division=0)
            if f > best_f:
                best_f, best_t = float(f), float(t)
        return best_t

    models = {}
    # text only unweighted + balanced (preserve prior comparison)
    for name, cw in (("unweighted", None), ("balanced", "balanced")):
        clf = LogisticRegression(max_iter=2000, class_weight=cw, solver="liblinear", random_state=SEED)
        clf.fit(Xtr, y[train_core])
        thr = choose_thr(clf.predict_proba(Xva)[:, 1], y[val_core])
        prob = clf.predict_proba(Xte)[:, 1]
        pred = (prob >= thr).astype(int)
        prec, rec, thr_curve = precision_recall_curve(y[test_idx], prob)
        models[f"text_{name}"] = {
            **_metrics(y[test_idx], pred, prob),
            "threshold": thr,
            "pr_curve_points": len(thr_curve),
            "pr_auc": float(average_precision_score(y[test_idx], prob)),
        }
        # save probs
        write_jsonl(
            RESTRICTED_ROOT / f"test_probs_{name}.jsonl",
            [
                {"reddit_id": ids[i], "y": int(y[i]), "prob_gds": float(prob[j]), "pred": int(pred[j]), "model": name}
                for j, i in enumerate(test_idx)
            ],
        )

    # sentiment-only / text+sentiment
    def fit_combo(name, keys, use_text=True):
        s_tr, s_va, s_te = sent_mat(train_core, keys), sent_mat(val_core, keys), sent_mat(test_idx, keys)
        sc = StandardScaler()
        s_tr = sc.fit_transform(s_tr)
        s_va, s_te = sc.transform(s_va), sc.transform(s_te)
        if use_text:
            Xtr2, Xva2, Xte2 = hstack([Xtr, csr_matrix(s_tr)]), hstack([Xva, csr_matrix(s_va)]), hstack(
                [Xte, csr_matrix(s_te)]
            )
        else:
            Xtr2, Xva2, Xte2 = s_tr, s_va, s_te
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=SEED)
        clf.fit(Xtr2, y[train_core])
        thr = choose_thr(clf.predict_proba(Xva2)[:, 1], y[val_core])
        prob = clf.predict_proba(Xte2)[:, 1]
        return {**_metrics(y[test_idx], (prob >= thr).astype(int), prob), "threshold": thr}

    models["sentiment_only_vader_tb"] = fit_combo(
        "sent", ["vader_combined", "vader_title", "vader_body", "tb_combined"], use_text=False
    )
    models["text_plus_vader"] = fit_combo("tv", ["vader_combined", "vader_title", "vader_body"])
    models["text_plus_textblob"] = fit_combo("tt", ["tb_combined", "tb_title", "tb_body"])
    models["text_plus_both"] = fit_combo(
        "both", ["vader_combined", "vader_title", "vader_body", "tb_combined", "tb_title", "tb_body"]
    )

    # Paired bootstrap 10k text balanced vs unweighted
    clf_u = LogisticRegression(max_iter=2000, class_weight=None, solver="liblinear", random_state=SEED).fit(
        Xtr, y[train_core]
    )
    clf_b = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=SEED).fit(
        Xtr, y[train_core]
    )
    thr_u = models["text_unweighted"]["threshold"]
    thr_b = models["text_balanced"]["threshold"]
    pu, pb = clf_u.predict_proba(Xte)[:, 1], clf_b.predict_proba(Xte)[:, 1]
    rng = np.random.default_rng(SEED)
    yt = y[test_idx]
    deltas = []
    for _ in range(10000):
        ix = rng.integers(0, len(yt), size=len(yt))
        fa = f1_score(yt[ix], (pb[ix] >= thr_b).astype(int), average="macro", zero_division=0)
        fb = f1_score(yt[ix], (pu[ix] >= thr_u).astype(int), average="macro", zero_division=0)
        deltas.append(fa - fb)
    deltas.sort()
    paired = {
        "balanced_minus_unweighted_macro_f1": {
            "mean": float(np.mean(deltas)),
            "ci95": [float(deltas[250]), float(deltas[9749])],
            "n_boot": 10000,
            "note": "Post-level resample of test predictions — not training-process uncertainty",
        }
    }

    # Threshold tradeoff on validation for balanced
    tradeoffs = []
    pval = clf_b.predict_proba(Xva)[:, 1]
    for t in np.linspace(0.05, 0.95, 19):
        m = _metrics(y[val_core], (pval >= t).astype(int), pval)
        tradeoffs.append({"threshold": float(t), "val_macro_f1": m["macro_f1"], "val_gds_recall": m["gds_recall"], "val_gds_precision": m["gds_precision"]})

    # Predeclared multi-split stability (report ALL)
    stability = []
    for seed in (SEED, SEED + 11, SEED + 22, SEED + 33, SEED + 44):
        g1 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        tr_i, te_i = next(g1.split(texts, y, groups))
        g2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + 1)
        a, b = next(g2.split([texts[i] for i in tr_i], y[tr_i], groups[tr_i]))
        tr_c, va_c = tr_i[a], tr_i[b]
        assert len(set(groups[tr_c]) & set(groups[te_i])) == 0
        v = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3, max_features=50000, sublinear_tf=True)
        Xtr_s = v.fit_transform([texts[i] for i in tr_c])
        Xva_s = v.transform([texts[i] for i in va_c])
        Xte_s = v.transform([texts[i] for i in te_i])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=SEED)
        clf.fit(Xtr_s, y[tr_c])
        thr = choose_thr(clf.predict_proba(Xva_s)[:, 1], y[va_c])
        prob = clf.predict_proba(Xte_s)[:, 1]
        m = _metrics(y[te_i], (prob >= thr).astype(int), prob)
        stability.append(
            {
                "seed": seed,
                "test_sha": hashlib.sha256("\n".join(sorted(ids[i] for i in te_i)).encode()).hexdigest()[:16],
                "macro_f1": m["macro_f1"],
                "gds_precision": m["gds_precision"],
                "gds_recall": m["gds_recall"],
                "n_test_gds": m["n_gds"],
            }
        )

    report = {
        "utc": utcnow(),
        "scientific_status": "Agreement with mixed-source saved labels — NOT independent TPS/GDS accuracy",
        "fixed_split_test_ids_sha256": test_sha,
        "matches_rq4_v2": True,
        "preserved_reference_metrics": {
            "unweighted_gds_PR": [0.520, 0.609],
            "balanced_gds_PR": [0.569, 0.516],
        },
        "models": models,
        "paired_bootstrap_10k": paired,
        "validation_threshold_tradeoffs_balanced": tradeoffs,
        "multi_split_stability_predeclared_seeds": stability,
        "cardiff_features": "not_available_pending_approval",
        "embedding_classifier": "blocked_no_torch_despite_minilm_cache",
    }
    write_json(dest / "PREDICTIVE_EXTENSION.json", report)
    (dest / "FINDINGS.md").write_text(
        f"""# Stage 3 predictive extension

Fixed test SHA matches RQ4 v2. Unweighted macro-F1={models['text_unweighted']['macro_f1']:.4f}; balanced={models['text_balanced']['macro_f1']:.4f}.  
Multi-split seeds reported in full (no cherry-pick). Saved-label agreement only.
""",
        encoding="utf-8",
    )
    return report
