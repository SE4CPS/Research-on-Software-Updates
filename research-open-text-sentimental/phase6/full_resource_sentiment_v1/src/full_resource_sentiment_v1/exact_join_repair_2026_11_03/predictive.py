"""Leakage-safe TPS/GDS predictive comparisons on frozen dataset (no protected outcomes)."""

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

from full_resource_sentiment_v1.exact_join_repair_2026_11_03 import SEED
from full_resource_sentiment_v1.exact_join_repair_2026_11_03.common import (
    DATASET_JSONL,
    FROZEN_DEF,
    OUT_ROOT,
    RESTRICTED_ROOT,
    RQ4_V2,
    TB_SCORES,
    TEXT_MANIFEST,
    load_jsonl,
    protected_ids_only,
    utcnow,
    write_json,
    write_jsonl,
)
from full_resource_sentiment_v1.master_research_corrections_2026_11_03.near_dup import build_components


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


def _choose_thr(prob, yv) -> float:
    best_t, best_f = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        f = f1_score(yv, (prob >= t).astype(int), average="macro", zero_division=0)
        if f > best_f:
            best_f, best_t = float(f), float(t)
    return best_t


def run_predictive() -> dict[str, Any]:
    dest = OUT_ROOT / "phase5_predictive"
    dest.mkdir(parents=True, exist_ok=True)
    prot = protected_ids_only()

    thread_rows = {}
    tr_path = RESTRICTED_ROOT / "author_community_thread_rows.jsonl"
    if tr_path.exists():
        thread_rows = {str(r["reddit_id"]): r for r in load_jsonl(tr_path)}

    texts_map = {str(r["reddit_id"]): r for r in load_jsonl(TEXT_MANIFEST)}
    tb = {str(r["reddit_id"]): r for r in load_jsonl(TB_SCORES)} if TB_SCORES.exists() else {}

    rows = []
    for r in load_jsonl(DATASET_JSONL):
        rid = str(r["reddit_id"])
        if rid in prot:
            continue
        if r.get("final_relevance") != "relevant":
            continue
        if r.get("final_discussion") not in {"TPS", "GDS"}:
            continue
        if r.get("sentiment_definition") != FROZEN_DEF:
            continue
        doc = texts_map.get(rid) or {"title": "", "body": ""}
        title, body = doc.get("title") or "", doc.get("body") or ""
        sent = r.get("sentiment") or {}
        tbr = tb.get(rid) or {}
        thr = thread_rows.get(rid) or {}
        tb_combined = tbr.get("textblob_combined_polarity")
        if tb_combined is None:
            tb_combined = tbr.get("polarity")
        rows.append(
            {
                "reddit_id": rid,
                "y": 1 if r["final_discussion"] == "GDS" else 0,
                "text": f"{title}\n{body}".strip(),
                "title": title,
                "body": body,
                "vader_combined": sent.get("combined_compound"),
                "tb_combined": tb_combined,
                "first_author_minus_post": thr.get("first_author_minus_post_vader"),
                "n_author": sent.get("n_author_replies_nonempty"),
                "n_community": sent.get("n_community_replies_nonempty"),
            }
        )

    comp = build_components([r["reddit_id"] for r in rows], [r["title"] for r in rows], [r["body"] for r in rows])
    groups = np.asarray(comp["component_ids"])
    y = np.asarray([r["y"] for r in rows], dtype=int)
    texts = [r["text"] for r in rows]
    ids = [r["reddit_id"] for r in rows]

    # Reproduce RQ4 v2 fixed split (same seed/sizes as corrections stage3)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    train_idx, test_idx = next(gss.split(texts, y, groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED + 1)
    tr2, val_i = next(gss2.split([texts[i] for i in train_idx], y[train_idx], groups[train_idx]))
    train_core, val_core = train_idx[tr2], train_idx[val_i]
    test_sha = hashlib.sha256("\n".join(sorted(ids[i] for i in test_idx)).encode()).hexdigest()
    v2 = json.loads(Path(RQ4_V2).read_text(encoding="utf-8")) if Path(RQ4_V2).exists() else {}
    expected = (v2.get("split") or {}).get("test_ids_sha256")
    fixed_ok = test_sha == expected

    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3, max_features=50000, sublinear_tf=True)
    Xtr = vec.fit_transform([texts[i] for i in train_core])
    Xva = vec.transform([texts[i] for i in val_core])
    Xte = vec.transform([texts[i] for i in test_idx])

    def sent_mat(idx):
        mat = []
        for i in idx:
            vc = rows[i].get("vader_combined")
            tc = rows[i].get("tb_combined")
            fa = rows[i].get("first_author_minus_post")
            mat.append(
                [
                    0.0 if vc is None else float(vc),
                    0.0 if vc is None else 1.0,
                    0.0 if tc is None else float(tc),
                    0.0 if tc is None else 1.0,
                    0.0 if fa is None else float(fa),
                    0.0 if fa is None else 1.0,
                ]
            )
        return np.asarray(mat, dtype=float)

    models: dict[str, Any] = {}
    for name, cw, use_sent in (
        ("text_unweighted", None, False),
        ("text_balanced", "balanced", False),
        ("text_plus_sentiment_balanced", "balanced", True),
    ):
        Xt_tr, Xt_va, Xt_te = Xtr, Xva, Xte
        if use_sent:
            Xt_tr = hstack([Xtr, csr_matrix(sent_mat(train_core))])
            Xt_va = hstack([Xva, csr_matrix(sent_mat(val_core))])
            Xt_te = hstack([Xte, csr_matrix(sent_mat(test_idx))])
        clf = LogisticRegression(max_iter=2000, class_weight=cw, solver="liblinear", random_state=SEED)
        clf.fit(Xt_tr, y[train_core])
        thr = _choose_thr(clf.predict_proba(Xt_va)[:, 1], y[val_core])
        prob = clf.predict_proba(Xt_te)[:, 1]
        pred = (prob >= thr).astype(int)
        mode = int(np.bincount(y[train_core]).argmax())
        maj_pred = np.full(len(test_idx), mode)
        models[name] = {
            **_metrics(y[test_idx], pred, prob),
            "threshold": thr,
            "majority_baseline_macro_f1": float(f1_score(y[test_idx], maj_pred, average="macro", zero_division=0)),
            "majority_baseline_gds_recall": float(((maj_pred == 1) & (y[test_idx] == 1)).sum() / max((y[test_idx] == 1).sum(), 1)),
        }
        write_jsonl(
            RESTRICTED_ROOT / f"predictive_test_probs_{name}.jsonl",
            [
                {
                    "reddit_id": ids[i],
                    "y": int(y[i]),
                    "prob_gds": float(prob[j]),
                    "pred": int(pred[j]),
                    "model": name,
                }
                for j, i in enumerate(test_idx)
            ],
        )

    # Paired bootstrap balanced − unweighted
    clf_u = LogisticRegression(max_iter=2000, class_weight=None, solver="liblinear", random_state=SEED)
    clf_b = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=SEED)
    clf_u.fit(Xtr, y[train_core])
    clf_b.fit(Xtr, y[train_core])
    thr_u = _choose_thr(clf_u.predict_proba(Xva)[:, 1], y[val_core])
    thr_b = _choose_thr(clf_b.predict_proba(Xva)[:, 1], y[val_core])
    pred_u = (clf_u.predict_proba(Xte)[:, 1] >= thr_u).astype(int)
    pred_b = (clf_b.predict_proba(Xte)[:, 1] >= thr_b).astype(int)
    yt = y[test_idx]
    rng = np.random.default_rng(SEED + 99)
    diffs = []
    for _ in range(10000):
        idx = rng.integers(0, len(yt), size=len(yt))
        fu = f1_score(yt[idx], pred_u[idx], average="macro", zero_division=0)
        fb = f1_score(yt[idx], pred_b[idx], average="macro", zero_division=0)
        diffs.append(fb - fu)
    diffs_a = np.asarray(diffs)
    boot = {
        "balanced_minus_unweighted_macro_f1_mean": float(diffs_a.mean()),
        "ci95": [float(np.percentile(diffs_a, 2.5)), float(np.percentile(diffs_a, 97.5))],
        "n_bootstrap": 10000,
    }

    # 5 grouped stability splits
    stability = []
    gss5 = GroupShuffleSplit(n_splits=5, test_size=0.25, random_state=SEED + 5)
    for fold, (tr, te) in enumerate(gss5.split(texts, y, groups)):
        gss_v = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED + 5 + fold)
        trc, vac = next(gss_v.split([texts[i] for i in tr], y[tr], groups[tr]))
        tr_core, va_core = tr[trc], tr[vac]
        v = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3, max_features=50000, sublinear_tf=True)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=SEED)
        Xtr_f = v.fit_transform([texts[i] for i in tr_core])
        Xva_f = v.transform([texts[i] for i in va_core])
        Xte_f = v.transform([texts[i] for i in te])
        clf.fit(Xtr_f, y[tr_core])
        thr = _choose_thr(clf.predict_proba(Xva_f)[:, 1], y[va_core])
        prob = clf.predict_proba(Xte_f)[:, 1]
        pred = (prob >= thr).astype(int)
        m = _metrics(y[te], pred, prob)
        m["fold"] = fold
        stability.append(m)

    prior_u = ((v2.get("models") or {}).get("tfidf_text_only") or {}).get("macro_f1")
    report = {
        "utc": utcnow(),
        "protected_ids_excluded": len(prot),
        "n_rows": len(rows),
        "fixed_rq4_v2_split": fixed_ok,
        "test_ids_sha256": test_sha,
        "expected_test_ids_sha256": expected,
        "n_train": int(len(train_core)),
        "n_val": int(len(val_core)),
        "n_test": int(len(test_idx)),
        "models": models,
        "paired_bootstrap_balanced_minus_unweighted": boot,
        "stability_5_grouped_splits": stability,
        "rq4_v2_tfidf_text_only_macro_f1_reference": prior_u,
        "scientific_status": "Saved-label agreement on frozen TPS/GDS; not human-validated purpose truth",
        "dataset_version": "frozen full_dataset 2026-11-02; no recovered posts merged",
        "matches_rq4_v2": fixed_ok,
    }
    write_json(dest / "PREDICTIVE_COMPARISONS.json", report)
    (dest / "FINDINGS.md").write_text(
        f"""# Predictive findings (frozen dataset)

- Fixed RQ4 v2 split hash match: **{fixed_ok}**
- Text unweighted macro-F1: **{models['text_unweighted']['macro_f1']:.4f}**
- Text balanced macro-F1: **{models['text_balanced']['macro_f1']:.4f}**
- Text+sentiment balanced macro-F1: **{models['text_plus_sentiment_balanced']['macro_f1']:.4f}**
- GDS precision/recall (balanced): **{models['text_balanced']['gds_precision']:.3f} / {models['text_balanced']['gds_recall']:.3f}**
- Majority baseline macro-F1: **{models['text_balanced']['majority_baseline_macro_f1']:.4f}**
- Paired bootstrap balanced−unweighted macro-F1 CI95: **{boot['ci95']}**
- No protected outcomes; no recovered posts merged
- Metrics = saved-label agreement only
""",
        encoding="utf-8",
    )
    return report
