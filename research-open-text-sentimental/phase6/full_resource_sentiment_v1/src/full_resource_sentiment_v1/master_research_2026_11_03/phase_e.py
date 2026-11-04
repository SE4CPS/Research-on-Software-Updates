"""Phase E: leakage-aware text vs text+sentiment TPS/GDS prediction (exploratory)."""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix, hstack

from full_resource_sentiment_v1.master_research_2026_11_03 import SEED
from full_resource_sentiment_v1.master_research_2026_11_03.common import (
    DATASET_JSONL,
    FROZEN_DEF,
    OUT_ROOT,
    RESTRICTED_ROOT,
    load_jsonl,
    sha256_text,
    utcnow,
    write_json,
)


def _near_dup_group(title: str, body: str) -> str:
    # Group near-duplicates by normalized prefix hash of title+body
    norm = " ".join((title or "").lower().split()) + "\n" + " ".join((body or "").lower().split())[:500]
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    # Positive class = GDS (1)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return {
        "n": int(len(y_true)),
        "n_gds": int((y_true == 1).sum()),
        "n_tps": int((y_true == 0).sum()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "tps_precision": float(p[0]),
        "tps_recall": float(r[0]),
        "tps_f1": float(f[0]),
        "gds_precision": float(p[1]),
        "gds_recall": float(r[1]),
        "gds_f1": float(f[1]),
        "pr_auc_gds": float(average_precision_score(y_true, y_prob)) if y_prob is not None else None,
        "confusion_matrix_labels_[TPS,GDS]": cm,
    }


def _fit_predict(
    X_train,
    y_train,
    X_test,
    *,
    class_weight: str | dict = "balanced",
) -> tuple[np.ndarray, np.ndarray]:
    clf = LogisticRegression(
        max_iter=2000,
        class_weight=class_weight,
        solver="liblinear",
        random_state=SEED,
    )
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    if hasattr(clf, "predict_proba"):
        prob = clf.predict_proba(X_test)[:, 1]
    else:
        prob = clf.decision_function(X_test)
    return pred, prob


def run_phase_e() -> dict[str, Any]:
    dest = OUT_ROOT / "phase_e_predictive_rq4"
    dest.mkdir(parents=True, exist_ok=True)
    texts = {
        str(r["reddit_id"]): r for r in load_jsonl(RESTRICTED_ROOT / "text_thread_manifest_restricted.jsonl")
    }
    tb_scores = {}
    tb_path = RESTRICTED_ROOT / "textblob_tps_gds_scores.jsonl"
    if tb_path.exists():
        for r in load_jsonl(tb_path):
            tb_scores[str(r["reddit_id"])] = r

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
        # Leakage guard: never include decision_source, confidence, reasons, labels in text features
        text = f"{title}\n{body}".strip()
        sent = r.get("sentiment") or {}
        tb = tb_scores.get(rid) or {}
        rows.append(
            {
                "reddit_id": rid,
                "y": 1 if r["final_discussion"] == "GDS" else 0,
                "text": text,
                "group": _near_dup_group(title, body),
                "decision_source": r.get("decision_source"),
                "selection_stratum": r.get("selection_stratum"),
                "vader_combined": sent.get("combined_compound"),
                "vader_title": sent.get("title_compound"),
                "vader_body": sent.get("body_compound"),
                "tb_combined": tb.get("textblob_combined_polarity"),
                "tb_title": tb.get("textblob_title_polarity"),
                "tb_body": tb.get("textblob_body_polarity"),
                "title_sha256": sha256_text(title),
                "body_sha256": sha256_text(body),
            }
        )

    y = np.asarray([r["y"] for r in rows], dtype=int)
    groups = np.asarray([r["group"] for r in rows])
    texts_list = [r["text"] for r in rows]
    ids = [r["reddit_id"] for r in rows]

    # Freeze grouped split once
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    train_idx, test_idx = next(gss.split(texts_list, y, groups))
    # Further split train into train/val for threshold selection (inside training only)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED + 1)
    tr2, val_idx = next(gss2.split([texts_list[i] for i in train_idx], y[train_idx], groups[train_idx]))
    train_core = train_idx[tr2]
    val_core = train_idx[val_idx]

    split_info = {
        "seed": SEED,
        "n_total": len(rows),
        "n_train": int(len(train_core)),
        "n_val": int(len(val_core)),
        "n_test": int(len(test_idx)),
        "test_gds": int(y[test_idx].sum()),
        "train_gds": int(y[train_core].sum()),
        "group_leakage_check": {
            "train_test_group_overlap": int(len(set(groups[train_core]) & set(groups[test_idx]))),
            "train_val_group_overlap": int(len(set(groups[train_core]) & set(groups[val_core]))),
        },
        "test_ids_sha256": hashlib.sha256("\n".join(sorted(ids[i] for i in test_idx)).encode()).hexdigest(),
    }
    assert split_info["group_leakage_check"]["train_test_group_overlap"] == 0

    # Vectorizer fit on train_core only
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=3,
        max_features=50000,
        sublinear_tf=True,
    )
    X_train_text = vectorizer.fit_transform([texts_list[i] for i in train_core])
    X_val_text = vectorizer.transform([texts_list[i] for i in val_core])
    X_test_text = vectorizer.transform([texts_list[i] for i in test_idx])

    def sentiment_matrix(idx: np.ndarray, keys: list[str]) -> np.ndarray:
        mat = []
        for i in idx:
            row = []
            for k in keys:
                v = rows[i].get(k)
                row.append(0.0 if v is None else float(v))
                row.append(0.0 if v is None else 1.0)  # observed indicator
            mat.append(row)
        return np.asarray(mat, dtype=float)

    def scale_fit_transform(train_arr: np.ndarray, *others: np.ndarray):
        scaler = StandardScaler()
        tr = scaler.fit_transform(train_arr)
        return (tr,) + tuple(scaler.transform(a) for a in others)

    results: dict[str, Any] = {}

    # Majority baseline on test (predict train majority)
    maj = Counter(y[train_core]).most_common(1)[0][0]
    maj_pred = np.full_like(y[test_idx], maj)
    results["majority"] = _metrics(y[test_idx], maj_pred, np.full(len(test_idx), float(maj)))

    # Stratified dummy: predict with train base rate randomly (seeded) — report separately
    rng = np.random.default_rng(SEED)
    p_gds = float(y[train_core].mean())
    strat_pred = (rng.random(len(test_idx)) < p_gds).astype(int)
    results["stratified_dummy"] = _metrics(y[test_idx], strat_pred, np.full(len(test_idx), p_gds))

    # Text-only TF-IDF
    # Choose threshold on val by max macro-F1
    pred_val, prob_val = _fit_predict(X_train_text, y[train_core], X_val_text)
    # Refit on train_core and apply default 0.5; also search threshold on val
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=SEED)
    clf.fit(X_train_text, y[train_core])
    prob_val = clf.predict_proba(X_val_text)[:, 1]
    best_t, best_f = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        f = f1_score(y[val_core], (prob_val >= t).astype(int), average="macro", zero_division=0)
        if f > best_f:
            best_f, best_t = float(f), float(t)
    prob_test = clf.predict_proba(X_test_text)[:, 1]
    pred_test = (prob_test >= best_t).astype(int)
    results["tfidf_text_only"] = {
        **_metrics(y[test_idx], pred_test, prob_test),
        "threshold_selected_on_val": best_t,
        "val_macro_f1_at_threshold": best_f,
    }

    def text_plus(name: str, keys: list[str]) -> dict[str, Any]:
        s_tr = sentiment_matrix(train_core, keys)
        s_va = sentiment_matrix(val_core, keys)
        s_te = sentiment_matrix(test_idx, keys)
        s_tr, s_va, s_te = scale_fit_transform(s_tr, s_va, s_te)
        Xtr = hstack([X_train_text, csr_matrix(s_tr)])
        Xva = hstack([X_val_text, csr_matrix(s_va)])
        Xte = hstack([X_test_text, csr_matrix(s_te)])
        clf2 = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=SEED)
        clf2.fit(Xtr, y[train_core])
        pv = clf2.predict_proba(Xva)[:, 1]
        bt, bf = 0.5, -1.0
        for t in np.linspace(0.05, 0.95, 19):
            f = f1_score(y[val_core], (pv >= t).astype(int), average="macro", zero_division=0)
            if f > bf:
                bf, bt = float(f), float(t)
        pt = clf2.predict_proba(Xte)[:, 1]
        return {
            **_metrics(y[test_idx], (pt >= bt).astype(int), pt),
            "threshold_selected_on_val": bt,
            "val_macro_f1_at_threshold": bf,
            "sentiment_keys": keys,
        }

    results["text_plus_vader"] = text_plus(
        "text_plus_vader", ["vader_combined", "vader_title", "vader_body"]
    )
    results["text_plus_textblob"] = text_plus(
        "text_plus_textblob", ["tb_combined", "tb_title", "tb_body"]
    )
    results["text_plus_vader_textblob"] = text_plus(
        "text_plus_both",
        ["vader_combined", "vader_title", "vader_body", "tb_combined", "tb_title", "tb_body"],
    )

    # Sentiment-only
    s_tr = sentiment_matrix(train_core, ["vader_combined", "vader_title", "vader_body", "tb_combined"])
    s_te = sentiment_matrix(test_idx, ["vader_combined", "vader_title", "vader_body", "tb_combined"])
    s_tr, s_te = scale_fit_transform(s_tr, s_te)
    pred_s, prob_s = _fit_predict(s_tr, y[train_core], s_te)
    results["sentiment_only_vader_textblob"] = _metrics(y[test_idx], pred_s, prob_s)

    # Paired bootstrap delta macro-F1: text+vader vs text-only
    def paired_delta(prob_a: np.ndarray, prob_b: np.ndarray, thr_a: float, thr_b: float) -> dict[str, Any]:
        rng = np.random.default_rng(SEED)
        y_t = y[test_idx]
        deltas = []
        for _ in range(2000):
            idx = rng.integers(0, len(y_t), size=len(y_t))
            fa = f1_score(y_t[idx], (prob_a[idx] >= thr_a).astype(int), average="macro", zero_division=0)
            fb = f1_score(y_t[idx], (prob_b[idx] >= thr_b).astype(int), average="macro", zero_division=0)
            deltas.append(fa - fb)
        deltas.sort()
        return {
            "mean_delta_macro_f1": float(np.mean(deltas)),
            "ci95": [float(deltas[50]), float(deltas[1949])],
            "n_boot": 2000,
            "note": "positive => model A better than B on macro-F1",
        }

    # Need stored probs — recompute text-only and text+vader probs for paired test
    # (already have in results via thresholds; recompute quickly)
    clf_text = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=SEED)
    clf_text.fit(X_train_text, y[train_core])
    p_text = clf_text.predict_proba(X_test_text)[:, 1]
    # text+vader
    s_tr = sentiment_matrix(train_core, ["vader_combined", "vader_title", "vader_body"])
    s_te = sentiment_matrix(test_idx, ["vader_combined", "vader_title", "vader_body"])
    s_tr, s_te = scale_fit_transform(s_tr, s_te)
    clf_tv = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=SEED)
    clf_tv.fit(hstack([X_train_text, csr_matrix(s_tr)]), y[train_core])
    p_tv = clf_tv.predict_proba(hstack([X_test_text, csr_matrix(s_te)]))[:, 1]

    paired = {
        "text_plus_vader_minus_text_only": paired_delta(
            p_tv,
            p_text,
            results["text_plus_vader"]["threshold_selected_on_val"],
            results["tfidf_text_only"]["threshold_selected_on_val"],
        )
    }

    # Provenance sensitivity: evaluate text-only on terra vs existing subsets of test
    provenance = {}
    for name, pred in (
        ("existing_in_test", lambda i: rows[i]["decision_source"] != "gpt-6.1-terra_ai_classification"),
        ("terra_in_test", lambda i: rows[i]["decision_source"] == "gpt-6.1-terra_ai_classification"),
    ):
        mask = np.asarray([pred(i) for i in test_idx])
        if mask.sum() < 5:
            provenance[name] = {"n": int(mask.sum())}
            continue
        sub_y = y[test_idx][mask]
        sub_p = p_text[mask]
        thr = results["tfidf_text_only"]["threshold_selected_on_val"]
        provenance[name] = _metrics(sub_y, (sub_p >= thr).astype(int), sub_p)

    report = {
        "utc": utcnow(),
        "task": "binary TPS(0) vs GDS(1) among frozen relevant; Other/Unclear excluded",
        "features_rule": "title/body only at posting time; no replies, labels, confidence, reasons, or decision_source",
        "cardiff_features": "not_available_blocked",
        "embedding_baseline": "not_run_no_local_pinned_minilm",
        "split": split_info,
        "models": results,
        "paired_bootstrap": paired,
        "provenance_sensitivity_text_only": provenance,
        "scientific_status": (
            "Exploratory agreement with mixed-source saved labels — NOT independent field accuracy. "
            "Human reference review still pending (200-case has zero decisions)."
        ),
    }
    write_json(dest / "RQ4_RESULTS.json", report)

    # Compact table
    import csv

    with (dest / "model_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "macro_f1",
                "gds_precision",
                "gds_recall",
                "gds_f1",
                "pr_auc_gds",
                "n_test",
                "n_gds_test",
            ],
        )
        w.writeheader()
        for name, m in results.items():
            w.writerow(
                {
                    "model": name,
                    "macro_f1": m.get("macro_f1"),
                    "gds_precision": m.get("gds_precision"),
                    "gds_recall": m.get("gds_recall"),
                    "gds_f1": m.get("gds_f1"),
                    "pr_auc_gds": m.get("pr_auc_gds"),
                    "n_test": m.get("n"),
                    "n_gds_test": m.get("n_gds"),
                }
            )

    (dest / "FINDINGS.md").write_text(
        f"""# Phase E findings — predictive RQ4 (exploratory)

**UTC:** {report['utc']}

Task: strict TPS vs GDS on frozen relevant posts (title/body only).  
Split grouped by near-duplicate text families; vectorizer/threshold fit on train/val only.

## Test metrics (see `model_comparison.csv`)
- Text-only TF-IDF macro-F1: **{results['tfidf_text_only']['macro_f1']:.4f}**; GDS P/R/F1 = {results['tfidf_text_only']['gds_precision']:.3f}/{results['tfidf_text_only']['gds_recall']:.3f}/{results['tfidf_text_only']['gds_f1']:.3f}
- Text+VADER macro-F1: **{results['text_plus_vader']['macro_f1']:.4f}**
- Text+TextBlob macro-F1: **{results['text_plus_textblob']['macro_f1']:.4f}**
- Paired Δ macro-F1 (text+VADER − text): {paired['text_plus_vader_minus_text_only']}

## Status
{report['scientific_status']}
Cardiff features blocked; embeddings not run.
""",
        encoding="utf-8",
    )
    (dest / "LIMITATIONS.md").write_text(
        """# Phase E limitations

- Labels are mixed human/AI provenance — exploratory prediction of saved labels.
- No independent human test set (200-case review empty).
- Strong class imbalance; report GDS P/R/F1 and PR-AUC, not accuracy.
- Cardiff/embedding baselines unavailable in this run.
- Single grouped split; temporal/subreddit transfer not exhaustively proven.
""",
        encoding="utf-8",
    )
    return report
