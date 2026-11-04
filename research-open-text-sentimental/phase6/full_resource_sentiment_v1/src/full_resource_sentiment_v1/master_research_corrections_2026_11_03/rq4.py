"""RQ4 correction: connected-component near-dup split + matched model rerun."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, hstack
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

from full_resource_sentiment_v1.master_research_corrections_2026_11_03 import SEED
from full_resource_sentiment_v1.master_research_corrections_2026_11_03.common import (
    DATASET_JSONL,
    FROZEN_DEF,
    ORIG_OUT,
    ORIG_RESTRICTED,
    OUT_ROOT,
    load_jsonl,
    sha256_text,
    utcnow,
    write_json,
)
from full_resource_sentiment_v1.master_research_corrections_2026_11_03.near_dup import (
    assert_no_cross_split_overlap,
    build_components,
)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    p, r, f, _s = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
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


def _load_rows() -> list[dict[str, Any]]:
    texts = {
        str(r["reddit_id"]): r for r in load_jsonl(ORIG_RESTRICTED / "text_thread_manifest_restricted.jsonl")
    }
    tb_scores = {}
    tb_path = ORIG_RESTRICTED / "textblob_tps_gds_scores.jsonl"
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
        text = f"{title}\n{body}".strip()
        sent = r.get("sentiment") or {}
        tb = tb_scores.get(rid) or {}
        rows.append(
            {
                "reddit_id": rid,
                "y": 1 if r["final_discussion"] == "GDS" else 0,
                "text": text,
                "title": title,
                "body": body,
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
    return rows


def run_rq4_correction() -> dict[str, Any]:
    dest = OUT_ROOT / "rq4_near_dup_split"
    dest.mkdir(parents=True, exist_ok=True)
    rows = _load_rows()
    ids = [r["reddit_id"] for r in rows]
    titles = [r["title"] for r in rows]
    bodies = [r["body"] for r in rows]
    y = np.asarray([r["y"] for r in rows], dtype=int)
    texts_list = [r["text"] for r in rows]

    comp = build_components(ids, titles, bodies)
    groups = np.asarray(comp["component_ids"])
    write_json(
        dest / "NEAR_DUP_AUDIT.json",
        {
            "utc": utcnow(),
            **{k: v for k, v in comp.items() if k != "component_ids"},
            "v1_reference": {
                "path": str(ORIG_OUT / "phase_e_predictive_rq4/RQ4_RESULTS.json"),
                "note": "v1 exploratory prefix-hash split preserved; not overwritten.",
            },
        },
    )

    # Freeze grouped split before fitting
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    train_idx, test_idx = next(gss.split(texts_list, y, groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED + 1)
    tr2, val_idx = next(gss2.split([texts_list[i] for i in train_idx], y[train_idx], groups[train_idx]))
    train_core = train_idx[tr2]
    val_core = train_idx[val_idx]

    overlap_tt = assert_no_cross_split_overlap(list(groups), train_core, test_idx)
    overlap_tv = assert_no_cross_split_overlap(list(groups), train_core, val_core)
    overlap_vt = assert_no_cross_split_overlap(list(groups), val_core, test_idx)
    assert overlap_tt == 0 and overlap_tv == 0 and overlap_vt == 0

    split_info = {
        "seed": SEED,
        "grouping": "connected_components_exact_full_plus_title_body_jaccard",
        "n_total": len(rows),
        "n_components": int(comp["n_components"]),
        "n_train": int(len(train_core)),
        "n_val": int(len(val_core)),
        "n_test": int(len(test_idx)),
        "test_gds": int(y[test_idx].sum()),
        "train_gds": int(y[train_core].sum()),
        "val_gds": int(y[val_core].sum()),
        "component_overlap": {
            "train_test": overlap_tt,
            "train_val": overlap_tv,
            "val_test": overlap_vt,
        },
        "test_ids_sha256": hashlib.sha256("\n".join(sorted(ids[i] for i in test_idx)).encode()).hexdigest(),
        "train_ids_sha256": hashlib.sha256("\n".join(sorted(ids[i] for i in train_core)).encode()).hexdigest(),
        "component_size_hist": comp["component_size_hist"],
    }
    write_json(dest / "SPLIT_FROZEN.json", split_info)

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
                row.append(0.0 if v is None else 1.0)
            mat.append(row)
        return np.asarray(mat, dtype=float)

    def scale_fit_transform(train_arr: np.ndarray, *others: np.ndarray):
        scaler = StandardScaler()
        tr = scaler.fit_transform(train_arr)
        return (tr,) + tuple(scaler.transform(a) for a in others)

    def choose_threshold(clf, X_val, y_val) -> tuple[float, float]:
        prob_val = clf.predict_proba(X_val)[:, 1]
        best_t, best_f = 0.5, -1.0
        for t in np.linspace(0.05, 0.95, 19):
            f = f1_score(y_val, (prob_val >= t).astype(int), average="macro", zero_division=0)
            if f > best_f:
                best_f, best_t = float(f), float(t)
        return best_t, best_f

    results: dict[str, Any] = {}

    maj = Counter(y[train_core]).most_common(1)[0][0]
    maj_pred = np.full_like(y[test_idx], maj)
    results["majority"] = _metrics(y[test_idx], maj_pred, np.full(len(test_idx), float(maj)))

    rng = np.random.default_rng(SEED)
    p_gds = float(y[train_core].mean())
    strat_pred = (rng.random(len(test_idx)) < p_gds).astype(int)
    results["stratified_dummy"] = _metrics(y[test_idx], strat_pred, np.full(len(test_idx), p_gds))

    clf_text = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=SEED)
    clf_text.fit(X_train_text, y[train_core])
    thr_text, val_f_text = choose_threshold(clf_text, X_val_text, y[val_core])
    p_text = clf_text.predict_proba(X_test_text)[:, 1]
    results["tfidf_text_only"] = {
        **_metrics(y[test_idx], (p_text >= thr_text).astype(int), p_text),
        "threshold_selected_on_val": thr_text,
        "val_macro_f1_at_threshold": val_f_text,
    }

    def text_plus(keys: list[str]) -> tuple[dict[str, Any], np.ndarray, float]:
        s_tr = sentiment_matrix(train_core, keys)
        s_va = sentiment_matrix(val_core, keys)
        s_te = sentiment_matrix(test_idx, keys)
        s_tr, s_va, s_te = scale_fit_transform(s_tr, s_va, s_te)
        Xtr = hstack([X_train_text, csr_matrix(s_tr)])
        Xva = hstack([X_val_text, csr_matrix(s_va)])
        Xte = hstack([X_test_text, csr_matrix(s_te)])
        clf2 = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=SEED)
        clf2.fit(Xtr, y[train_core])
        bt, bf = choose_threshold(clf2, Xva, y[val_core])
        pt = clf2.predict_proba(Xte)[:, 1]
        return (
            {
                **_metrics(y[test_idx], (pt >= bt).astype(int), pt),
                "threshold_selected_on_val": bt,
                "val_macro_f1_at_threshold": bf,
                "sentiment_keys": keys,
            },
            pt,
            bt,
        )

    results["text_plus_vader"], p_tv, thr_tv = text_plus(["vader_combined", "vader_title", "vader_body"])
    results["text_plus_textblob"], p_tb, thr_tb = text_plus(["tb_combined", "tb_title", "tb_body"])
    results["text_plus_vader_textblob"], _p_both, _thr_both = text_plus(
        ["vader_combined", "vader_title", "vader_body", "tb_combined", "tb_title", "tb_body"]
    )

    s_tr = sentiment_matrix(train_core, ["vader_combined", "vader_title", "vader_body", "tb_combined"])
    s_te = sentiment_matrix(test_idx, ["vader_combined", "vader_title", "vader_body", "tb_combined"])
    s_tr, s_te = scale_fit_transform(s_tr, s_te)
    clf_s = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=SEED)
    clf_s.fit(s_tr, y[train_core])
    ps = clf_s.predict_proba(s_te)[:, 1]
    results["sentiment_only_vader_textblob"] = _metrics(y[test_idx], (ps >= 0.5).astype(int), ps)

    def paired_delta(prob_a: np.ndarray, thr_a: float, prob_b: np.ndarray, thr_b: float) -> dict[str, Any]:
        rng_b = np.random.default_rng(SEED)
        y_t = y[test_idx]
        deltas = []
        for _ in range(2000):
            idx = rng_b.integers(0, len(y_t), size=len(y_t))
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

    paired = {
        "text_plus_vader_minus_text_only": paired_delta(p_tv, thr_tv, p_text, thr_text),
        "text_plus_textblob_minus_text_only": paired_delta(p_tb, thr_tb, p_text, thr_text),
    }

    # Temporal / source holdout sensitivity (fit on complementary training strata only)
    sensitivity: dict[str, Any] = {}
    for name, train_mask_fn, test_mask_fn in (
        (
            "train_older_test_october",
            lambda i: rows[i].get("selection_stratum") == "older",
            lambda i: rows[i].get("selection_stratum") == "october_2026",
        ),
        (
            "train_non_terra_test_terra",
            lambda i: rows[i].get("decision_source") != "gpt-6.1-terra_ai_classification",
            lambda i: rows[i].get("decision_source") == "gpt-6.1-terra_ai_classification",
        ),
    ):
        tr_m = np.asarray([train_mask_fn(i) for i in range(len(rows))])
        te_m = np.asarray([test_mask_fn(i) for i in range(len(rows))])
        # Drop any near-dup component that appears in both sides
        tr_comp = set(groups[tr_m].tolist())
        te_comp = set(groups[te_m].tolist())
        leak = tr_comp & te_comp
        te_m2 = np.asarray([te_m[i] and groups[i] not in leak for i in range(len(rows))])
        tr_m2 = np.asarray([tr_m[i] and groups[i] not in leak for i in range(len(rows))])
        if tr_m2.sum() < 50 or te_m2.sum() < 20 or y[te_m2].sum() < 3:
            sensitivity[name] = {
                "status": "insufficient_n",
                "n_train": int(tr_m2.sum()),
                "n_test": int(te_m2.sum()),
                "components_removed_for_leakage": len(leak),
            }
            continue
        vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3, max_features=50000, sublinear_tf=True)
        Xtr = vec.fit_transform([texts_list[i] for i in np.where(tr_m2)[0]])
        Xte = vec.transform([texts_list[i] for i in np.where(te_m2)[0]])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=SEED)
        clf.fit(Xtr, y[tr_m2])
        # Sentiment-augmented
        s_tr = sentiment_matrix(np.where(tr_m2)[0], ["vader_combined", "vader_title", "vader_body"])
        s_te = sentiment_matrix(np.where(te_m2)[0], ["vader_combined", "vader_title", "vader_body"])
        s_tr, s_te = scale_fit_transform(s_tr, s_te)
        clf2 = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=SEED)
        clf2.fit(hstack([Xtr, csr_matrix(s_tr)]), y[tr_m2])
        pt = clf.predict_proba(Xte)[:, 1]
        pv = clf2.predict_proba(hstack([Xte, csr_matrix(s_te)]))[:, 1]
        sensitivity[name] = {
            "n_train": int(tr_m2.sum()),
            "n_test": int(te_m2.sum()),
            "test_gds": int(y[te_m2].sum()),
            "components_removed_for_leakage": len(leak),
            "tfidf_text_only": _metrics(y[te_m2], (pt >= 0.5).astype(int), pt),
            "text_plus_vader": _metrics(y[te_m2], (pv >= 0.5).astype(int), pv),
            "delta_macro_f1_text_plus_vader_minus_text": float(
                _metrics(y[te_m2], (pv >= 0.5).astype(int), pv)["macro_f1"]
                - _metrics(y[te_m2], (pt >= 0.5).astype(int), pt)["macro_f1"]
            ),
        }

    import json
    from pathlib import Path

    v1: dict[str, Any] = {}
    v1_path = ORIG_OUT / "phase_e_predictive_rq4/RQ4_RESULTS.json"
    if v1_path.exists():
        v1 = json.loads(Path(v1_path).read_text(encoding="utf-8"))

    report = {
        "utc": utcnow(),
        "version": "rq4_v2_connected_component_near_dup",
        "task": "binary TPS(0) vs GDS(1) among frozen relevant; Other/Unclear excluded",
        "features_rule": "title/body only at posting time; no replies, labels, confidence, reasons, or decision_source",
        "scientific_status": (
            "Exploratory agreement with mixed-source saved labels — NOT independent field accuracy. "
            "Do not claim sentiment helps or hurts unless stable across v1 and v2 designs."
        ),
        "split": split_info,
        "near_dup_policy": comp["policy"],
        "models": results,
        "paired_bootstrap": paired,
        "sensitivity_holdouts": sensitivity,
        "v1_exploratory_reference": {
            "path": str(v1_path),
            "test_macro_f1_text": (v1.get("models") or {}).get("tfidf_text_only", {}).get("macro_f1"),
            "test_macro_f1_text_vader": (v1.get("models") or {}).get("text_plus_vader", {}).get("macro_f1"),
            "test_ids_sha256": (v1.get("split") or {}).get("test_ids_sha256"),
        },
    }
    write_json(dest / "RQ4_RESULTS_V2.json", report)

    with (dest / "model_comparison_v2.csv").open("w", encoding="utf-8", newline="") as handle:
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
        f"""# RQ4 v2 findings — connected-component near-dup split

**UTC:** {report['utc']}  
**Status:** exploratory label agreement; v1 retained at `{v1_path}`.

## Grouping
- Components: **{comp['n_components']}** from {len(rows)} posts; size≥2 components: {comp['n_components_size_ge2']}
- Policy: exact full text + shared title (≥{comp['policy']['title_min_chars']} chars) with body Jaccard ≥{comp['policy']['title_body_jaccard']}
- Cross-split component overlap train∩test/val: **0**

## Test metrics (v2)
- Text-only macro-F1: **{results['tfidf_text_only']['macro_f1']:.4f}** (GDS P/R/F1 {results['tfidf_text_only']['gds_precision']:.3f}/{results['tfidf_text_only']['gds_recall']:.3f}/{results['tfidf_text_only']['gds_f1']:.3f})
- Text+VADER macro-F1: **{results['text_plus_vader']['macro_f1']:.4f}**
- Text+TextBlob macro-F1: **{results['text_plus_textblob']['macro_f1']:.4f}**
- Paired Δ (text+VADER − text): {paired['text_plus_vader_minus_text_only']}

## Interpretation guard
Do **not** claim sentiment helps or hurts unless the direction is stable across v1 (prefix hash) and v2 (connected components) and holdouts.
""",
        encoding="utf-8",
    )
    (dest / "LIMITATIONS.md").write_text(
        """# RQ4 v2 limitations

- Mixed-source labels; no human reference accuracy.
- Near-dup policy catches exact text and same-title high body Jaccard; paraphrases with different titles may remain ungrouped.
- Single primary split + two holdouts; not exhaustive transfer proof.
- Thresholds predeclared but body-Jaccard still a heuristic.
""",
        encoding="utf-8",
    )
    return report
