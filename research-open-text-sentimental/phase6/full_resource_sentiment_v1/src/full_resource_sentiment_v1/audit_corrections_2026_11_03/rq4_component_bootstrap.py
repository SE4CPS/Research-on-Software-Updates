"""Item 5: RQ4 with component-level paired bootstrap + five-split sentiment sensitivity."""

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
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupShuffleSplit

from full_resource_sentiment_v1.audit_corrections_2026_11_03 import SEED
from full_resource_sentiment_v1.audit_corrections_2026_11_03.common import (
    CARDIFF_SCORES,
    DATASET_JSONL,
    FROZEN_DEF,
    OUT_ROOT,
    RQ4_V2,
    TB_SCORES,
    TEXT_MANIFEST,
    load_jsonl,
    protected_ids_only,
    utcnow,
    write_json,
)
from full_resource_sentiment_v1.master_research_corrections_2026_11_03.near_dup import build_components


def run_rq4() -> dict[str, Any]:
    dest = OUT_ROOT / "item5_rq4_component_bootstrap"
    dest.mkdir(parents=True, exist_ok=True)
    prot = protected_ids_only()
    texts_map = {str(r["reddit_id"]): r for r in load_jsonl(TEXT_MANIFEST)}
    tb = {str(r["reddit_id"]): r for r in load_jsonl(TB_SCORES)} if TB_SCORES.exists() else {}
    cardiff = {
        r["reddit_id"]: r
        for r in load_jsonl(CARDIFF_SCORES)
        if r.get("stratum") == "original_6425_frozen_comparable" and r.get("unit") == "original_post"
    }

    rows = []
    for r in load_jsonl(DATASET_JSONL):
        rid = str(r["reddit_id"])
        if rid in prot:
            continue
        if r.get("final_relevance") != "relevant" or r.get("final_discussion") not in {"TPS", "GDS"}:
            continue
        if r.get("sentiment_definition") != FROZEN_DEF:
            continue
        doc = texts_map.get(rid) or {"title": "", "body": ""}
        title, body = doc.get("title") or "", doc.get("body") or ""
        sent = r.get("sentiment") or {}
        tbr = tb.get(rid) or {}
        cf = cardiff.get(rid) or {}
        rows.append(
            {
                "reddit_id": rid,
                "y": 1 if r["final_discussion"] == "GDS" else 0,
                "text": f"{title}\n{body}".strip(),
                "title": title,
                "body": body,
                "vader": sent.get("combined_compound"),
                "textblob": tbr.get("textblob_combined_polarity"),
                "cardiff_sign": cf.get("cardiff_polarity_sign"),
                "cardiff_prob_pos": (cf.get("cardiff_probs") or [None, None, None])[2]
                if cf.get("cardiff_probs")
                else None,
            }
        )

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
    v2 = json.loads(Path(RQ4_V2).read_text(encoding="utf-8")) if Path(RQ4_V2).exists() else {}
    fixed_ok = test_sha == (v2.get("split") or {}).get("test_ids_sha256")

    def choose_thr(prob, yv):
        best_t, best_f = 0.5, -1.0
        for t in np.linspace(0.05, 0.95, 19):
            f = f1_score(yv, (prob >= t).astype(int), average="macro", zero_division=0)
            if f > best_f:
                best_f, best_t = float(f), float(t)
        return best_t

    def metrics(yt, pred, prob):
        p, r, f, _ = precision_recall_fscore_support(yt, pred, labels=[0, 1], zero_division=0)
        return {
            "macro_f1": float(f1_score(yt, pred, average="macro", zero_division=0)),
            "gds_precision": float(p[1]),
            "gds_recall": float(r[1]),
            "gds_f1": float(f[1]),
            "pr_auc_gds": float(average_precision_score(yt, prob)),
            "brier": float(brier_score_loss(yt, prob)),
            "confusion_matrix_[TPS,GDS]": confusion_matrix(yt, pred, labels=[0, 1]).tolist(),
        }

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

    models = {}
    for name, cw, keys in (
        ("text_unweighted", None, []),
        ("text_balanced", "balanced", []),
        ("text_plus_vader_tb_balanced", "balanced", ["vader", "textblob"]),
        ("text_plus_vader_tb_cardiff_balanced", "balanced", ["vader", "textblob", "cardiff_sign", "cardiff_prob_pos"]),
    ):
        Xt_tr, Xt_va, Xt_te = Xtr, Xva, Xte
        if keys:
            Xt_tr = hstack([Xtr, csr_matrix(sent_mat(train_core, keys))])
            Xt_va = hstack([Xva, csr_matrix(sent_mat(val_core, keys))])
            Xt_te = hstack([Xte, csr_matrix(sent_mat(test_idx, keys))])
        clf = LogisticRegression(max_iter=2000, class_weight=cw, solver="liblinear", random_state=SEED)
        clf.fit(Xt_tr, y[train_core])
        thr = choose_thr(clf.predict_proba(Xt_va)[:, 1], y[val_core])
        prob = clf.predict_proba(Xt_te)[:, 1]
        pred = (prob >= thr).astype(int)
        models[name] = {**metrics(y[test_idx], pred, prob), "threshold": thr}

    def fit_pred(keys, cw="balanced"):
        Xt_tr, Xt_va, Xt_te = Xtr, Xva, Xte
        if keys:
            Xt_tr = hstack([Xtr, csr_matrix(sent_mat(train_core, keys))])
            Xt_va = hstack([Xva, csr_matrix(sent_mat(val_core, keys))])
            Xt_te = hstack([Xte, csr_matrix(sent_mat(test_idx, keys))])
        clf = LogisticRegression(max_iter=2000, class_weight=cw, solver="liblinear", random_state=SEED)
        clf.fit(Xt_tr, y[train_core])
        thr = choose_thr(clf.predict_proba(Xt_va)[:, 1], y[val_core])
        return (clf.predict_proba(Xt_te)[:, 1] >= thr).astype(int)

    pred_text = fit_pred([])
    pred_two = fit_pred(["vader", "textblob"])
    pred_cf = fit_pred(["vader", "textblob", "cardiff_sign", "cardiff_prob_pos"])
    yt = y[test_idx]
    test_groups = groups[test_idx]
    uniq_comp = np.unique(test_groups)

    def component_bootstrap(pred_a, pred_b, seed, n_boot=10000):
        """Resample whole duplicate components in the test set."""
        rng = np.random.default_rng(seed)
        # map component -> indices within test
        comp_to_idx = {c: np.where(test_groups == c)[0] for c in uniq_comp}
        diffs = []
        for _ in range(n_boot):
            chosen = rng.choice(uniq_comp, size=len(uniq_comp), replace=True)
            idx = np.concatenate([comp_to_idx[c] for c in chosen])
            fa = f1_score(yt[idx], pred_a[idx], average="macro", zero_division=0)
            fb = f1_score(yt[idx], pred_b[idx], average="macro", zero_division=0)
            diffs.append(fb - fa)
        a = np.asarray(diffs)
        return {
            "mean": float(a.mean()),
            "ci95": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))],
            "n_bootstrap": n_boot,
            "bootstrap_unit": "near_dup_component",
            "n_test_components": int(len(uniq_comp)),
            "seed": seed,
        }

    boot_two_minus_text = component_bootstrap(pred_text, pred_two, SEED + 99)
    boot_cf_minus_text = component_bootstrap(pred_text, pred_cf, SEED + 100)
    boot_cf_minus_two = component_bootstrap(pred_two, pred_cf, SEED + 101)

    # Five-split stability for text-only AND text+sentiment
    stability = []
    gss5 = GroupShuffleSplit(n_splits=5, test_size=0.25, random_state=SEED + 5)
    for fold, (tr, te) in enumerate(gss5.split(texts, y, groups)):
        gss_v = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED + 5 + fold)
        trc, vac = next(gss_v.split([texts[i] for i in tr], y[tr], groups[tr]))
        tr_core, va_core = tr[trc], tr[vac]
        v = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3, max_features=50000, sublinear_tf=True)
        Xtr_f = v.fit_transform([texts[i] for i in tr_core])
        Xva_f = v.transform([texts[i] for i in va_core])
        Xte_f = v.transform([texts[i] for i in te])
        fold_models = {}
        for name, keys in (
            ("text_balanced", []),
            ("text_plus_vader_tb_balanced", ["vader", "textblob"]),
            ("text_plus_vader_tb_cardiff_balanced", ["vader", "textblob", "cardiff_sign", "cardiff_prob_pos"]),
        ):
            Xt_tr, Xt_va, Xt_te = Xtr_f, Xva_f, Xte_f
            if keys:
                Xt_tr = hstack([Xtr_f, csr_matrix(sent_mat(tr_core, keys))])
                Xt_va = hstack([Xva_f, csr_matrix(sent_mat(va_core, keys))])
                Xt_te = hstack([Xte_f, csr_matrix(sent_mat(te, keys))])
            clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=SEED)
            clf.fit(Xt_tr, y[tr_core])
            thr = choose_thr(clf.predict_proba(Xt_va)[:, 1], y[va_core])
            prob = clf.predict_proba(Xt_te)[:, 1]
            pred = (prob >= thr).astype(int)
            fold_models[name] = metrics(y[te], pred, prob)
        stability.append({"fold": fold, "models": fold_models})

    report = {
        "utc": utcnow(),
        "fixed_rq4_v2_split": fixed_ok,
        "test_ids_sha256": test_sha,
        "n_train": int(len(train_core)),
        "n_val": int(len(val_core)),
        "n_test": int(len(test_idx)),
        "models": models,
        "component_bootstrap_paired_macro_f1": {
            "two_feature_minus_text_balanced": boot_two_minus_text,
            "cardiff_inclusive_minus_text_balanced": boot_cf_minus_text,
            "cardiff_inclusive_minus_two_feature": boot_cf_minus_two,
        },
        "stability_5_grouped_with_sentiment_models": stability,
        "scientific_status": "Saved-label agreement only — not human accuracy; no test tuning",
        "prior_row_bootstrap_downgraded": (
            "Previous sealed row-bootstrap interval is exploratory; this packet uses component resampling"
        ),
    }
    write_json(dest / "RQ4_COMPONENT_BOOTSTRAP.json", report)
    return report
