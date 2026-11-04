"""RQ1–RQ4 on corrected dataset v2 (original-only and AI-assisted strata)."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
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

from full_resource_sentiment_v1.correction_rq_cardiff_2026_11_03 import DATASET_VERSION, N_BOOT, SEED
from full_resource_sentiment_v1.correction_rq_cardiff_2026_11_03.common import (
    COMMENT_SCORES,
    DATASET_JSONL,
    EXACT_JOIN_RESTRICTED,
    FROZEN_DEF,
    OUT_ROOT,
    RESTRICTED_ROOT,
    RQ4_V2,
    TB_SCORES,
    TEXT_MANIFEST,
    THREAD_METRICS,
    load_jsonl,
    protected_ids_only,
    utcnow,
    write_json,
)
from full_resource_sentiment_v1.master_research_corrections_2026_11_03.near_dup import build_components
from full_resource_sentiment_v1.master_research_corrections_2026_11_03.textblob_checks import (
    independent_stratified_mean_diff,
)
from full_resource_sentiment_v1.scoring import early_late_or_none


def _contrast(g, t, seed):
    return independent_stratified_mean_diff(g, t, seed=seed, n_boot=N_BOOT)


def run_rq1_rq3() -> dict[str, Any]:
    dest = OUT_ROOT / "stage3_rq1_rq3"
    dest.mkdir(parents=True, exist_ok=True)
    union = load_jsonl(RESTRICTED_ROOT / f"{DATASET_VERSION}.jsonl")
    cardiff_scores = {}
    cpath = RESTRICTED_ROOT / "cardiff_full_scores.jsonl"
    if cpath.exists():
        for r in load_jsonl(cpath):
            if r.get("unit") == "original_post":
                cardiff_scores[(r["reddit_id"], r.get("stratum"))] = r

    # --- Original-only frozen comparable ---
    comments_by: dict[str, list] = defaultdict(list)
    for c in load_jsonl(COMMENT_SCORES):
        comments_by[str(c["reddit_id"])].append(c)
    tm = {str(r["reddit_id"]): r for r in load_jsonl(THREAD_METRICS)}
    tb_map = {str(r["reddit_id"]): r for r in load_jsonl(TB_SCORES)} if TB_SCORES.exists() else {}
    analysis = load_jsonl(EXACT_JOIN_RESTRICTED / "analysis_units_exact_join.jsonl")
    tb_unit = defaultdict(dict)
    for u in analysis:
        if u.get("textblob") is not None:
            tb_unit[u["reddit_id"]][u["unit"]] = float(u["textblob"])

    orig_rows = []
    for r in load_jsonl(DATASET_JSONL):
        if r.get("final_relevance") != "relevant" or r.get("final_discussion") not in {"TPS", "GDS"}:
            continue
        if r.get("sentiment_definition") != FROZEN_DEF:
            continue
        rid = str(r["reddit_id"])
        sent = r.get("sentiment") or {}
        post = sent.get("combined_compound")
        clist = sorted(
            comments_by.get(rid, []),
            key=lambda c: (
                c.get("timestamp_epoch_seconds") is None,
                c.get("timestamp_epoch_seconds") or 0,
                c.get("chronological_index") or 0,
            ),
        )
        author = [c for c in clist if c.get("role") == "author" and c.get("compound") is not None]
        community = [c for c in clist if c.get("role") == "community" and c.get("compound") is not None]
        vals = [float(c["compound"]) for c in author]
        cvals = [float(c["compound"]) for c in community]
        n_a = int(sent.get("n_author_replies_nonempty") or 0)
        n_c = int(sent.get("n_community_replies_nonempty") or 0)
        tbr = tb_map.get(rid) or {}
        tb_post = tbr.get("textblob_combined_polarity")
        if tb_post is None:
            tb_post = tb_unit[rid].get("original_post")
        cf = cardiff_scores.get((rid, "original_6425_frozen_comparable"))
        orig_rows.append(
            {
                "reddit_id": rid,
                "label": r["final_discussion"],
                "stratum": "original_only",
                "decision_source": r.get("decision_source"),
                "n_author": n_a,
                "n_community": n_c,
                "post_vader": float(post) if post is not None else None,
                "post_textblob": float(tb_post) if tb_post is not None else None,
                "post_cardiff_sign": float(cf["cardiff_polarity_sign"]) if cf else None,
                "title_textblob": tb_unit[rid].get("title_only"),
                "body_textblob": tb_unit[rid].get("body_only"),
                "first_author_minus_post": (vals[0] - float(post)) if vals and post is not None else None,
                "second_minus_first": (vals[1] - vals[0]) if len(vals) >= 2 else None,
                "author_shift_ge4": early_late_or_none(vals) if n_a >= 4 else None,
                "author_minus_community_mean": (
                    float(np.mean(vals)) - float(np.mean(cvals)) if vals and cvals else None
                ),
                "post_to_first_reply_hours": (tm.get(rid) or {}).get("post_to_first_reply_hours")
                if (tm.get(rid) or {}).get("post_to_first_reply_eligible")
                else None,
                "text_len": int(sent.get("combined_len") or 0) if sent.get("combined_len") is not None else None,
            }
        )

    # AI-assisted stratum (posts only; VADER/TB/Cardiff from cardiff full input scores)
    ai_rows = []
    for r in union:
        if r.get("record_origin") != "ai_assisted_screen_candidate_v2":
            continue
        if r.get("final_discussion") not in {"TPS", "GDS"}:
            continue
        if r.get("exact_dup_with_train"):
            continue  # primary AI analysis excludes exact train dups
        rid = r["reddit_id"]
        cf = cardiff_scores.get((rid, "ai_assisted_v2"))
        # Prefer cardiff file which also has vader/textblob for AI posts
        ai_rows.append(
            {
                "reddit_id": rid,
                "label": r["final_discussion"],
                "stratum": "ai_assisted_v2",
                "decision_source": r.get("decision_source"),
                "n_author": r.get("n_author"),
                "n_community": r.get("n_community"),
                "post_vader": float(cf["vader"]) if cf and cf.get("vader") is not None else None,
                "post_textblob": float(cf["textblob"]) if cf and cf.get("textblob") is not None else None,
                "post_cardiff_sign": float(cf["cardiff_polarity_sign"]) if cf else None,
                "near_dup_with_train": r.get("near_dup_with_train"),
            }
        )

    def gds_tps(rows, key, seed):
        g = [float(r[key]) for r in rows if r["label"] == "GDS" and r.get(key) is not None]
        t = [float(r[key]) for r in rows if r["label"] == "TPS" and r.get(key) is not None]
        return _contrast(g, t, seed)

    def pack(rows, seed_base):
        return {
            "n": len(rows),
            "n_gds": sum(1 for r in rows if r["label"] == "GDS"),
            "n_tps": sum(1 for r in rows if r["label"] == "TPS"),
            "rq1_post_vader": gds_tps(rows, "post_vader", seed_base),
            "rq1_post_textblob": gds_tps(rows, "post_textblob", seed_base + 1),
            "rq1_post_cardiff_sign": gds_tps(rows, "post_cardiff_sign", seed_base + 2),
            "rq1_title_textblob": gds_tps(rows, "title_textblob", seed_base + 3)
            if any(r.get("title_textblob") is not None for r in rows)
            else None,
            "rq1_body_textblob": gds_tps(rows, "body_textblob", seed_base + 4)
            if any(r.get("body_textblob") is not None for r in rows)
            else None,
        }

    orig_ge1 = [r for r in orig_rows if (r.get("n_author") or 0) >= 1]
    orig_ge2 = [r for r in orig_rows if (r.get("n_author") or 0) >= 2]
    orig_ge4 = [r for r in orig_rows if (r.get("n_author") or 0) >= 4]
    orig_both = [r for r in orig_rows if (r.get("n_author") or 0) >= 1 and (r.get("n_community") or 0) >= 1]

    results = {
        "utc": utcnow(),
        "dataset_version": DATASET_VERSION,
        "interpretation": "Saved-label / AI-candidate contrasts; not human accuracy; sentiment change ≠ resolution",
        "rq1_original_only": pack(orig_rows, SEED + 100),
        "rq1_ai_assisted_v2_excl_exact_train_dups": pack(ai_rows, SEED + 200),
        "rq2_original_author_ge1_first_minus_post": {
            "eligibility": {"n_tps": sum(1 for r in orig_ge1 if r["label"] == "TPS"), "n_gds": sum(1 for r in orig_ge1 if r["label"] == "GDS")},
            "contrast": gds_tps(orig_ge1, "first_author_minus_post", SEED + 300),
        },
        "rq2_original_author_ge2_second_minus_first": {
            "eligibility": {"n_tps": sum(1 for r in orig_ge2 if r["label"] == "TPS" and r.get("second_minus_first") is not None), "n_gds": sum(1 for r in orig_ge2 if r["label"] == "GDS" and r.get("second_minus_first") is not None)},
            "contrast": gds_tps(orig_ge2, "second_minus_first", SEED + 301),
        },
        "rq2_original_author_ge4_shift": {
            "eligibility": {"n_tps": sum(1 for r in orig_ge4 if r["label"] == "TPS" and r.get("author_shift_ge4") is not None), "n_gds": sum(1 for r in orig_ge4 if r["label"] == "GDS" and r.get("author_shift_ge4") is not None)},
            "contrast": gds_tps(orig_ge4, "author_shift_ge4", SEED + 302),
        },
        "rq3_author_minus_community": {
            "eligibility": {"n": len(orig_both), "n_gds": sum(1 for r in orig_both if r["label"] == "GDS"), "n_tps": sum(1 for r in orig_both if r["label"] == "TPS")},
            "contrast": gds_tps(orig_both, "author_minus_community_mean", SEED + 400),
        },
        "sensitivity_ai_excl_near_dup_also": pack(
            [r for r in ai_rows if not r.get("near_dup_with_train")], SEED + 210
        ),
    }
    write_json(dest / "RQ1_RQ3_RESULTS.json", results)
    return results


def run_rq4() -> dict[str, Any]:
    dest = OUT_ROOT / "stage3_rq4_predictive"
    dest.mkdir(parents=True, exist_ok=True)
    prot = protected_ids_only()
    texts_map = {str(r["reddit_id"]): r for r in load_jsonl(TEXT_MANIFEST)}
    tb = {str(r["reddit_id"]): r for r in load_jsonl(TB_SCORES)} if TB_SCORES.exists() else {}
    cardiff = {}
    cpath = RESTRICTED_ROOT / "cardiff_full_scores.jsonl"
    if cpath.exists():
        for r in load_jsonl(cpath):
            if r.get("unit") == "original_post" and r.get("stratum") == "original_6425_frozen_comparable":
                cardiff[r["reddit_id"]] = r

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
        mode = int(np.bincount(y[train_core]).argmax())
        maj = np.full(len(test_idx), mode)
        models[name] = {
            **metrics(y[test_idx], pred, prob),
            "threshold": thr,
            "majority_macro_f1": float(f1_score(y[test_idx], maj, average="macro", zero_division=0)),
        }

    # Paired bootstrap text_balanced vs text_plus_vader_tb
    def preds(keys, cw="balanced"):
        Xt_tr, Xt_va, Xt_te = Xtr, Xva, Xte
        if keys:
            Xt_tr = hstack([Xtr, csr_matrix(sent_mat(train_core, keys))])
            Xt_va = hstack([Xva, csr_matrix(sent_mat(val_core, keys))])
            Xt_te = hstack([Xte, csr_matrix(sent_mat(test_idx, keys))])
        clf = LogisticRegression(max_iter=2000, class_weight=cw, solver="liblinear", random_state=SEED)
        clf.fit(Xt_tr, y[train_core])
        thr = choose_thr(clf.predict_proba(Xt_va)[:, 1], y[val_core])
        return (clf.predict_proba(Xt_te)[:, 1] >= thr).astype(int)

    pred_base = preds([])
    pred_sent = preds(["vader", "textblob"])
    yt = y[test_idx]
    rng = np.random.default_rng(SEED + 99)
    diffs = []
    for _ in range(10000):
        idx = rng.integers(0, len(yt), size=len(yt))
        diffs.append(
            f1_score(yt[idx], pred_sent[idx], average="macro", zero_division=0)
            - f1_score(yt[idx], pred_base[idx], average="macro", zero_division=0)
        )
    diffs_a = np.asarray(diffs)

    # Stability 5 splits
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
        thr = choose_thr(clf.predict_proba(Xva_f)[:, 1], y[va_core])
        prob = clf.predict_proba(Xte_f)[:, 1]
        pred = (prob >= thr).astype(int)
        m = metrics(y[te], pred, prob)
        m["fold"] = fold
        stability.append(m)

    report = {
        "utc": utcnow(),
        "dataset_version": DATASET_VERSION,
        "protected_excluded": len(prot),
        "fixed_rq4_v2_split": fixed_ok,
        "test_ids_sha256": test_sha,
        "n_train": int(len(train_core)),
        "n_val": int(len(val_core)),
        "n_test": int(len(test_idx)),
        "models": models,
        "paired_bootstrap_sentiment_minus_text_macro_f1": {
            "mean": float(diffs_a.mean()),
            "ci95": [float(np.percentile(diffs_a, 2.5)), float(np.percentile(diffs_a, 97.5))],
            "n_bootstrap": 10000,
        },
        "stability_5_grouped": stability,
        "scientific_status": "Saved-label agreement on frozen original TPS/GDS only; protected unused for tuning",
    }
    write_json(dest / "RQ4_RESULTS.json", report)
    return report


def run_rq() -> dict[str, Any]:
    r13 = run_rq1_rq3()
    r4 = run_rq4()
    return {"rq1_rq3": r13, "rq4": r4}
