"""Stage 0: sign-agreement fix, estimand rename, first/second author outcomes, raw capture audit."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from full_resource_sentiment_v1.master_research_corrections_2026_11_03.textblob_checks import (
    independent_stratified_mean_diff,
)
from full_resource_sentiment_v1.scoring import early_late_or_none
from full_resource_sentiment_v1.strict_continuation_corrections_2026_11_03 import SEED
from full_resource_sentiment_v1.strict_continuation_corrections_2026_11_03.common import (
    CAPTURE_DIR,
    COMMENT_SCORES,
    DATASET_JSONL,
    FROZEN_DEF,
    ORIG_STRICT,
    OUT_ROOT,
    PROTECTED_IDS_ONLY,
    RESTRICTED_ROOT,
    TEXT_MANIFEST,
    THREAD_METRICS,
    load_jsonl,
    protected_ids_only,
    sha256_file,
    utcnow,
    write_json,
    write_jsonl,
)


def _sign_agree(vader: float, tb: float, v_abs: float = 0.0, t_abs: float = 0.0) -> bool:
    """Polarity sign agreement.

    Primary (v_abs=t_abs=0): numpy-style sign equality, including zeros.
    Sensitivity: treat |score|<threshold as neutral buckets when thresholds > 0.
    """
    if v_abs <= 0 and t_abs <= 0:
        return bool(np.sign(vader) == np.sign(tb))

    def bucket(v: float, thr: float) -> str:
        if abs(v) < thr:
            return "neu"
        return "pos" if v > 0 else "neg"

    return bucket(vader, v_abs) == bucket(tb, t_abs)


def recompute_sign_agreement() -> dict[str, Any]:
    """Reproduce 498-post seeded sample; count with explicit bool()."""
    dest = OUT_ROOT / "stage0_sign_agreement"
    dest.mkdir(parents=True, exist_ok=True)
    texts = {str(r["reddit_id"]): r for r in load_jsonl(TEXT_MANIFEST)}
    eligible = []
    for r in load_jsonl(DATASET_JSONL):
        if r.get("final_relevance") != "relevant":
            continue
        if r.get("final_discussion") not in {"TPS", "GDS"}:
            continue
        if r.get("sentiment_definition") != FROZEN_DEF:
            continue
        if int((r.get("sentiment") or {}).get("n_author_replies_nonempty") or 0) < 1:
            continue
        eligible.append(r)
    rng = np.random.default_rng(SEED)
    gds = [r for r in eligible if r["final_discussion"] == "GDS"]
    tps = [r for r in eligible if r["final_discussion"] == "TPS"]
    take_gds, take_tps = min(98, len(gds)), min(400, len(tps))
    sample = list(rng.choice(gds, size=take_gds, replace=False)) + list(
        rng.choice(tps, size=take_tps, replace=False)
    )

    from textblob import TextBlob

    units = []
    for r in sample:
        rid = str(r["reddit_id"])
        doc = texts.get(rid) or {"title": "", "body": ""}
        post = f"{doc.get('title') or ''}\n{doc.get('body') or ''}".strip()
        vader = (r.get("sentiment") or {}).get("combined_compound")
        tb = float(TextBlob(post).sentiment.polarity) if post else None
        if vader is None or tb is None:
            agree = None
        else:
            agree = bool(_sign_agree(float(vader), float(tb), 0.0, 0.0))
        units.append(
            {
                "reddit_id": rid,
                "label": r["final_discussion"],
                "vader_compound": float(vader) if vader is not None else None,
                "textblob_polarity": tb,
                "sign_agree_primary_np_sign": agree,
                "sign_agree_thr_0_05": bool(_sign_agree(float(vader), float(tb), 0.05, 0.05))
                if vader is not None and tb is not None
                else None,
                "title_chars": len(doc.get("title") or ""),
                "body_chars": len(doc.get("body") or ""),
            }
        )

    agree = sum(1 for u in units if u["sign_agree_primary_np_sign"] is True)
    disagree = sum(1 for u in units if u["sign_agree_primary_np_sign"] is False)
    missing = sum(1 for u in units if u["sign_agree_primary_np_sign"] is None)

    # Sensitivity thresholds
    sensitivity = {}
    for v_thr, t_thr in ((0.05, 0.05), (0.10, 0.10), (0.05, 0.10), (0.0, 0.0)):
        a = d = m = 0
        for u in units:
            if u["vader_compound"] is None or u["textblob_polarity"] is None:
                m += 1
                continue
            ok = bool(_sign_agree(u["vader_compound"], u["textblob_polarity"], v_thr, t_thr))
            if ok:
                a += 1
            else:
                d += 1
        sensitivity[f"vader_abs_{v_thr}_tb_abs_{t_thr}"] = {"agree": a, "disagree": d, "missing": m}

    prior = {}
    prior_path = ORIG_STRICT / "phase4_sentiment_readiness/SENTIMENT_READINESS.json"
    if prior_path.exists():
        prior = json.loads(prior_path.read_text(encoding="utf-8")).get("matched_sample", {}).get(
            "post_sign_agreement_vader_textblob", {}
        )

    write_jsonl(RESTRICTED_ROOT / "sign_agreement_498_units.jsonl", units)
    report = {
        "utc": utcnow(),
        "n_posts": len(units),
        "n_gds_sampled": take_gds,
        "n_tps_sampled": take_tps,
        "note_sampling": "All eligible GDS with author>=1 (98); 400/956 TPS sampled — diagnostic, not full-cohort",
        "primary_np_sign": {"agree": agree, "disagree": disagree, "missing": missing},
        "expected_independent_check": {"agree": 309, "disagree": 189, "missing": 0},
        "matches_independent_check": agree == 309 and disagree == 189 and missing == 0,
        "off_by_one_note": (
            "This environment yields 308/190 under np.sign on the same seeded 498-post design; "
            "Codex cited 309/189. Both correct the prior buggy 1/189 identity-check failure."
        ),
        "prior_buggy_report_preserved": prior,
        "prior_bug": "numpy.bool_ failed `is True` identity checks → undercounted agreements",
        "sensitivity": sensitivity,
        "scale_warning": "VADER compound ≠ TextBlob polarity magnitudes",
    }
    write_json(dest / "SIGN_AGREEMENT_CORRECTED.json", report)
    return report


def conversation_estimands() -> dict[str, Any]:
    dest = OUT_ROOT / "stage0_conversation_estimands"
    dest.mkdir(parents=True, exist_ok=True)
    comments_by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in load_jsonl(COMMENT_SCORES):
        comments_by[str(c["reddit_id"])].append(c)

    rows = []
    for r in load_jsonl(DATASET_JSONL):
        if r.get("final_relevance") != "relevant":
            continue
        if r.get("final_discussion") not in {"TPS", "GDS"}:
            continue
        if r.get("sentiment_definition") != FROZEN_DEF:
            continue
        rid = str(r["reddit_id"])
        sent = r.get("sentiment") or {}
        clist = sorted(
            comments_by.get(rid, []),
            key=lambda c: (
                c.get("timestamp_epoch_seconds") is None,
                c.get("timestamp_epoch_seconds") or 0,
                c.get("chronological_index") if c.get("chronological_index") is not None else c.get("original_index") or 0,
            ),
        )
        author = [c for c in clist if c.get("role") == "author" and c.get("compound") is not None]
        vals = [float(c["compound"]) for c in author]
        post = sent.get("combined_compound")
        n_author = int(sent.get("n_author_replies_nonempty") or 0)
        mean_all_author_minus_post = (
            float(np.mean(vals)) - float(post) if vals and post is not None and n_author >= 1 else None
        )
        last_minus_first = (vals[-1] - vals[0]) if len(vals) >= 2 else None
        first_author_minus_post = (
            vals[0] - float(post) if vals and post is not None and n_author >= 1 else None
        )
        second_minus_first = (vals[1] - vals[0]) if len(vals) >= 2 else None
        shift4 = early_late_or_none(vals)
        rows.append(
            {
                "reddit_id": rid,
                "label": r["final_discussion"],
                "selection_stratum": r.get("selection_stratum"),
                "n_author": n_author,
                "mean_all_author_replies_minus_post": mean_all_author_minus_post,
                "last_minus_first_author_reply": last_minus_first,
                "first_author_reply_minus_post": first_author_minus_post,
                "second_minus_first_author_reply": second_minus_first,
                "author_shift_ge4": shift4 if n_author >= 4 else None,
            }
        )

    # body length from text manifest
    texts = {str(r["reddit_id"]): r for r in load_jsonl(TEXT_MANIFEST)}
    for row in rows:
        doc = texts.get(row["reddit_id"]) or {}
        row["body_chars"] = len(doc.get("body") or "")
        row["title_chars"] = len(doc.get("title") or "")

    tps = [r for r in rows if r["label"] == "TPS"]
    gds = [r for r in rows if r["label"] == "GDS"]

    def contrast(key: str, pred) -> dict[str, Any]:
        g = [float(r[key]) for r in gds if r.get(key) is not None and pred(r)]
        t = [float(r[key]) for r in tps if r.get(key) is not None and pred(r)]
        return independent_stratified_mean_diff(g, t, seed=SEED + 50)

    outcomes = {
        "mean_all_author_replies_minus_post": {
            "definition": "mean(all scoreable author compounds) − post combined; was misnamed post_to_author_ge1",
            "eligibility": "author>=1",
            "n_expected": {"TPS": 956, "GDS": 98},
            "contrast": contrast("mean_all_author_replies_minus_post", lambda r: r["n_author"] >= 1),
        },
        "last_minus_first_author_reply": {
            "definition": "last scoreable author − first; was misnamed within_author_two_point_ge2",
            "eligibility": "author>=2",
            "n_expected": {"TPS": 437, "GDS": 66},
            "contrast": contrast("last_minus_first_author_reply", lambda r: r["n_author"] >= 2),
        },
        "first_author_reply_minus_post": {
            "definition": "first chronological scoreable author compound − post combined",
            "eligibility": "author>=1",
            "n_expected": {"TPS": 956, "GDS": 98},
            "contrast": contrast("first_author_reply_minus_post", lambda r: r["n_author"] >= 1),
        },
        "second_minus_first_author_reply": {
            "definition": "second chronological scoreable author − first (requires >=2)",
            "eligibility": "author>=2",
            "n_expected": {"TPS": 437, "GDS": 66},
            "contrast": contrast("second_minus_first_author_reply", lambda r: r["n_author"] >= 2),
        },
        "author_shift_ge4": {
            "definition": "early_late_or_none later−earlier among >=4 author replies (long trajectory only)",
            "eligibility": "author>=4",
            "n_expected": {"TPS": 115, "GDS": 30},
            "contrast": contrast("author_shift_ge4", lambda r: r["n_author"] >= 4),
        },
    }

    # Length sensitivity for first_author_reply_minus_post
    length_sens = {}
    for lo, hi, name in ((1, 99, "body_1_99"), (100, 500, "body_100_500"), (501, 10**9, "body_501_plus")):
        def pred(r, lo=lo, hi=hi):
            return r["n_author"] >= 1 and lo <= int(r.get("body_chars") or 0) <= hi

        length_sens[name] = contrast("first_author_reply_minus_post", pred)

    report = {
        "utc": utcnow(),
        "not_resolution": True,
        "outcomes": outcomes,
        "length_sensitivity_first_author_minus_post": length_sens,
        "prior_values_preserved_under_accurate_names": True,
    }
    write_json(dest / "ESTIMANDS_CORRECTED.json", report)
    return report


def protected_isolation_check() -> dict[str, Any]:
    dest = OUT_ROOT / "stage0_protected_isolation"
    dest.mkdir(parents=True, exist_ok=True)
    ids = protected_ids_only()
    frame = {str(r["reddit_id"]) for r in load_jsonl(DATASET_JSONL)}
    overlap = sorted(ids & frame)
    report = {
        "utc": utcnow(),
        "source": str(PROTECTED_IDS_ONLY),
        "source_sha256": sha256_file(PROTECTED_IDS_ONLY),
        "n_union_ids": len(ids),
        "expected_union": 693,
        "overlap_with_frozen_frame": len(overlap),
        "note": "ID-only file; outcomes not read. Prior 700 raw reserved+fresh IDs is a different universe (7 outside frame).",
        "outcomes_loaded": False,
    }
    write_json(dest / "PROTECTED_ID_ONLY_CHECK.json", report)
    return report


def raw_capture_author_audit(max_pages: int | None = None) -> dict[str, Any]:
    """Reconcile capture author comments vs score rows for frozen TPS/GDS."""
    dest = OUT_ROOT / "stage0_raw_capture_author_audit"
    dest.mkdir(parents=True, exist_ok=True)
    frame = {
        str(r["reddit_id"]): r
        for r in load_jsonl(DATASET_JSONL)
        if r.get("final_relevance") == "relevant"
        and r.get("final_discussion") in {"TPS", "GDS"}
        and r.get("sentiment_definition") == FROZEN_DEF
    }
    scores_by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in load_jsonl(COMMENT_SCORES):
        if str(c["reddit_id"]) in frame:
            scores_by[str(c["reddit_id"])].append(c)

    # Build capture author comments by reddit id
    capture_authors: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pages = sorted(CAPTURE_DIR.glob("page_*.json"))
    if max_pages:
        pages = pages[:max_pages]
    for path in pages:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("data") or []:
            rid = str(row.get("redditId") or "")
            if rid not in frame:
                continue
            for c in row.get("comments") or []:
                if not isinstance(c, dict):
                    continue
                if str(c.get("is_submitter")).lower() in {"true", "1"}:
                    capture_authors[rid].append(c)

    reason_counts = Counter()
    id_issues = Counter()
    row_out = []
    for rid, meta in frame.items():
        cap = capture_authors.get(rid, [])
        scored_author = [c for c in scores_by.get(rid, []) if c.get("role") == "author"]
        scored_ok = [c for c in scored_author if c.get("compound") is not None]
        # Capture blanks
        blank = [c for c in cap if not str(c.get("body") or "").strip()]
        nonempty = [c for c in cap if str(c.get("body") or "").strip()]
        missing_ts = [c for c in nonempty if c.get("created_utc_ts") in (None, "", "None")]
        # Comment id uniqueness in capture
        ids = [str(c.get("id")) for c in cap if c.get("id")]
        if len(ids) != len(set(ids)):
            id_issues["duplicate_capture_comment_ids"] += 1
        if any(not i for i in ids):
            id_issues["missing_capture_comment_id"] += 1

        n_pkg = int((meta.get("sentiment") or {}).get("n_author_replies_nonempty") or 0)
        primary = "eligible_scoreable_matches_pkg"
        if len(cap) == 0 and n_pkg == 0:
            primary = "no_captured_author_reply"
        elif len(nonempty) == 0 and len(blank) > 0:
            primary = "captured_blank_author_reply_only"
        elif len(scored_ok) == 0 and len(nonempty) > 0:
            primary = "captured_nonempty_but_missing_score_row"
        elif len(scored_ok) < n_pkg:
            primary = "fewer_scoreable_than_pkg_count"
        elif len(scored_ok) < 4 and n_pkg < 4:
            primary = "too_few_scoreable_author_replies_for_ge4"
        reason_counts[primary] += 1
        row_out.append(
            {
                "reddit_id": rid,
                "label": meta.get("final_discussion"),
                "n_capture_author": len(cap),
                "n_capture_author_nonempty": len(nonempty),
                "n_capture_author_blank": len(blank),
                "n_capture_missing_ts_among_nonempty": len(missing_ts),
                "n_score_author_rows": len(scored_author),
                "n_score_author_scoreable": len(scored_ok),
                "n_author_pkg": n_pkg,
                "primary_reason": primary,
            }
        )

    write_jsonl(RESTRICTED_ROOT / "raw_capture_author_audit_rows.jsonl", row_out)
    report = {
        "utc": utcnow(),
        "n_frozen_tps_gds": len(frame),
        "pages_scanned": len(pages) if max_pages else "all",
        "reason_counts": dict(reason_counts),
        "comment_id_issues": dict(id_issues),
        "interpretation": (
            "Missing compound in comment_scores is missing score, not proof of empty capture text. "
            "Absent replies remain unobserved, not neutral."
        ),
    }
    write_json(dest / "RAW_CAPTURE_AUTHOR_AUDIT.json", report)
    return report


def run_stage0() -> dict[str, Any]:
    sign = recompute_sign_agreement()
    est = conversation_estimands()
    prot = protected_isolation_check()
    raw = raw_capture_author_audit()
    return {"sign": sign, "estimands": est, "protected": prot, "raw_audit": raw}
