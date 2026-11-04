"""Exact capture_page + reddit_id + original_index join; recompute reply TextBlob."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from full_resource_sentiment_v1.comments import extract_comment_text
from full_resource_sentiment_v1.exact_join_repair_2026_11_03 import N_BOOT, SEED
from full_resource_sentiment_v1.exact_join_repair_2026_11_03.common import (
    CAPTURE_DIR,
    COMMENT_SCORES,
    DATASET_JSONL,
    FROZEN_DEF,
    OUT_ROOT,
    PRIOR_MATCHED,
    RESTRICTED_ROOT,
    TEXT_MANIFEST,
    frozen_frame_ids,
    load_jsonl,
    sha256_file,
    sha256_text,
    utcnow,
    write_json,
    write_jsonl,
)
from full_resource_sentiment_v1.master_research_corrections_2026_11_03.textblob_checks import (
    independent_stratified_mean_diff,
)


def _tb(text: str) -> float | None:
    from textblob import TextBlob

    if not (text or "").strip():
        return None
    return float(TextBlob(text).sentiment.polarity)


def _sign_agree(v: float, t: float, v_thr: float = 0.0, t_thr: float = 0.0) -> bool:
    if abs(v) < v_thr and abs(t) < t_thr and (v_thr > 0 or t_thr > 0):
        return True
    return int(np.sign(v)) == int(np.sign(t))


def load_capture_index(frame_ids: set[str]) -> dict[tuple[int, str], list[dict[str, Any]]]:
    """Map (capture_page, reddit_id) → comments in original file order."""
    out: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for path in sorted(CAPTURE_DIR.glob("page_*.json")):
        page = int(path.stem.split("_")[1])
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("data") or []:
            rid = str(row.get("redditId") or "")
            if rid not in frame_ids:
                continue
            comments = [c for c in (row.get("comments") or []) if isinstance(c, dict)]
            out[(page, rid)] = comments
    return out


def join_score_row(
    score: dict[str, Any],
    comments: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Exact join by original_index. No positional/role-order fallback."""
    rid = str(score["reddit_id"])
    page = int(score["capture_page"])
    oi = int(score["original_index"])
    adapter_role = str(score.get("role") or "")
    status = "ok"
    comment_id = None
    body = ""
    text_field = "absent"
    is_submitter = None
    body_hash = None
    role_flag_disagree = False

    if comments is None:
        status = "missing_capture"
    elif oi < 0 or oi >= len(comments):
        status = "index_out_of_range"
    else:
        raw = comments[oi]
        comment_id = str(raw.get("id") or "") or None
        body, text_field = extract_comment_text(raw)
        body_hash = sha256_text(body) if body is not None else None
        is_submitter = str(raw.get("is_submitter")).lower() in {"true", "1"}
        if adapter_role == "author" and not is_submitter:
            role_flag_disagree = True
        elif adapter_role == "community" and is_submitter:
            role_flag_disagree = True
        if not (body or "").strip():
            status = "blank_body"
        # Duplicate index detection left to caller aggregation

    return {
        "reddit_id": rid,
        "capture_page": page,
        "original_index": oi,
        "chronological_index": score.get("chronological_index"),
        "score_comment_key": score.get("comment_key"),
        "adapter_role": adapter_role,
        "author_role_basis": score.get("author_role_basis"),
        "is_author_adapter": score.get("is_author"),
        "capture_is_submitter": is_submitter,
        "role_flag_disagree": role_flag_disagree,
        "comment_id": comment_id,
        "text_field": text_field,
        "text_hash": body_hash,
        "text_chars": len(body or ""),
        "join_status": status,
        "vader_compound": float(score["compound"]) if score.get("compound") is not None else None,
        "timestamp_epoch_seconds": score.get("timestamp_epoch_seconds"),
        "timestamp_utc": score.get("timestamp_utc"),
        "source_comment_scores_sha256": None,  # filled once per batch
        "_body_for_scoring": body if status in {"ok", "blank_body"} else "",
    }


def audit_order_vs_exact(frame_ids: set[str], cap_index: dict[tuple[int, str], list[dict[str, Any]]]) -> dict[str, Any]:
    """Reproduce Codex mismatch counts for documentation."""
    scores_by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in load_jsonl(COMMENT_SCORES):
        if str(c["reddit_id"]) in frame_ids:
            scores_by[str(c["reddit_id"])].append(c)

    order_candidates = mismatches = no_cand = 0
    for rid, scored in scores_by.items():
        pages = {int(c["capture_page"]) for c in scored if c.get("capture_page") is not None}
        comments = None
        for p in pages:
            if (p, rid) in cap_index:
                comments = cap_index[(p, rid)]
                break
        if comments is None:
            continue
        cap_author = sorted(
            [c for c in comments if str(c.get("is_submitter")).lower() in {"true", "1"}],
            key=lambda c: (c.get("created_utc_ts") in (None, ""), int(c.get("created_utc_ts") or 0)),
        )
        cap_comm = sorted(
            [c for c in comments if str(c.get("is_submitter")).lower() not in {"true", "1"}],
            key=lambda c: (c.get("created_utc_ts") in (None, ""), int(c.get("created_utc_ts") or 0)),
        )
        scored_s = sorted(
            scored,
            key=lambda c: (
                c.get("timestamp_epoch_seconds") is None,
                c.get("timestamp_epoch_seconds") or 0,
                c.get("chronological_index") or 0,
            ),
        )
        ai = ci = 0
        for sc in scored_s:
            if sc.get("compound") is None:
                continue
            role = sc.get("role")
            oi = int(sc["original_index"])
            exact = comments[oi] if 0 <= oi < len(comments) else None
            if role == "author":
                order = cap_author[ai] if ai < len(cap_author) else None
                if order is not None:
                    ai += 1
            elif role == "community":
                order = cap_comm[ci] if ci < len(cap_comm) else None
                if order is not None:
                    ci += 1
            else:
                order = None
            if order is None:
                no_cand += 1
                continue
            order_candidates += 1
            if exact is not None:
                eid = str(exact.get("id") or "")
                oid = str(order.get("id") or "")
                if eid and oid and eid != oid:
                    mismatches += 1
    return {
        "order_join_candidates": order_candidates,
        "order_join_mismatched_ids": mismatches,
        "order_join_no_candidate": no_cand,
        "expected_codex": {"candidates": 10983, "mismatches": 44, "no_candidate": 2},
        "matches_codex_audit": order_candidates == 10983 and mismatches == 44 and no_cand == 2,
    }


def run_exact_join_and_sentiment() -> dict[str, Any]:
    dest = OUT_ROOT / "phase1_exact_join"
    dest.mkdir(parents=True, exist_ok=True)
    frame_ids = frozen_frame_ids()
    cap_index = load_capture_index(frame_ids)
    score_sha = sha256_file(COMMENT_SCORES)
    texts = {str(r["reddit_id"]): r for r in load_jsonl(TEXT_MANIFEST)}
    labels = {
        str(r["reddit_id"]): r["final_discussion"]
        for r in load_jsonl(DATASET_JSONL)
        if str(r["reddit_id"]) in frame_ids
    }

    order_audit = audit_order_vs_exact(frame_ids, cap_index)

    matched_units: list[dict[str, Any]] = []
    analysis_units: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    role_disagree_rows: list[dict[str, Any]] = []
    dup_index_keys: Counter[str] = Counter()

    # --- Posts (unaffected join; recompute TextBlob for full-cohort contrasts) ---
    for rid in sorted(frame_ids):
        doc = texts.get(rid) or {"title": "", "body": ""}
        title = doc.get("title") or ""
        body = doc.get("body") or ""
        post = f"{title}\n{body}".strip()
        # pull vader from dataset
        # batched below via dataset load — fill in second pass
        analysis_units.append(
            {
                "reddit_id": rid,
                "label": labels[rid],
                "unit": "original_post",
                "title_chars": len(title),
                "body_chars": len(body),
                "text_chars": len(post),
                "text_hash": sha256_text(post),
                "join_status": "post_manifest",
                "join_affected": False,
            }
        )

    # Attach post VADER + TextBlob
    post_sent = {}
    for r in load_jsonl(DATASET_JSONL):
        rid = str(r["reddit_id"])
        if rid not in frame_ids:
            continue
        sent = r.get("sentiment") or {}
        post_sent[rid] = sent.get("combined_compound")
    for u in analysis_units:
        if u["unit"] != "original_post":
            continue
        rid = u["reddit_id"]
        doc = texts.get(rid) or {"title": "", "body": ""}
        post = f"{doc.get('title') or ''}\n{doc.get('body') or ''}".strip()
        tb = _tb(post)
        vader = post_sent.get(rid)
        u["vader"] = float(vader) if vader is not None else None
        u["textblob"] = tb
        u["sign_agree"] = (
            bool(_sign_agree(float(vader), float(tb))) if vader is not None and tb is not None else None
        )

    # Title / body separate TextBlob (post-level segments)
    for rid in sorted(frame_ids):
        doc = texts.get(rid) or {"title": "", "body": ""}
        title = (doc.get("title") or "").strip()
        body = (doc.get("body") or "").strip()
        for unit, text in (("title_only", title), ("body_only", body)):
            if not text:
                continue
            analysis_units.append(
                {
                    "reddit_id": rid,
                    "label": labels[rid],
                    "unit": unit,
                    "vader": None,  # no frozen VADER title/body split in package scores
                    "textblob": _tb(text),
                    "text_chars": len(text),
                    "text_hash": sha256_text(text),
                    "join_status": "post_manifest",
                    "join_affected": False,
                }
            )

    # --- Reply exact joins ---
    scores = [c for c in load_jsonl(COMMENT_SCORES) if str(c["reddit_id"]) in frame_ids]
    for sc in scores:
        key = f"{sc['reddit_id']}:{sc['capture_page']}:{sc['original_index']}"
        dup_index_keys[key] += 1
        page = int(sc["capture_page"])
        rid = str(sc["reddit_id"])
        comments = cap_index.get((page, rid))
        joined = join_score_row(sc, comments)
        joined["source_comment_scores_sha256"] = score_sha
        status_counts[joined["join_status"]] += 1
        if joined["role_flag_disagree"]:
            role_disagree_rows.append(
                {
                    "reddit_id": rid,
                    "capture_page": page,
                    "original_index": joined["original_index"],
                    "adapter_role": joined["adapter_role"],
                    "capture_is_submitter": joined["capture_is_submitter"],
                    "comment_id": joined["comment_id"],
                    "text_hash": joined["text_hash"],
                }
            )
        body = joined.pop("_body_for_scoring")
        tb = _tb(body) if joined["join_status"] == "ok" and body.strip() else None
        if joined["join_status"] == "blank_body":
            tb = None
        joined["textblob"] = tb
        joined["label"] = labels.get(rid)
        joined["sign_agree_vader_textblob"] = (
            bool(_sign_agree(float(joined["vader_compound"]), float(tb)))
            if joined["vader_compound"] is not None and tb is not None
            else None
        )
        # Public-safe matched unit (no full text, no username)
        public_unit = {k: v for k, v in joined.items() if k != "_body_for_scoring"}
        matched_units.append(public_unit)

    # Duplicate index audit
    n_dup_keys = sum(1 for k, n in dup_index_keys.items() if n > 1)

    # Chronological author/community analysis units from exact join
    by_rid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for u in matched_units:
        by_rid[u["reddit_id"]].append(u)

    for rid, units in by_rid.items():
        author = sorted(
            [u for u in units if u["adapter_role"] == "author" and u.get("vader_compound") is not None],
            key=lambda u: (
                u.get("timestamp_epoch_seconds") is None,
                u.get("timestamp_epoch_seconds") or 0,
                u.get("chronological_index") or 0,
            ),
        )
        community = sorted(
            [u for u in units if u["adapter_role"] == "community" and u.get("vader_compound") is not None],
            key=lambda u: (
                u.get("timestamp_epoch_seconds") is None,
                u.get("timestamp_epoch_seconds") or 0,
                u.get("chronological_index") or 0,
            ),
        )
        for i, u in enumerate(author):
            if i == 0:
                uname = "first_author_reply"
            elif i == len(author) - 1 and len(author) > 1:
                uname = "later_author_reply"
            else:
                uname = "author_reply_mid"
            if uname in {"first_author_reply", "later_author_reply"} or i == 1:
                unit_name = "second_author_reply" if i == 1 else uname
                analysis_units.append(
                    {
                        "reddit_id": rid,
                        "label": labels[rid],
                        "unit": unit_name,
                        "vader": u["vader_compound"],
                        "textblob": u["textblob"],
                        "text_chars": u["text_chars"],
                        "text_hash": u["text_hash"],
                        "join_status": u["join_status"],
                        "join_affected": True,
                        "comment_id": u["comment_id"],
                        "original_index": u["original_index"],
                        "sign_agree": u["sign_agree_vader_textblob"],
                    }
                )
        if community:
            u0 = community[0]
            analysis_units.append(
                {
                    "reddit_id": rid,
                    "label": labels[rid],
                    "unit": "first_community_reply",
                    "vader": u0["vader_compound"],
                    "textblob": u0["textblob"],
                    "text_chars": u0["text_chars"],
                    "text_hash": u0["text_hash"],
                    "join_status": u0["join_status"],
                    "join_affected": True,
                    "comment_id": u0["comment_id"],
                    "original_index": u0["original_index"],
                    "sign_agree": u0["sign_agree_vader_textblob"],
                }
            )

    write_jsonl(RESTRICTED_ROOT / "matched_units_exact_join.jsonl", matched_units)
    write_jsonl(RESTRICTED_ROOT / "analysis_units_exact_join.jsonl", analysis_units)
    write_jsonl(RESTRICTED_ROOT / "role_flag_disagreements.jsonl", role_disagree_rows)

    def contrast(unit: str, field: str) -> dict[str, Any]:
        g = [
            float(u[field])
            for u in analysis_units
            if u["unit"] == unit and u["label"] == "GDS" and u.get(field) is not None
        ]
        t = [
            float(u[field])
            for u in analysis_units
            if u["unit"] == unit and u["label"] == "TPS" and u.get(field) is not None
        ]
        return independent_stratified_mean_diff(g, t, seed=SEED + 60, n_boot=N_BOOT)

    contrasts = {
        "post_vader": contrast("original_post", "vader"),
        "post_textblob": contrast("original_post", "textblob"),
        "title_textblob": contrast("title_only", "textblob"),
        "body_textblob": contrast("body_only", "textblob"),
        "first_author_vader": contrast("first_author_reply", "vader"),
        "first_author_textblob": contrast("first_author_reply", "textblob"),
        "second_author_vader": contrast("second_author_reply", "vader"),
        "second_author_textblob": contrast("second_author_reply", "textblob"),
        "later_author_vader": contrast("later_author_reply", "vader"),
        "later_author_textblob": contrast("later_author_reply", "textblob"),
        "first_community_vader": contrast("first_community_reply", "vader"),
        "first_community_textblob": contrast("first_community_reply", "textblob"),
    }

    # Delta vs prior role-order TextBlob contrasts
    prior = json.loads(PRIOR_MATCHED.read_text(encoding="utf-8")) if PRIOR_MATCHED.exists() else {}
    prior_c = prior.get("contrasts_gds_minus_tps") or {}
    deltas = {}
    for key in (
        "first_author_textblob",
        "first_community_textblob",
        "later_author_vader",
        "first_author_vader",
        "post_textblob",
        "post_vader",
    ):
        old = (prior_c.get(key) or {}).get("mean_diff")
        new = (contrasts.get(key) or {}).get("mean_diff")
        if old is not None and new is not None:
            deltas[key] = {
                "prior_role_order_mean_diff": old,
                "exact_join_mean_diff": new,
                "delta_exact_minus_prior": float(new) - float(old),
                "prior_n_a": (prior_c.get(key) or {}).get("n_a"),
                "exact_n_a": (contrasts.get(key) or {}).get("n_a"),
                "prior_n_b": (prior_c.get(key) or {}).get("n_b"),
                "exact_n_b": (contrasts.get(key) or {}).get("n_b"),
            }

    # Reply-level sign agreement (exact join TextBlob)
    reply_agree = sum(
        1
        for u in matched_units
        if u.get("sign_agree_vader_textblob") is True and u.get("join_status") == "ok"
    )
    reply_disagree = sum(
        1
        for u in matched_units
        if u.get("sign_agree_vader_textblob") is False and u.get("join_status") == "ok"
    )
    post_units = [u for u in analysis_units if u["unit"] == "original_post"]
    post_agree = sum(1 for u in post_units if u.get("sign_agree") is True)
    post_disagree = sum(1 for u in post_units if u.get("sign_agree") is False)

    unit_counts = Counter(u["unit"] for u in analysis_units)
    report = {
        "utc": utcnow(),
        "n_frame_posts": len(frame_ids),
        "n_score_rows_joined": len(matched_units),
        "join_status_counts": dict(status_counts),
        "n_role_flag_disagreements": len(role_disagree_rows),
        "role_flag_disagreement_note": (
            "Retain adapter role; do not overwrite from is_submitter. "
            "Expected ~2 disagreements per Codex audit."
        ),
        "n_duplicate_page_rid_index_keys": n_dup_keys,
        "order_vs_exact_audit": order_audit,
        "unit_counts": dict(unit_counts),
        "contrasts_gds_minus_tps": contrasts,
        "delta_vs_prior_role_order_join": deltas,
        "post_sign_agree": {"agree": post_agree, "disagree": post_disagree, "n": len(post_units)},
        "reply_sign_agree_exact_join": {
            "agree": reply_agree,
            "disagree": reply_disagree,
            "n_ok_with_both_scores": reply_agree + reply_disagree,
        },
        "original_post_join_unaffected": True,
        "no_positional_fallback": True,
        "scale_warning": "Do not equate VADER compound with TextBlob polarity",
        "restricted_outputs": [
            "matched_units_exact_join.jsonl",
            "analysis_units_exact_join.jsonl",
            "role_flag_disagreements.jsonl",
        ],
    }
    write_json(dest / "EXACT_JOIN_SENTIMENT.json", report)
    write_json(dest / "JOIN_AUDIT_SUMMARY.json", {
        "utc": utcnow(),
        "order_vs_exact": order_audit,
        "status_counts": dict(status_counts),
        "role_flag_disagreements": len(role_disagree_rows),
        "duplicate_keys": n_dup_keys,
        "comment_scores_sha256": score_sha,
    })
    return report
