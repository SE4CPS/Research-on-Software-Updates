"""Rebuild the same post/reply texts used in the length-sensitivity tree."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from full_resource_sentiment_v1.comments import extract_comment_text
from full_resource_sentiment_v1.sentiment_length_sensitivity_2026_11_03.common import (
    AUDIT_RESTRICTED,
    CAPTURE_DIR,
    COMMENT_SCORES,
    CORR_RESTRICTED,
    DATASET_JSONL,
    EXACT_JOIN_RESTRICTED,
    FROZEN_DEF,
    TEXT_MANIFEST,
)
from full_resource_sentiment_v1.senti4sd_comparison_2026_11_03.common import (
    LENGTH_RESTRICTED,
    load_jsonl,
    sha256_text,
)


def _ai_post_texts(need: set[str]) -> dict[str, str]:
    post_text: dict[str, str] = {}
    inp = CORR_RESTRICTED / "cardiff_full_input_units.jsonl"
    if inp.exists():
        for r in load_jsonl(inp):
            if r.get("stratum") == "ai_assisted_v2" and r.get("unit") == "original_post":
                rid = str(r["reddit_id"])
                if rid in need:
                    post_text[rid] = r.get("text") or ""
    missing = {rid for rid in need if rid not in post_text or not (post_text.get(rid) or "").strip()}
    if missing:
        for path in sorted(CAPTURE_DIR.glob("page_*.json")):
            if not missing:
                break
            payload = json.loads(path.read_text(encoding="utf-8"))
            for row in payload.get("data") or []:
                rid = str(row.get("redditId") or "")
                if rid not in missing:
                    continue
                title = "" if row.get("title") is None else str(row.get("title"))
                body = "" if row.get("author_description") is None else str(row.get("author_description"))
                post_text[rid] = f"{title}\n{body}".strip()
                missing.discard(rid)
    return post_text


def build_ai_units() -> list[dict[str, Any]]:
    """Same 5,942 AI pairs as length-sensitivity; emit post + first-author reply units."""
    pair_meta = {str(r["reddit_id"]): r for r in load_jsonl(LENGTH_RESTRICTED / "ai_assisted_v2_pair_rows.jsonl")}
    threads = {
        r["reddit_id"]: r
        for r in load_jsonl(AUDIT_RESTRICTED / "ai_reply_thread_rows.jsonl")
        if r.get("first_author_minus_post_vader") is not None
    }
    reply_units = {
        u["reddit_id"]: u
        for u in load_jsonl(AUDIT_RESTRICTED / "ai_reply_units_for_cardiff.jsonl")
        if u.get("unit") == "first_author_reply"
    }
    assert len(threads) == 5942, f"expected 5942 AI threads, got {len(threads)}"
    post_text = _ai_post_texts(set(threads))

    units: list[dict[str, Any]] = []
    for rid, th in threads.items():
        ru = reply_units.get(rid)
        if not ru:
            continue
        post = post_text.get(rid) or ""
        reply = ru.get("text") or ""
        if not post.strip() or not reply.strip():
            continue
        meta = pair_meta.get(rid) or {}
        label = th["label"]
        for role, text in (("original_post", post), ("first_author_reply", reply)):
            units.append(
                {
                    "cohort": "ai_assisted_v2",
                    "reddit_id": rid,
                    "label": label,
                    "unit": role,
                    "text": text,
                    "text_chars": len(text),
                    "text_hash": sha256_text(text),
                    "source": meta.get("source"),
                    "date_band": meta.get("date_band"),
                    # Prior instrument scores on the same unit (full text)
                    "vader": meta.get("full_post_vader") if role == "original_post" else meta.get("full_reply_vader"),
                    "textblob": meta.get("full_post_textblob")
                    if role == "original_post"
                    else meta.get("full_reply_textblob"),
                    "cardiff_sign": meta.get("full_post_cardiff_sign")
                    if role == "original_post"
                    else meta.get("full_reply_cardiff_sign"),
                }
            )
    n_posts = sum(1 for u in units if u["unit"] == "original_post")
    assert n_posts == 5942, f"expected 5942 AI posts, got {n_posts}"
    return units


def build_frozen_units() -> list[dict[str, Any]]:
    pair_meta = {str(r["reddit_id"]): r for r in load_jsonl(LENGTH_RESTRICTED / "original_frozen_pair_rows.jsonl")}
    texts = {str(r["reddit_id"]): r for r in load_jsonl(TEXT_MANIFEST)}
    matched = {
        (m["reddit_id"], m.get("original_index")): m
        for m in load_jsonl(EXACT_JOIN_RESTRICTED / "matched_units_exact_join.jsonl")
    }
    analysis = load_jsonl(EXACT_JOIN_RESTRICTED / "analysis_units_exact_join.jsonl")
    first_author = {
        u["reddit_id"]: u for u in analysis if u.get("unit") == "first_author_reply" and u.get("label") in {"TPS", "GDS"}
    }
    need_ids = set(first_author)
    pages = {
        int(matched[(u["reddit_id"], u.get("original_index"))]["capture_page"])
        for u in first_author.values()
        if (u["reddit_id"], u.get("original_index")) in matched
    }
    cap: dict[str, list] = {}
    for page in pages:
        path = CAPTURE_DIR / f"page_{page:05d}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("data") or []:
            rid = str(row.get("redditId") or "")
            if rid in need_ids:
                cap[rid] = [c for c in (row.get("comments") or []) if isinstance(c, dict)]

    comments_by: dict[str, list] = defaultdict(list)
    for c in load_jsonl(COMMENT_SCORES):
        if str(c["reddit_id"]) in need_ids and c.get("role") == "author" and c.get("compound") is not None:
            comments_by[str(c["reddit_id"])].append(c)

    units: list[dict[str, Any]] = []
    for r in load_jsonl(DATASET_JSONL):
        if r.get("final_relevance") != "relevant" or r.get("final_discussion") not in {"TPS", "GDS"}:
            continue
        if r.get("sentiment_definition") != FROZEN_DEF:
            continue
        rid = str(r["reddit_id"])
        fa = first_author.get(rid)
        if not fa:
            continue
        doc = texts.get(rid) or {}
        post = f"{doc.get('title') or ''}\n{doc.get('body') or ''}".strip()
        m = matched.get((rid, fa.get("original_index")))
        reply = ""
        if m and rid in cap:
            oi = int(m["original_index"])
            if 0 <= oi < len(cap[rid]):
                reply, _ = extract_comment_text(cap[rid][oi])
                reply = reply or ""
        if not post.strip() or not reply.strip():
            continue
        auth = sorted(
            comments_by.get(rid, []),
            key=lambda c: (
                c.get("timestamp_epoch_seconds") is None,
                c.get("timestamp_epoch_seconds") or 0,
                c.get("chronological_index") or 0,
            ),
        )
        if not auth:
            continue
        meta = pair_meta.get(rid) or {}
        label = r["final_discussion"]
        for role, text in (("original_post", post), ("first_author_reply", reply)):
            units.append(
                {
                    "cohort": "original_frozen",
                    "reddit_id": rid,
                    "label": label,
                    "unit": role,
                    "text": text,
                    "text_chars": len(text),
                    "text_hash": sha256_text(text),
                    "source": meta.get("source"),
                    "date_band": meta.get("date_band"),
                    "vader": meta.get("full_post_vader") if role == "original_post" else meta.get("full_reply_vader"),
                    "textblob": meta.get("full_post_textblob")
                    if role == "original_post"
                    else meta.get("full_reply_textblob"),
                    "cardiff_sign": meta.get("full_post_cardiff_sign")
                    if role == "original_post"
                    else meta.get("full_reply_cardiff_sign"),
                }
            )
    n_posts = sum(1 for u in units if u["unit"] == "original_post")
    expected = len(pair_meta)
    assert n_posts == expected, f"expected {expected} frozen posts, got {n_posts}"
    return units
