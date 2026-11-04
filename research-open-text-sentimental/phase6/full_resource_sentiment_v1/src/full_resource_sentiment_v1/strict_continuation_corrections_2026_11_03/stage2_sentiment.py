"""Stage 2: matched post/reply TextBlob+VADER; Cardiff & SentiStrength-SE proposals."""

from __future__ import annotations

import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from full_resource_sentiment_v1.master_research_corrections_2026_11_03.textblob_checks import (
    independent_stratified_mean_diff,
)
from full_resource_sentiment_v1.strict_continuation_corrections_2026_11_03 import SEED
from full_resource_sentiment_v1.strict_continuation_corrections_2026_11_03.common import (
    CAPTURE_DIR,
    COMMENT_SCORES,
    DATASET_JSONL,
    FROZEN_DEF,
    OUT_ROOT,
    RESTRICTED_ROOT,
    TEXT_MANIFEST,
    load_jsonl,
    utcnow,
    write_json,
    write_jsonl,
)
from full_resource_sentiment_v1.strict_continuation_corrections_2026_11_03.stage0_repairs import _sign_agree


def _tb(text: str) -> float | None:
    from textblob import TextBlob

    if not (text or "").strip():
        return None
    return float(TextBlob(text).sentiment.polarity)


def matched_role_sentiment() -> dict[str, Any]:
    dest = OUT_ROOT / "stage2_matched_role_sentiment"
    dest.mkdir(parents=True, exist_ok=True)
    texts = {str(r["reddit_id"]): r for r in load_jsonl(TEXT_MANIFEST)}
    # Build capture comments indexed by reddit_id sorted by time
    capture_comments: dict[str, list[dict[str, Any]]] = defaultdict(list)
    frame_ids = {
        str(r["reddit_id"])
        for r in load_jsonl(DATASET_JSONL)
        if r.get("final_relevance") == "relevant"
        and r.get("final_discussion") in {"TPS", "GDS"}
        and r.get("sentiment_definition") == FROZEN_DEF
    }
    for path in sorted(CAPTURE_DIR.glob("page_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("data") or []:
            rid = str(row.get("redditId") or "")
            if rid not in frame_ids:
                continue
            for c in row.get("comments") or []:
                if isinstance(c, dict):
                    capture_comments[rid].append(c)
    for rid in capture_comments:
        capture_comments[rid].sort(
            key=lambda c: (c.get("created_utc_ts") in (None, ""), int(c.get("created_utc_ts") or 0))
        )

    scores_by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in load_jsonl(COMMENT_SCORES):
        if str(c["reddit_id"]) in frame_ids:
            scores_by[str(c["reddit_id"])].append(c)

    # Count all matched text units for Cardiff proposal
    unit_counts = Counter = __import__("collections").Counter
    uc = unit_counts()
    token_lens = []

    units = []
    # Full frozen cohort posts + replies (TextBlob can be heavy; do all posts, sample replies if needed)
    # User asked matched texts — do all frozen TPS/GDS posts; all scoreable author/community with joinable text
    for r in load_jsonl(DATASET_JSONL):
        rid = str(r["reddit_id"])
        if rid not in frame_ids:
            continue
        label = r["final_discussion"]
        doc = texts.get(rid) or {"title": "", "body": ""}
        post = f"{doc.get('title') or ''}\n{doc.get('body') or ''}".strip()
        sent = r.get("sentiment") or {}
        vader_post = sent.get("combined_compound")
        tb_post = _tb(post)
        uc["original_post"] += 1
        token_lens.append(len(post.split()))
        units.append(
            {
                "reddit_id": rid,
                "label": label,
                "unit": "original_post",
                "vader": vader_post,
                "textblob": tb_post,
                "text_chars": len(post),
                "sign_agree": bool(_sign_agree(float(vader_post), float(tb_post)))
                if vader_post is not None and tb_post is not None
                else None,
            }
        )
        # Join capture comments to score rows by chronological order among same role
        cap = capture_comments.get(rid, [])
        scored = sorted(
            scores_by.get(rid, []),
            key=lambda c: (
                c.get("timestamp_epoch_seconds") is None,
                c.get("timestamp_epoch_seconds") or 0,
                c.get("chronological_index") or 0,
            ),
        )
        # Map scoreable author/community in order to capture comments with matching is_submitter
        cap_author = [c for c in cap if str(c.get("is_submitter")).lower() in {"true", "1"}]
        cap_comm = [c for c in cap if str(c.get("is_submitter")).lower() not in {"true", "1"}]
        ai = ci = 0
        author_scored = [c for c in scored if c.get("role") == "author" and c.get("compound") is not None]
        comm_scored = [c for c in scored if c.get("role") == "community" and c.get("compound") is not None]
        for i, sc in enumerate(author_scored):
            body = ""
            if i < len(cap_author):
                body = str(cap_author[i].get("body") or "")
            unit_name = "first_author_reply" if i == 0 else ("later_author_reply" if i == len(author_scored) - 1 and i > 0 else f"author_reply_{i}")
            if i == 0:
                unit_name = "first_author_reply"
            elif i == len(author_scored) - 1 and len(author_scored) > 1:
                unit_name = "later_author_reply"
            else:
                unit_name = "author_reply_mid"
            tb = _tb(body) if body.strip() else None
            uc[unit_name if unit_name in {"first_author_reply", "later_author_reply"} else "author_reply_other"] += 1
            if body:
                token_lens.append(len(body.split()))
            if unit_name in {"first_author_reply", "later_author_reply"} or i < 3:
                units.append(
                    {
                        "reddit_id": rid,
                        "label": label,
                        "unit": unit_name,
                        "vader": float(sc["compound"]),
                        "textblob": tb,
                        "text_chars": len(body),
                        "join_ok": bool(body.strip()),
                        "sign_agree": bool(_sign_agree(float(sc["compound"]), float(tb)))
                        if tb is not None
                        else None,
                    }
                )
        for i, sc in enumerate(comm_scored[:2]):  # first community (+ optional second)
            body = str(cap_comm[i].get("body") or "") if i < len(cap_comm) else ""
            tb = _tb(body) if body.strip() else None
            uc["first_community_reply" if i == 0 else "community_reply_other"] += 1
            if body:
                token_lens.append(len(body.split()))
            if i == 0:
                units.append(
                    {
                        "reddit_id": rid,
                        "label": label,
                        "unit": "first_community_reply",
                        "vader": float(sc["compound"]),
                        "textblob": tb,
                        "text_chars": len(body),
                        "join_ok": bool(body.strip()),
                        "sign_agree": bool(_sign_agree(float(sc["compound"]), float(tb)))
                        if tb is not None
                        else None,
                    }
                )

    write_jsonl(RESTRICTED_ROOT / "matched_role_sentiment_units.jsonl", units)

    def contrast_for(unit: str, field: str):
        g = [float(u[field]) for u in units if u["unit"] == unit and u["label"] == "GDS" and u.get(field) is not None]
        t = [float(u[field]) for u in units if u["unit"] == unit and u["label"] == "TPS" and u.get(field) is not None]
        return independent_stratified_mean_diff(g, t, seed=SEED + 60)

    contrasts = {
        "post_vader": contrast_for("original_post", "vader"),
        "post_textblob": contrast_for("original_post", "textblob"),
        "first_author_vader": contrast_for("first_author_reply", "vader"),
        "first_author_textblob": contrast_for("first_author_reply", "textblob"),
        "later_author_vader": contrast_for("later_author_reply", "vader"),
        "first_community_vader": contrast_for("first_community_reply", "vader"),
        "first_community_textblob": contrast_for("first_community_reply", "textblob"),
    }
    post_units = [u for u in units if u["unit"] == "original_post"]
    agree = sum(1 for u in post_units if u["sign_agree"] is True)
    disagree = sum(1 for u in post_units if u["sign_agree"] is False)

    report = {
        "utc": utcnow(),
        "n_units_written": len(units),
        "unit_counts": dict(uc),
        "token_length_summary": {
            "n": len(token_lens),
            "mean": float(np.mean(token_lens)) if token_lens else None,
            "p50": float(np.median(token_lens)) if token_lens else None,
            "p90": float(np.percentile(token_lens, 90)) if token_lens else None,
            "max": int(max(token_lens)) if token_lens else None,
        },
        "full_cohort_posts": True,
        "prior_498_diagnostic_note": "Prior 498 = all 98 GDS author>=1 + 400/956 TPS; this run scores all frozen posts",
        "post_sign_agree": {"agree": agree, "disagree": disagree, "n": len(post_units)},
        "contrasts_gds_minus_tps": contrasts,
        "scale_warning": "Do not equate VADER compound with TextBlob polarity",
        "stage1_vader_preserved": True,
    }
    write_json(dest / "MATCHED_ROLE_SENTIMENT.json", report)
    return report


def cardiff_proposal(matched: dict[str, Any]) -> dict[str, Any]:
    dest = OUT_ROOT / "stage2_resource_proposals"
    dest.mkdir(parents=True, exist_ok=True)
    hub = Path.home() / ".cache/huggingface/hub/models--cardiffnlp--twitter-roberta-base-sentiment-latest"
    snaps = []
    for s in sorted((hub / "snapshots").iterdir()) if hub.exists() else []:
        files = sorted(p.name for p in s.rglob("*") if p.is_file())
        snaps.append({"rev": s.name, "files": files})
    n_posts = matched.get("unit_counts", {}).get("original_post", 3259)
    n_first_auth = matched.get("unit_counts", {}).get("first_author_reply", 0)
    n_later = matched.get("unit_counts", {}).get("later_author_reply", 0)
    n_comm = matched.get("unit_counts", {}).get("first_community_reply", 0)
    n_units = n_posts + n_first_auth + n_later + n_comm
    proposal = {
        "model": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "pinned_rev_preferred": "3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7",
        "cache_snapshots": snaps,
        "torch_installed": importlib.util.find_spec("torch") is not None,
        "transformers_installed": importlib.util.find_spec("transformers") is not None,
        "license_note": "Verify CardiffNLP model card license before use",
        "n_text_units": {
            "original_posts": n_posts,
            "first_author_replies": n_first_auth,
            "later_author_replies": n_later,
            "first_community_replies": n_comm,
            "total_matched_units_min": n_units,
            "token_length_summary": matched.get("token_length_summary"),
        },
        "plan": {
            "canary": {"n": 20, "seed": SEED, "est_minutes": 5},
            "bounded_matched": {"n_units": n_units, "est_minutes_cpu": "60–180"},
            "optional_full_eligible": "same as bounded if counts equal frozen eligible",
        },
        "disk_ram": {
            "torch_transformers_install_gb": "1–3",
            "inference_ram_gb": "2–4",
            "free_disk_check_required": True,
        },
        "money": 0,
        "privacy": "local restricted outputs only",
        "rollback": "delete venv + new result shards",
        "status": "AWAITING_APPROVAL — not executed",
    }
    write_json(dest / "CARDIFF_APPROVAL_PROPOSAL.json", proposal)
    return proposal


def sentistrength_se_feasibility() -> dict[str, Any]:
    dest = OUT_ROOT / "stage2_resource_proposals"
    dest.mkdir(parents=True, exist_ok=True)
    report = {
        "utc": utcnow(),
        "tool": "SentiStrength-SE (Islam & Zibran / UNO LASER)",
        "not_to_confuse_with": ["generic SentiStrength", "SentiSE (amiangshu/SentiSE, LGPL-3.0)"],
        "implementation": {
            "language": "Java CLI",
            "historical_availability": "GitLab path cited in paper (2017): gitlab.com/.../SentiStrengthSE Tool MSR — verify current URL",
            "paper_pdf": "https://laser.cs.uno.edu/resources/Publication/SentiStrength-SE.pdf",
            "local_copy_present": False,
        },
        "licensing": {
            "inherits_sentistrength": True,
            "sentistrength_base": "Commercial closed-source; free for academic research (typical terms — confirm before use)",
            "sentiSE_is_different": True,
        },
        "runtime_needs": ["Java JRE", "SentiStrength-SE jar", "lexicon/data files", "possibly base SentiStrength data"],
        "output_mapping": "Positive and negative strength integers; map to polarity via documented rule (sum/sign) — must pin mapping before analysis",
        "blocker": (
            "No verified local SentiStrength-SE package in workspace; GitLab availability/license confirmation "
            "and possible base SentiStrength academic license acceptance required before install."
        ),
        "defensible_alternatives_if_blocked": [
            "VADER (accepted Stage-1 baseline)",
            "TextBlob PatternAnalyzer (already scored)",
            "Cardiff RoBERTa after approval",
            "Senti4SD (MIT) as open SE-oriented alternative — separate feasibility if needed",
        ],
        "status": "FEASIBILITY_DOCUMENTED — not installed; do not call generic SentiStrength 'SentiStrength-SE'",
        "approval_gate": "Confirm license + obtain jar/lexicons before any run",
    }
    write_json(dest / "SENTISTRENGTH_SE_FEASIBILITY.json", report)
    return report


def run_stage2() -> dict[str, Any]:
    matched = matched_role_sentiment()
    cardiff = cardiff_proposal(matched)
    ssse = sentistrength_se_feasibility()
    return {"matched": matched, "cardiff": cardiff, "sentistrength_se": ssse}
