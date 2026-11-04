"""Item 3: AI-cohort reply scoring via exact join + RQ2/RQ3 estimands (separate strata)."""

from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from full_resource_sentiment_v1.audit_corrections_2026_11_03 import (
    CARDIFF_MODEL,
    CARDIFF_REV,
    N_BOOT,
    SEED,
)
from full_resource_sentiment_v1.audit_corrections_2026_11_03.common import (
    CAPTURE_DIR,
    CARDIFF_SCORES,
    COMMENT_SCORES,
    DATASET_JSONL,
    FROZEN_DEF,
    OUT_ROOT,
    RESTRICTED_ROOT,
    UNION_V2,
    VENV_CARDIFF,
    load_jsonl,
    sha256_file,
    sha256_text,
    utcnow,
    write_json,
    write_jsonl,
)
from full_resource_sentiment_v1.comments import extract_comment_text
from full_resource_sentiment_v1.master_research_corrections_2026_11_03.textblob_checks import (
    independent_stratified_mean_diff,
)
from full_resource_sentiment_v1.scoring import early_late_or_none


def _tb(text: str) -> float | None:
    from textblob import TextBlob

    if not (text or "").strip():
        return None
    return float(TextBlob(text).sentiment.polarity)


def run_expanded_conversation() -> dict[str, Any]:
    dest = OUT_ROOT / "item3_expanded_conversation"
    dest.mkdir(parents=True, exist_ok=True)

    ai_meta = {
        r["reddit_id"]: r
        for r in load_jsonl(UNION_V2)
        if r.get("record_origin") == "ai_assisted_screen_candidate_v2"
        and r.get("final_discussion") in {"TPS", "GDS"}
    }
    ai_ids = set(ai_meta)
    labels = {rid: ai_meta[rid]["final_discussion"] for rid in ai_ids}

    # Exact-join score rows for AI IDs
    scores_by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in load_jsonl(COMMENT_SCORES):
        rid = str(c["reddit_id"])
        if rid in ai_ids:
            scores_by[rid].append(c)

    # Load capture comments for AI IDs that have scores
    need = set(scores_by)
    cap_index: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for path in sorted(CAPTURE_DIR.glob("page_*.json")):
        page = int(path.stem.split("_")[1])
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("data") or []:
            rid = str(row.get("redditId") or "")
            if rid not in need:
                continue
            cap_index[(page, rid)] = [c for c in (row.get("comments") or []) if isinstance(c, dict)]

    missing = {"no_comment_scores": 0, "index_miss": 0, "blank_body": 0, "ok": 0}
    units = []  # for TextBlob + later Cardiff
    thread_rows = []

    for rid in sorted(ai_ids):
        lab = labels[rid]
        scored = sorted(
            scores_by.get(rid, []),
            key=lambda c: (
                c.get("timestamp_epoch_seconds") is None,
                c.get("timestamp_epoch_seconds") or 0,
                c.get("chronological_index") or 0,
            ),
        )
        if not scored:
            missing["no_comment_scores"] += 1
        author_vals = []
        author_units = []
        community_vals = []
        first_comm_unit = None
        for sc in scored:
            if sc.get("compound") is None:
                continue
            role = sc.get("role")
            if role == "author":
                author_vals.append(float(sc["compound"]))
                # Exact-join body only for first two author replies (TextBlob + Cardiff)
                if len(author_units) < 2:
                    page = int(sc["capture_page"])
                    oi = int(sc["original_index"])
                    comments = cap_index.get((page, rid))
                    body = ""
                    status = "ok"
                    if comments is None or oi < 0 or oi >= len(comments):
                        status = "index_miss"
                        missing["index_miss"] += 1
                    else:
                        body, _ = extract_comment_text(comments[oi])
                        body = body or ""
                        if not body.strip():
                            status = "blank_body"
                            missing["blank_body"] += 1
                        else:
                            missing["ok"] += 1
                    tb = _tb(body) if status == "ok" else None
                    author_units.append(
                        {
                            "reddit_id": rid,
                            "label": lab,
                            "role": "author",
                            "original_index": oi,
                            "capture_page": page,
                            "chronological_index": sc.get("chronological_index"),
                            "timestamp_epoch_seconds": sc.get("timestamp_epoch_seconds"),
                            "vader": float(sc["compound"]),
                            "textblob": tb,
                            "join_status": status,
                            "text_chars": len(body),
                            "text_hash": sha256_text(body) if body else None,
                            "text": body if status == "ok" else "",
                            "unit": "first_author_reply" if len(author_units) == 0 else "second_author_reply",
                        }
                    )
            elif role == "community":
                community_vals.append(float(sc["compound"]))
                if first_comm_unit is None:
                    page = int(sc["capture_page"])
                    oi = int(sc["original_index"])
                    comments = cap_index.get((page, rid))
                    body = ""
                    status = "ok"
                    if comments is None or oi < 0 or oi >= len(comments):
                        status = "index_miss"
                        missing["index_miss"] += 1
                    else:
                        body, _ = extract_comment_text(comments[oi])
                        body = body or ""
                        if not body.strip():
                            status = "blank_body"
                            missing["blank_body"] += 1
                        else:
                            missing["ok"] += 1
                    tb = _tb(body) if status == "ok" else None
                    first_comm_unit = {
                        "reddit_id": rid,
                        "label": lab,
                        "role": "community",
                        "original_index": oi,
                        "capture_page": page,
                        "chronological_index": sc.get("chronological_index"),
                        "timestamp_epoch_seconds": sc.get("timestamp_epoch_seconds"),
                        "vader": float(sc["compound"]),
                        "textblob": tb,
                        "join_status": status,
                        "text_chars": len(body),
                        "text_hash": sha256_text(body) if body else None,
                        "text": body if status == "ok" else "",
                        "unit": "first_community_reply",
                    }

        units.extend(author_units)
        if first_comm_unit is not None:
            units.append(first_comm_unit)
        author_tb = [u.get("textblob") for u in author_units]

        thread_rows.append(
            {
                "reddit_id": rid,
                "label": lab,
                "stratum": "ai_assisted_v2",
                "n_author": len(author_vals),
                "n_community": len(community_vals),
                "first_author_minus_post_vader": None,  # filled after post vader
                "second_minus_first_vader": (author_vals[1] - author_vals[0]) if len(author_vals) >= 2 else None,
                "author_shift_ge4_vader": early_late_or_none(author_vals) if len(author_vals) >= 4 else None,
                "author_minus_community_mean_vader": (
                    float(np.mean(author_vals)) - float(np.mean(community_vals))
                    if author_vals and community_vals
                    else None
                ),
                "first_author_vader": author_vals[0] if author_vals else None,
                "second_author_vader": author_vals[1] if len(author_vals) >= 2 else None,
                "first_author_textblob": author_tb[0] if author_tb and author_tb[0] is not None else None,
                "second_author_textblob": author_tb[1] if len(author_tb) >= 2 and author_tb[1] is not None else None,
                "missing_first_author_reply": len(author_vals) == 0,
                "missing_second_author_reply": len(author_vals) < 2,
                "missing_community": len(community_vals) == 0,
            }
        )

    # Attach post VADER from existing Cardiff AI post scores
    post_by = {
        r["reddit_id"]: r
        for r in load_jsonl(CARDIFF_SCORES)
        if r.get("stratum") == "ai_assisted_v2" and r.get("unit") == "original_post"
    }
    for row in thread_rows:
        p = post_by.get(row["reddit_id"])
        if p and p.get("vader") is not None and row.get("first_author_vader") is not None:
            row["post_vader"] = float(p["vader"])
            row["first_author_minus_post_vader"] = float(row["first_author_vader"]) - float(p["vader"])
        else:
            row["post_vader"] = float(p["vader"]) if p and p.get("vader") is not None else None

    write_jsonl(RESTRICTED_ROOT / "ai_reply_thread_rows.jsonl", thread_rows)
    # Persist reply units for Cardiff (text included restricted)
    write_jsonl(RESTRICTED_ROOT / "ai_reply_units_for_cardiff.jsonl", units)

    # Cardiff on AI reply units
    cardiff_status = {"status": "SKIPPED", "n": 0}
    if VENV_CARDIFF.exists() and units:
        units_path = RESTRICTED_ROOT / "ai_reply_units_for_cardiff.jsonl"
        out_path = RESTRICTED_ROOT / "ai_reply_cardiff_scores.jsonl"
        meta_path = dest / "AI_REPLY_CARDIFF_META.json"
        script = dest / "_infer_ai_replies.py"
        script.write_text(
            f"""#!/usr/bin/env python3
import json, time, resource
from pathlib import Path
MODEL="{CARDIFF_MODEL}"; REV="{CARDIFF_REV}"
units=[json.loads(l) for l in open(r"{units_path}") if l.strip()]
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
t0=time.time()
tok=AutoTokenizer.from_pretrained(MODEL, revision=REV, local_files_only=True)
model=AutoModelForSequenceClassification.from_pretrained(MODEL, revision=REV, local_files_only=True)
model.eval(); load_s=time.time()-t0
id2label={{int(k):v for k,v in model.config.id2label.items()}}
rows=[]; n_trunc=0; max_rss=0
for i,u in enumerate(units):
    text=u.get("text") or ""
    full=len(tok(text, truncation=False)["input_ids"]) if text else 0
    enc=tok(text, return_tensors="pt", truncation=True, max_length=512)
    trunc=full>512
    if trunc: n_trunc+=1
    with torch.no_grad():
        logits=model(**enc).logits[0]
        probs=torch.softmax(logits, dim=-1).tolist()
    pred=int(logits.argmax()); label=id2label.get(pred, str(pred))
    name=str(label).lower(); pol=-1.0 if "neg" in name else (1.0 if "pos" in name else 0.0)
    try: max_rss=max(max_rss, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception: pass
    rows.append({{k:u.get(k) for k in ("reddit_id","label","unit","vader","textblob","text_hash","text_chars","join_status")}} | {{
        "cardiff_label": str(label), "cardiff_polarity_sign": pol, "cardiff_probs": probs,
        "n_tokens": int(enc["input_ids"].shape[1]), "full_token_len": full, "truncated": trunc,
    }})
    if (i+1)%1000==0: print(json.dumps({{"progress":i+1}}), flush=True)
Path(r"{out_path}").write_text("\\n".join(json.dumps(r,sort_keys=True) for r in rows)+"\\n")
Path(r"{meta_path}").write_text(json.dumps({{"n":len(rows),"n_truncated":n_trunc,"load_s":load_s,"max_rss":max_rss}},indent=2))
print(json.dumps({{"ok":True,"n":len(rows),"trunc":n_trunc}}))
""",
            encoding="utf-8",
        )
        print(f"  Cardiff on {len(units)} AI reply units...", flush=True)
        proc = subprocess.run(
            [str(VENV_CARDIFF / "bin/python"), str(script)],
            capture_output=True,
            text=True,
            timeout=7200,
        )
        cardiff_status = {
            "status": "COMPLETE" if proc.returncode == 0 and out_path.exists() else "FAILED",
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-1000:],
            "stderr_tail": proc.stderr[-1500:],
            "n_units": len(units),
            "sha256": sha256_file(out_path) if out_path.exists() else None,
        }
        write_json(dest / "AI_REPLY_CARDIFF_RUN.json", cardiff_status)

    def contrast(rows, key, seed):
        g = [float(r[key]) for r in rows if r["label"] == "GDS" and r.get(key) is not None]
        t = [float(r[key]) for r in rows if r["label"] == "TPS" and r.get(key) is not None]
        return independent_stratified_mean_diff(g, t, seed=seed, n_boot=N_BOOT)

    ge1 = [r for r in thread_rows if r["n_author"] >= 1]
    ge2 = [r for r in thread_rows if r["n_author"] >= 2]
    ge4 = [r for r in thread_rows if r["n_author"] >= 4]
    both = [r for r in thread_rows if r["n_author"] >= 1 and r["n_community"] >= 1]

    # Original frozen RQ2/RQ3 for side-by-side (recomputed with corrected cliffs)
    comments_by: dict[str, list] = defaultdict(list)
    for c in load_jsonl(COMMENT_SCORES):
        comments_by[str(c["reddit_id"])].append(c)
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
        author = [float(c["compound"]) for c in clist if c.get("role") == "author" and c.get("compound") is not None]
        community = [float(c["compound"]) for c in clist if c.get("role") == "community" and c.get("compound") is not None]
        n_a = int(sent.get("n_author_replies_nonempty") or 0)
        n_c = int(sent.get("n_community_replies_nonempty") or 0)
        orig_rows.append(
            {
                "reddit_id": rid,
                "label": r["final_discussion"],
                "n_author": n_a,
                "n_community": n_c,
                "first_author_minus_post_vader": (author[0] - float(post)) if author and post is not None else None,
                "second_minus_first_vader": (author[1] - author[0]) if len(author) >= 2 else None,
                "author_shift_ge4_vader": early_late_or_none(author) if n_a >= 4 else None,
                "author_minus_community_mean_vader": (
                    float(np.mean(author)) - float(np.mean(community)) if author and community else None
                ),
            }
        )

    report = {
        "utc": utcnow(),
        "interpretation_rule": "Sentiment change ≠ issue resolution",
        "ai_join_missingness": missing,
        "ai_n_threads": len(thread_rows),
        "ai_eligibility": {
            "author_ge1": len(ge1),
            "author_ge2": len(ge2),
            "author_ge4": len(ge4),
            "both_roles": len(both),
            "n_gds_ge1": sum(1 for r in ge1 if r["label"] == "GDS"),
            "n_tps_ge1": sum(1 for r in ge1 if r["label"] == "TPS"),
            "n_gds_ge2": sum(1 for r in ge2 if r["label"] == "GDS"),
            "n_tps_ge2": sum(1 for r in ge2 if r["label"] == "TPS"),
            "n_gds_both": sum(1 for r in both if r["label"] == "GDS"),
            "n_tps_both": sum(1 for r in both if r["label"] == "TPS"),
        },
        "rq2_ai_first_author_minus_post": contrast(ge1, "first_author_minus_post_vader", SEED + 300),
        "rq2_ai_second_minus_first": contrast(ge2, "second_minus_first_vader", SEED + 301),
        "rq2_ai_author_ge4_shift": contrast(ge4, "author_shift_ge4_vader", SEED + 302),
        "rq3_ai_author_minus_community": contrast(both, "author_minus_community_mean_vader", SEED + 400),
        "rq2_original_first_author_minus_post": contrast(
            [r for r in orig_rows if r["n_author"] >= 1], "first_author_minus_post_vader", SEED + 310
        ),
        "rq2_original_second_minus_first": contrast(
            [r for r in orig_rows if r["n_author"] >= 2], "second_minus_first_vader", SEED + 311
        ),
        "rq2_original_author_ge4_shift": contrast(
            [r for r in orig_rows if r["n_author"] >= 4], "author_shift_ge4_vader", SEED + 312
        ),
        "rq3_original_author_minus_community": contrast(
            [r for r in orig_rows if r["n_author"] >= 1 and r["n_community"] >= 1],
            "author_minus_community_mean_vader",
            SEED + 410,
        ),
        "cardiff_reply_scoring": cardiff_status,
        "strata_kept_separate": True,
        "scientific_status": "AI-selected labels unconfirmed; VADER from frozen comment_scores + exact-join TextBlob",
    }
    write_json(dest / "EXPANDED_CONVERSATION_RQ2_RQ3.json", report)
    return report
