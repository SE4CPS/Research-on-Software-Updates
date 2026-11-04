"""Full local Cardiff scoring on frozen matched units + AI v2 posts; instrument comparisons."""

from __future__ import annotations

import json
import resource
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from full_resource_sentiment_v1.comments import extract_comment_text
from full_resource_sentiment_v1.correction_rq_cardiff_2026_11_03 import (
    CARDIFF_MODEL,
    CARDIFF_REV,
    DATASET_VERSION,
    SEED,
)
from full_resource_sentiment_v1.correction_rq_cardiff_2026_11_03.common import (
    CAPTURE_DIR,
    EXACT_JOIN_RESTRICTED,
    OUT_ROOT,
    RESTRICTED_ROOT,
    TEXT_MANIFEST,
    VENV_CARDIFF,
    load_jsonl,
    sha256_file,
    sha256_text,
    utcnow,
    write_json,
    write_jsonl,
)


def _tb(text: str) -> float | None:
    from textblob import TextBlob

    if not (text or "").strip():
        return None
    return float(TextBlob(text).sentiment.polarity)


_VADER = None


def _vader(text: str) -> float | None:
    global _VADER
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    if not (text or "").strip():
        return None
    if _VADER is None:
        _VADER = SentimentIntensityAnalyzer()
    return float(_VADER.polarity_scores(text)["compound"])


def _sign(x: float | None, thr: float = 0.0) -> int | None:
    if x is None:
        return None
    if abs(float(x)) < thr:
        return 0
    return int(np.sign(float(x)))


def _load_frozen_units() -> list[dict[str, Any]]:
    units = load_jsonl(EXACT_JOIN_RESTRICTED / "analysis_units_exact_join.jsonl")
    matched = {(m["reddit_id"], m.get("original_index")): m for m in load_jsonl(EXACT_JOIN_RESTRICTED / "matched_units_exact_join.jsonl")}
    texts = {str(r["reddit_id"]): r for r in load_jsonl(TEXT_MANIFEST)}
    need = {u["reddit_id"] for u in units if u.get("unit") != "original_post"}
    pages = {
        int(matched[(u["reddit_id"], u.get("original_index"))]["capture_page"])
        for u in units
        if u.get("unit") != "original_post" and (u["reddit_id"], u.get("original_index")) in matched
    }
    cap: dict[str, list] = {}
    for page in pages:
        path = CAPTURE_DIR / f"page_{page:05d}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("data") or []:
            rid = str(row.get("redditId") or "")
            if rid in need:
                cap[rid] = [c for c in (row.get("comments") or []) if isinstance(c, dict)]

    out = []
    for u in units:
        if u.get("unit") not in {
            "original_post",
            "title_only",
            "body_only",
            "first_author_reply",
            "second_author_reply",
            "later_author_reply",
            "first_community_reply",
        }:
            continue
        if u.get("label") not in {"TPS", "GDS"}:
            continue
        if u["unit"] == "original_post":
            doc = texts.get(u["reddit_id"]) or {}
            text = f"{doc.get('title') or ''}\n{doc.get('body') or ''}".strip()
        elif u["unit"] == "title_only":
            text = ((texts.get(u["reddit_id"]) or {}).get("title") or "").strip()
        elif u["unit"] == "body_only":
            text = ((texts.get(u["reddit_id"]) or {}).get("body") or "").strip()
        else:
            text = ""
            m = matched.get((u["reddit_id"], u.get("original_index")))
            if m and u["reddit_id"] in cap:
                oi = int(m["original_index"])
                comments = cap[u["reddit_id"]]
                if 0 <= oi < len(comments):
                    text, _ = extract_comment_text(comments[oi])
                    text = text or ""
        if not text.strip():
            continue
        out.append(
            {
                "stratum": "original_6425_frozen_comparable",
                "reddit_id": u["reddit_id"],
                "label": u["label"],
                "unit": u["unit"],
                "original_index": u.get("original_index"),
                "vader": u.get("vader"),
                "textblob": u.get("textblob"),
                "text": text,
                "text_hash": sha256_text(text),
                "text_chars": len(text),
            }
        )
    return out


def _load_ai_post_units() -> list[dict[str, Any]]:
    union = load_jsonl(RESTRICTED_ROOT / f"{DATASET_VERSION}.jsonl")
    ai_ids = {
        r["reddit_id"]
        for r in union
        if r.get("record_origin") == "ai_assisted_screen_candidate_v2" and r.get("final_discussion") in {"TPS", "GDS"}
    }
    # Load texts from capture
    remaining = set(ai_ids)
    texts: dict[str, dict[str, str]] = {}
    for path in sorted(CAPTURE_DIR.glob("page_*.json")):
        if not remaining:
            break
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("data") or []:
            rid = str(row.get("redditId") or "")
            if rid not in remaining:
                continue
            title = "" if row.get("title") is None else str(row.get("title"))
            body = "" if row.get("author_description") is None else str(row.get("author_description"))
            texts[rid] = {"title": title, "body": body}
            remaining.discard(rid)
    label = {
        r["reddit_id"]: r["final_discussion"]
        for r in union
        if r.get("record_origin") == "ai_assisted_screen_candidate_v2"
    }
    out = []
    for rid in sorted(ai_ids):
        doc = texts.get(rid) or {"title": "", "body": ""}
        title, body = doc.get("title") or "", doc.get("body") or ""
        post = f"{title}\n{body}".strip()
        if not post:
            continue
        # Compute VADER/TextBlob for AI posts (not in frozen scores)
        out.append(
            {
                "stratum": "ai_assisted_v2",
                "reddit_id": rid,
                "label": label[rid],
                "unit": "original_post",
                "vader": _vader(post),
                "textblob": _tb(post),
                "text": post,
                "text_hash": sha256_text(post),
                "text_chars": len(post),
                "title_textblob": _tb(title) if title.strip() else None,
                "body_textblob": _tb(body) if body.strip() else None,
            }
        )
    return out


def _write_infer_worker(units_path: Path, out_path: Path, shard_meta: Path) -> Path:
    script = OUT_ROOT / "stage2_cardiff" / "_infer_worker_full.py"
    script.parent.mkdir(parents=True, exist_ok=True)
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
model.eval()
load_s=time.time()-t0
id2label={{int(k):v for k,v in model.config.id2label.items()}}
rows=[]; max_rss=0; n_trunc=0
for i,u in enumerate(units):
    text=u.get("text") or ""
    t1=time.time()
    full_len=len(tok(text, truncation=False)["input_ids"]) if text else 0
    enc=tok(text, return_tensors="pt", truncation=True, max_length=512)
    n_tok=int(enc["input_ids"].shape[1]); truncated=full_len>512
    if truncated: n_trunc+=1
    with torch.no_grad():
        logits=model(**enc).logits[0]
        probs=torch.softmax(logits, dim=-1).tolist()
    pred_i=int(logits.argmax()); label=id2label.get(pred_i, str(pred_i))
    name=str(label).lower()
    pol=-1.0 if "neg" in name else (1.0 if "pos" in name else 0.0)
    try:
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss; max_rss=max(max_rss,rss)
    except Exception:
        rss=None
    rows.append({{
        "stratum": u.get("stratum"),
        "reddit_id": u.get("reddit_id"),
        "label": u.get("label"),
        "unit": u.get("unit"),
        "vader": u.get("vader"),
        "textblob": u.get("textblob"),
        "text_hash": u.get("text_hash"),
        "text_chars": u.get("text_chars"),
        "cardiff_label": str(label),
        "cardiff_polarity_sign": pol,
        "cardiff_probs": probs,
        "n_tokens": n_tok,
        "full_token_len": full_len,
        "truncated": bool(truncated),
        "latency_s": time.time()-t1,
    }})
    if (i+1)%500==0:
        print(json.dumps({{"progress": i+1, "n": len(units)}}), flush=True)
meta={{"load_seconds":load_s,"n":len(rows),"n_truncated":n_trunc,"max_rss_raw":max_rss,"model":MODEL,"revision":REV}}
Path(r"{out_path}").write_text("\\n".join(json.dumps(r,sort_keys=True) for r in rows)+"\\n")
Path(r"{shard_meta}").write_text(json.dumps(meta,indent=2,sort_keys=True))
print(json.dumps({{"ok":True,"n":len(rows),"trunc":n_trunc,"load_s":load_s}}))
""",
        encoding="utf-8",
    )
    return script


def _compare(rows: list[dict[str, Any]], thr: float = 0.0) -> dict[str, Any]:
    n = cv = ct = vt = 0
    for r in rows:
        if r.get("vader") is None or r.get("textblob") is None or r.get("cardiff_polarity_sign") is None:
            continue
        n += 1
        cs = _sign(r["cardiff_polarity_sign"], 0.0)  # already -1/0/1
        # For Cardiff, neutral is class; for VADER/TB use thr
        vs = _sign(r["vader"], thr)
        ts = _sign(r["textblob"], thr)
        # Map cardiff: already discrete; for thr>0 treat cardiff 0 as neutral always
        if cs == vs:
            cv += 1
        if cs == ts:
            ct += 1
        if vs == ts:
            vt += 1
    n = max(n, 1)
    return {
        "n": n,
        "neutral_threshold": thr,
        "cardiff_vs_vader": cv / n,
        "cardiff_vs_textblob": ct / n,
        "vader_vs_textblob": vt / n,
    }


def run_cardiff_full() -> dict[str, Any]:
    dest = OUT_ROOT / "stage2_cardiff"
    dest.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage("/").free / (1024**3)
    resource_gate = {
        "utc": utcnow(),
        "free_disk_gib": round(free, 3),
        "venv_exists": VENV_CARDIFF.exists(),
        "min_free_gib": 1.5,
        "ok": free >= 1.5 and VENV_CARDIFF.exists(),
    }
    write_json(dest / "RESOURCE_MONITOR_START.json", resource_gate)
    if not resource_gate["ok"]:
        write_json(dest / "CARDIFF_FULL_STATUS.json", {"status": "BLOCKED", "resource": resource_gate})
        return {"status": "BLOCKED", "resource": resource_gate}

    print("  assembling frozen matched units...", flush=True)
    frozen = _load_frozen_units()
    print(f"  frozen units with text: {len(frozen)}", flush=True)
    print("  assembling AI v2 original posts + VADER/TextBlob...", flush=True)
    ai = _load_ai_post_units()
    print(f"  AI post units: {len(ai)}", flush=True)

    # Persist units without huge duplication in public tree — restricted only
    all_units = frozen + ai
    units_path = RESTRICTED_ROOT / "cardiff_full_input_units.jsonl"
    write_jsonl(units_path, all_units)

    out_path = RESTRICTED_ROOT / "cardiff_full_scores.jsonl"
    meta_path = dest / "CARDIFF_FULL_META.json"
    script = _write_infer_worker(units_path, out_path, meta_path)
    py = VENV_CARDIFF / "bin" / "python"
    print(f"  running Cardiff on {len(all_units)} units...", flush=True)
    t0 = time.time()
    proc = subprocess.run([str(py), str(script)], capture_output=True, text=True, timeout=7200)
    elapsed = time.time() - t0
    write_json(
        dest / "CARDIFF_FULL_RUN.json",
        {"returncode": proc.returncode, "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-2500:], "elapsed_s": elapsed},
    )
    if proc.returncode != 0 or not out_path.exists():
        write_json(dest / "CARDIFF_FULL_STATUS.json", {"status": "FAILED", "run": "see CARDIFF_FULL_RUN.json"})
        return {"status": "FAILED"}

    rows = load_jsonl(out_path)
    truncated = [r for r in rows if r.get("truncated")]
    write_jsonl(RESTRICTED_ROOT / "cardiff_truncated_units.jsonl", truncated)

    # Pilot-style comparisons + neutral bands
    frozen_rows = [r for r in rows if r.get("stratum") == "original_6425_frozen_comparable"]
    ai_rows = [r for r in rows if r.get("stratum") == "ai_assisted_v2"]
    report = {
        "utc": utcnow(),
        "status": "COMPLETE",
        "n_scored": len(rows),
        "n_frozen_units": len(frozen_rows),
        "n_ai_posts": len(ai_rows),
        "n_truncated": len(truncated),
        "elapsed_s": elapsed,
        "sha256_scores": sha256_file(out_path),
        "agreement_exact_zero_frozen": _compare(frozen_rows, 0.0),
        "agreement_neutral_0_05_frozen": _compare(frozen_rows, 0.05),
        "agreement_exact_zero_ai_posts": _compare(ai_rows, 0.0),
        "truncated_sensitivity": {
            "n": len(truncated),
            "agreement_exact_zero_truncated_only": _compare(truncated, 0.0),
            "agreement_exact_zero_nontruncated_frozen": _compare(
                [r for r in frozen_rows if not r.get("truncated")], 0.0
            ),
        },
        "by_unit_frozen": dict(Counter(r.get("unit") for r in frozen_rows)),
        "note": "Instrument agreement ≠ accuracy; pilot remains exploratory reference",
        "resource_end_free_gib": round(shutil.disk_usage("/").free / (1024**3), 3),
    }
    write_json(dest / "CARDIFF_FULL_RESULTS.json", report)
    write_json(dest / "CARDIFF_FULL_STATUS.json", {"status": "COMPLETE", "n": len(rows)})
    return report
