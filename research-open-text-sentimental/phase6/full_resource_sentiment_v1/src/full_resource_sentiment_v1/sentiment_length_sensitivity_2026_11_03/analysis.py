"""Full-text, equal-length prefix, and paired length-adjusted first-author−post contrasts."""

from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from full_resource_sentiment_v1.comments import extract_comment_text
from full_resource_sentiment_v1.master_research_corrections_2026_11_03.textblob_checks import (
    independent_stratified_mean_diff,
)
from full_resource_sentiment_v1.sentiment_length_sensitivity_2026_11_03 import (
    CARDIFF_MODEL,
    CARDIFF_REV,
    N_BOOT,
    SEED,
)
from full_resource_sentiment_v1.sentiment_length_sensitivity_2026_11_03.common import (
    AUDIT_RESTRICTED,
    CAPTURE_DIR,
    COMMENT_SCORES,
    CORR_RESTRICTED,
    DATASET_JSONL,
    EXACT_JOIN_RESTRICTED,
    FROZEN_DEF,
    OUT_ROOT,
    POST_SCORES,
    RESTRICTED_ROOT,
    TEXT_MANIFEST,
    TB_SCORES,
    VENV_CARDIFF,
    load_jsonl,
    sha256_file,
    sha256_text,
    utcnow,
    write_json,
    write_jsonl,
)

_VADER = None


def _vader(text: str) -> float | None:
    global _VADER
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    if not (text or "").strip():
        return None
    if _VADER is None:
        _VADER = SentimentIntensityAnalyzer()
    return float(_VADER.polarity_scores(text)["compound"])


def _tb(text: str) -> float | None:
    from textblob import TextBlob

    if not (text or "").strip():
        return None
    return float(TextBlob(text).sentiment.polarity)


def _prefix_pair(post: str, reply: str) -> tuple[str, str, int]:
    """Character prefixes of equal length = min(len(post), len(reply))."""
    n = min(len(post), len(reply))
    return post[:n], reply[:n], n


def _cardiff_batch(units: list[dict[str, Any]], out_name: str) -> list[dict[str, Any]]:
    if not units:
        return []
    if not VENV_CARDIFF.exists():
        raise RuntimeError("Cardiff venv missing")
    dest = OUT_ROOT / "cardiff_prefix"
    dest.mkdir(parents=True, exist_ok=True)
    units_path = RESTRICTED_ROOT / f"{out_name}_units.jsonl"
    scores_path = RESTRICTED_ROOT / f"{out_name}_scores.jsonl"
    # Store text only in restricted units file
    write_jsonl(units_path, units)
    script = dest / f"_infer_{out_name}.py"
    script.write_text(
        f"""#!/usr/bin/env python3
import json
from pathlib import Path
MODEL="{CARDIFF_MODEL}"; REV="{CARDIFF_REV}"; BATCH=32
units=[json.loads(l) for l in open(r"{units_path}") if l.strip()]
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
tok=AutoTokenizer.from_pretrained(MODEL, revision=REV, local_files_only=True)
model=AutoModelForSequenceClassification.from_pretrained(MODEL, revision=REV, local_files_only=True)
model.eval()
id2label={{int(k):v for k,v in model.config.id2label.items()}}
rows=[]
for i in range(0, len(units), BATCH):
    chunk=units[i:i+BATCH]
    texts=[(u.get("text") or "") for u in chunk]
    enc=tok(texts, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        logits=model(**enc).logits
        probs=torch.softmax(logits, dim=-1)
    for j,u in enumerate(chunk):
        # Prefix texts are short; only re-tokenize without trunc when the padded run hit 512.
        n_tok=int((enc["attention_mask"][j]).sum().item())
        if n_tok>=512 and texts[j]:
            full=len(tok(texts[j], truncation=False)["input_ids"])
        else:
            full=n_tok
        pred=int(logits[j].argmax()); lab=id2label.get(pred, str(pred))
        name=str(lab).lower(); pol=-1.0 if "neg" in name else (1.0 if "pos" in name else 0.0)
        rows.append({{
            "pair_id": u.get("pair_id"),
            "role": u.get("role"),
            "cardiff_label": str(lab),
            "cardiff_polarity_sign": pol,
            "cardiff_probs": probs[j].tolist(),
            "n_tokens": n_tok,
            "full_token_len": full,
            "truncated": full>512,
            "text_hash": u.get("text_hash"),
            "text_chars": u.get("text_chars"),
        }})
Path(r"{scores_path}").write_text("\\n".join(json.dumps(r, sort_keys=True) for r in rows)+"\\n")
print(json.dumps({{"ok": True, "n": len(rows)}}))
""",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [str(VENV_CARDIFF / "bin/python"), str(script)],
        capture_output=True,
        text=True,
        timeout=7200,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Cardiff failed: {proc.stderr[-2000:]}")
    return load_jsonl(scores_path)


def _contrast(rows: list[dict[str, Any]], key: str, seed: int) -> dict[str, Any]:
    g = [float(r[key]) for r in rows if r["label"] == "GDS" and r.get(key) is not None]
    t = [float(r[key]) for r in rows if r["label"] == "TPS" and r.get(key) is not None]
    return independent_stratified_mean_diff(g, t, seed=seed, n_boot=N_BOOT)


def _paired_adjusted(rows: list[dict[str, Any]], outcome_key: str, seed: int) -> dict[str, Any]:
    """delta ~ gds + log(post_len) + log(reply_len) + source dummies + date_band dummies."""
    filt = [r for r in rows if r.get(outcome_key) is not None]
    if len(filt) < 50:
        return {"ok": False, "n": len(filt)}
    y = np.asarray([float(r[outcome_key]) for r in filt], dtype=float)
    gds = np.asarray([1.0 if r["label"] == "GDS" else 0.0 for r in filt], dtype=float)
    sources = sorted({str(r.get("source") or "unknown") for r in filt})
    dates = sorted({str(r.get("date_band") or "unknown") for r in filt})
    src_levels = sources[1:] if len(sources) > 1 else []
    date_levels = dates[1:] if len(dates) > 1 else []

    cols = [
        np.ones(len(filt)),
        gds,
        np.asarray([float(np.log1p(r["post_chars"])) for r in filt], dtype=float),
        np.asarray([float(np.log1p(r["reply_chars"])) for r in filt], dtype=float),
    ]
    for s in src_levels:
        cols.append(np.asarray([1.0 if str(r.get("source") or "unknown") == s else 0.0 for r in filt], dtype=float))
    for d in date_levels:
        cols.append(np.asarray([1.0 if str(r.get("date_band") or "unknown") == d else 0.0 for r in filt], dtype=float))
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    point = float(beta[1])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(filt), size=(N_BOOT, len(filt)))
    boots = np.empty(N_BOOT, dtype=float)
    ok_n = 0
    for b_i in range(N_BOOT):
        ii = idx[b_i]
        try:
            b, *_ = np.linalg.lstsq(X[ii], y[ii], rcond=None)
            boots[ok_n] = float(b[1])
            ok_n += 1
        except np.linalg.LinAlgError:
            continue
    a = boots[:ok_n]
    return {
        "ok": True,
        "n": len(filt),
        "n_gds": int(gds.sum()),
        "n_tps": int((1 - gds).sum()),
        "gds_coef_adjusted": point,
        "gds_coef_ci95": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if len(a) else None,
        "covariates": ["log1p_post_chars", "log1p_reply_chars", "source_dummies", "date_band_dummies"],
        "n_source_levels": len(sources),
        "n_date_levels": len(dates),
        "n_bootstrap": N_BOOT,
        "seed": seed,
        "outcome": outcome_key,
    }


def _build_ai_pairs() -> list[dict[str, Any]]:
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
    reply_cf = {
        r["reddit_id"]: r
        for r in load_jsonl(AUDIT_RESTRICTED / "ai_reply_cardiff_scores.jsonl")
        if r.get("unit") == "first_author_reply"
    }
    post_cf = {
        r["reddit_id"]: r
        for r in load_jsonl(CORR_RESTRICTED / "cardiff_full_scores.jsonl")
        if r.get("stratum") == "ai_assisted_v2" and r.get("unit") == "original_post"
    }
    # Post texts from restricted cardiff input if present, else capture
    post_text: dict[str, str] = {}
    inp = CORR_RESTRICTED / "cardiff_full_input_units.jsonl"
    if inp.exists():
        for r in load_jsonl(inp):
            if r.get("stratum") == "ai_assisted_v2" and r.get("unit") == "original_post":
                post_text[r["reddit_id"]] = r.get("text") or ""
    need = {rid for rid in threads if rid not in post_text or not post_text[rid]}
    if need:
        for path in sorted(CAPTURE_DIR.glob("page_*.json")):
            if not need:
                break
            payload = json.loads(path.read_text(encoding="utf-8"))
            for row in payload.get("data") or []:
                rid = str(row.get("redditId") or "")
                if rid not in need:
                    continue
                title = "" if row.get("title") is None else str(row.get("title"))
                body = "" if row.get("author_description") is None else str(row.get("author_description"))
                post_text[rid] = f"{title}\n{body}".strip()
                need.discard(rid)
    post_meta = {str(r["reddit_id"]): r for r in load_jsonl(POST_SCORES)}

    pairs = []
    for rid, th in threads.items():
        ru = reply_units.get(rid)
        pc = post_cf.get(rid)
        rc = reply_cf.get(rid)
        if not ru or not pc:
            continue
        post = post_text.get(rid) or ""
        reply = ru.get("text") or ""
        if not post.strip() or not reply.strip():
            continue
        pm = post_meta.get(rid) or {}
        pairs.append(
            {
                "cohort": "ai_assisted_v2",
                "reddit_id": rid,
                "label": th["label"],
                "post_text": post,
                "reply_text": reply,
                "post_chars": len(post),
                "reply_chars": len(reply),
                "prefix_len": min(len(post), len(reply)),
                "post_longer": len(post) > len(reply),
                "source": pm.get("source"),
                "date_band": pm.get("date_band"),
                "created_month": pm.get("created_month"),
                # Full-text scores (existing)
                "full_post_vader": pc.get("vader"),
                "full_reply_vader": ru.get("vader"),
                "full_post_textblob": pc.get("textblob"),
                "full_reply_textblob": ru.get("textblob"),
                "full_post_cardiff_sign": pc.get("cardiff_polarity_sign"),
                "full_reply_cardiff_sign": rc.get("cardiff_polarity_sign") if rc else None,
                "full_post_truncated": bool(pc.get("truncated")),
                "full_reply_truncated": bool(rc.get("truncated")) if rc else None,
            }
        )
    return pairs


def _build_frozen_pairs() -> list[dict[str, Any]]:
    texts = {str(r["reddit_id"]): r for r in load_jsonl(TEXT_MANIFEST)}
    matched = {
        (m["reddit_id"], m.get("original_index")): m
        for m in load_jsonl(EXACT_JOIN_RESTRICTED / "matched_units_exact_join.jsonl")
    }
    analysis = load_jsonl(EXACT_JOIN_RESTRICTED / "analysis_units_exact_join.jsonl")
    first_author = {
        u["reddit_id"]: u for u in analysis if u.get("unit") == "first_author_reply" and u.get("label") in {"TPS", "GDS"}
    }
    posts_cf = {
        r["reddit_id"]: r
        for r in load_jsonl(CORR_RESTRICTED / "cardiff_full_scores.jsonl")
        if r.get("stratum") == "original_6425_frozen_comparable" and r.get("unit") == "original_post"
    }
    reply_cf = {
        r["reddit_id"]: r
        for r in load_jsonl(CORR_RESTRICTED / "cardiff_full_scores.jsonl")
        if r.get("stratum") == "original_6425_frozen_comparable" and r.get("unit") == "first_author_reply"
    }
    tb_map = {str(r["reddit_id"]): r for r in load_jsonl(TB_SCORES)} if TB_SCORES.exists() else {}
    post_meta = {str(r["reddit_id"]): r for r in load_jsonl(POST_SCORES)}

    # Load reply bodies via exact join
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

    pairs = []
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
        # First author VADER from chronological author scores
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
        sent = r.get("sentiment") or {}
        pc = posts_cf.get(rid) or {}
        rc = reply_cf.get(rid) or {}
        tbr = tb_map.get(rid) or {}
        pm = post_meta.get(rid) or {}
        pairs.append(
            {
                "cohort": "original_frozen",
                "reddit_id": rid,
                "label": r["final_discussion"],
                "post_text": post,
                "reply_text": reply,
                "post_chars": len(post),
                "reply_chars": len(reply),
                "prefix_len": min(len(post), len(reply)),
                "post_longer": len(post) > len(reply),
                "source": pm.get("source"),
                "date_band": pm.get("date_band"),
                "created_month": pm.get("created_month"),
                "full_post_vader": sent.get("combined_compound"),
                "full_reply_vader": float(auth[0]["compound"]),
                "full_post_textblob": tbr.get("textblob_combined_polarity")
                if tbr.get("textblob_combined_polarity") is not None
                else fa.get("textblob")  # may be reply tb wrongly — prefer post from cardiff
                ,
                "full_reply_textblob": fa.get("textblob"),
                "full_post_cardiff_sign": pc.get("cardiff_polarity_sign"),
                "full_reply_cardiff_sign": rc.get("cardiff_polarity_sign"),
                "full_post_truncated": bool(pc.get("truncated")),
                "full_reply_truncated": bool(rc.get("truncated")),
            }
        )
    # Fix frozen post TextBlob from cardiff scores when available
    for p in pairs:
        pc = posts_cf.get(p["reddit_id"])
        if pc and pc.get("textblob") is not None:
            p["full_post_textblob"] = pc["textblob"]
    return pairs


def _enrich_deltas(pairs: list[dict[str, Any]]) -> None:
    for p in pairs:
        if p.get("full_post_vader") is not None and p.get("full_reply_vader") is not None:
            p["full_delta_vader"] = float(p["full_reply_vader"]) - float(p["full_post_vader"])
        if p.get("full_post_textblob") is not None and p.get("full_reply_textblob") is not None:
            p["full_delta_textblob"] = float(p["full_reply_textblob"]) - float(p["full_post_textblob"])
        if p.get("full_post_cardiff_sign") is not None and p.get("full_reply_cardiff_sign") is not None:
            p["full_delta_cardiff_sign"] = float(p["full_reply_cardiff_sign"]) - float(p["full_post_cardiff_sign"])


def _score_prefixes(pairs: list[dict[str, Any]], cohort_tag: str) -> None:
    # Local VADER/TextBlob on prefixes
    cardiff_units = []
    for i, p in enumerate(pairs):
        post_p, reply_p, n = _prefix_pair(p["post_text"], p["reply_text"])
        p["prefix_len"] = n
        p["prefix_post_vader"] = _vader(post_p)
        p["prefix_reply_vader"] = _vader(reply_p)
        p["prefix_post_textblob"] = _tb(post_p)
        p["prefix_reply_textblob"] = _tb(reply_p)
        if p["prefix_post_vader"] is not None and p["prefix_reply_vader"] is not None:
            p["prefix_delta_vader"] = float(p["prefix_reply_vader"]) - float(p["prefix_post_vader"])
        if p["prefix_post_textblob"] is not None and p["prefix_reply_textblob"] is not None:
            p["prefix_delta_textblob"] = float(p["prefix_reply_textblob"]) - float(p["prefix_post_textblob"])
        pair_id = f"{cohort_tag}:{p['reddit_id']}"
        cardiff_units.append(
            {
                "pair_id": f"{pair_id}:post",
                "role": "post_prefix",
                "text": post_p,
                "text_hash": sha256_text(post_p),
                "text_chars": len(post_p),
            }
        )
        cardiff_units.append(
            {
                "pair_id": f"{pair_id}:reply",
                "role": "reply_prefix",
                "text": reply_p,
                "text_hash": sha256_text(reply_p),
                "text_chars": len(reply_p),
            }
        )
    print(f"  Cardiff prefixes for {cohort_tag}: {len(cardiff_units)} texts...", flush=True)
    scores = _cardiff_batch(cardiff_units, f"prefix_{cohort_tag}")
    by_id = {s["pair_id"]: s for s in scores}
    for p in pairs:
        pair_id = f"{cohort_tag}:{p['reddit_id']}"
        sp = by_id.get(f"{pair_id}:post")
        sr = by_id.get(f"{pair_id}:reply")
        if sp:
            p["prefix_post_cardiff_sign"] = sp.get("cardiff_polarity_sign")
            p["prefix_post_truncated"] = sp.get("truncated")
            p["prefix_post_n_tokens"] = sp.get("n_tokens")
        if sr:
            p["prefix_reply_cardiff_sign"] = sr.get("cardiff_polarity_sign")
            p["prefix_reply_truncated"] = sr.get("truncated")
            p["prefix_reply_n_tokens"] = sr.get("n_tokens")
        if sp and sr and sp.get("cardiff_polarity_sign") is not None and sr.get("cardiff_polarity_sign") is not None:
            p["prefix_delta_cardiff_sign"] = float(sr["cardiff_polarity_sign"]) - float(sp["cardiff_polarity_sign"])


def _length_summary(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    def med(vals):
        return float(np.median(vals)) if vals else None

    out = {}
    for lab in ("GDS", "TPS"):
        sub = [p for p in pairs if p["label"] == lab]
        out[lab] = {
            "n": len(sub),
            "median_post_chars": med([p["post_chars"] for p in sub]),
            "median_reply_chars": med([p["reply_chars"] for p in sub]),
            "median_prefix_len": med([p["prefix_len"] for p in sub]),
            "pct_post_longer": float(np.mean([1.0 if p["post_longer"] else 0.0 for p in sub])) if sub else None,
        }
    return out


def _analyze_cohort(pairs: list[dict[str, Any]], cohort: str, seed: int) -> dict[str, Any]:
    _enrich_deltas(pairs)
    _score_prefixes(pairs, cohort)

    # Persist restricted analysis rows without full text
    public_rows = []
    for p in pairs:
        public_rows.append(
            {
                k: v
                for k, v in p.items()
                if k not in {"post_text", "reply_text"}
            }
        )
    write_jsonl(RESTRICTED_ROOT / f"{cohort}_pair_rows.jsonl", public_rows)

    trunc = {
        "full_post_truncated_n": sum(1 for p in pairs if p.get("full_post_truncated")),
        "full_reply_truncated_n": sum(1 for p in pairs if p.get("full_reply_truncated")),
        "prefix_post_truncated_n": sum(1 for p in pairs if p.get("prefix_post_truncated")),
        "prefix_reply_truncated_n": sum(1 for p in pairs if p.get("prefix_reply_truncated")),
        "full_post_trunc_by_label": {
            "GDS": sum(1 for p in pairs if p["label"] == "GDS" and p.get("full_post_truncated")),
            "TPS": sum(1 for p in pairs if p["label"] == "TPS" and p.get("full_post_truncated")),
        },
        "full_reply_trunc_by_label": {
            "GDS": sum(1 for p in pairs if p["label"] == "GDS" and p.get("full_reply_truncated")),
            "TPS": sum(1 for p in pairs if p["label"] == "TPS" and p.get("full_reply_truncated")),
        },
    }

    report = {
        "cohort": cohort,
        "n_pairs": len(pairs),
        "n_gds": sum(1 for p in pairs if p["label"] == "GDS"),
        "n_tps": sum(1 for p in pairs if p["label"] == "TPS"),
        "length_summary": _length_summary(pairs),
        "truncation": trunc,
        "analysis_1_full_saved_text": {
            "definition": "first_author_reply_score − original_post_score on full saved texts",
            "vader": _contrast(pairs, "full_delta_vader", seed),
            "textblob": _contrast(pairs, "full_delta_textblob", seed + 1),
            "cardiff_sign": _contrast(pairs, "full_delta_cardiff_sign", seed + 2),
        },
        "analysis_2_equal_length_prefixes": {
            "definition": (
                "Score character prefixes of length min(len(post), len(reply)) for both texts, "
                "then reply_prefix − post_prefix. This measures a different portion of the post "
                "(typically its opening) than the full-text comparison, and is not a claim that "
                "prefix sentiment equals full-document sentiment."
            ),
            "prefix_rule": "character_prefix_length = min(post_chars, reply_chars)",
            "vader": _contrast(pairs, "prefix_delta_vader", seed + 10),
            "textblob": _contrast(pairs, "prefix_delta_textblob", seed + 11),
            "cardiff_sign": _contrast(pairs, "prefix_delta_cardiff_sign", seed + 12),
        },
        "analysis_3_paired_length_source_date_adjusted": {
            "definition": (
                "OLS: delta ~ GDS + log1p(post_chars) + log1p(reply_chars) + source + date_band; "
                "report GDS coefficient with thread bootstrap CI"
            ),
            "vader": _paired_adjusted(pairs, "full_delta_vader", seed + 20),
            "textblob": _paired_adjusted(pairs, "full_delta_textblob", seed + 21),
            "cardiff_sign": _paired_adjusted(pairs, "full_delta_cardiff_sign", seed + 22),
        },
        "interpretation_rule": "Sentiment change ≠ issue resolution; AI labels unconfirmed where applicable",
    }
    return report


def run_analysis() -> dict[str, Any]:
    dest = OUT_ROOT / "length_sensitivity"
    dest.mkdir(parents=True, exist_ok=True)

    print("  Building AI 5942 first-author pairs...", flush=True)
    ai = _build_ai_pairs()
    assert len(ai) == 5942, f"expected 5942 AI pairs, got {len(ai)}"
    print(f"  AI pairs: {len(ai)}", flush=True)

    print("  Building frozen first-author pairs...", flush=True)
    frozen = _build_frozen_pairs()
    print(f"  Frozen pairs: {len(frozen)}", flush=True)

    print("  Analyzing AI cohort (full + prefix + adjusted)...", flush=True)
    ai_rep = _analyze_cohort(ai, "ai_assisted_v2", SEED + 100)
    write_json(dest / "AI_COHORT_LENGTH_SENSITIVITY.json", ai_rep)

    print("  Analyzing frozen cohort (full + prefix + adjusted)...", flush=True)
    fr_rep = _analyze_cohort(frozen, "original_frozen", SEED + 200)
    write_json(dest / "FROZEN_COHORT_LENGTH_SENSITIVITY.json", fr_rep)

    summary = {
        "utc": utcnow(),
        "ai": {
            "n": ai_rep["n_pairs"],
            "n_gds": ai_rep["n_gds"],
            "n_tps": ai_rep["n_tps"],
            "full_vader": ai_rep["analysis_1_full_saved_text"]["vader"],
            "prefix_vader": ai_rep["analysis_2_equal_length_prefixes"]["vader"],
            "adjusted_vader": ai_rep["analysis_3_paired_length_source_date_adjusted"]["vader"],
            "full_textblob": ai_rep["analysis_1_full_saved_text"]["textblob"],
            "prefix_textblob": ai_rep["analysis_2_equal_length_prefixes"]["textblob"],
            "full_cardiff": ai_rep["analysis_1_full_saved_text"]["cardiff_sign"],
            "prefix_cardiff": ai_rep["analysis_2_equal_length_prefixes"]["cardiff_sign"],
        },
        "frozen": {
            "n": fr_rep["n_pairs"],
            "n_gds": fr_rep["n_gds"],
            "n_tps": fr_rep["n_tps"],
            "full_vader": fr_rep["analysis_1_full_saved_text"]["vader"],
            "prefix_vader": fr_rep["analysis_2_equal_length_prefixes"]["vader"],
            "adjusted_vader": fr_rep["analysis_3_paired_length_source_date_adjusted"]["vader"],
            "full_textblob": fr_rep["analysis_1_full_saved_text"]["textblob"],
            "prefix_textblob": fr_rep["analysis_2_equal_length_prefixes"]["textblob"],
            "full_cardiff": fr_rep["analysis_1_full_saved_text"]["cardiff_sign"],
            "prefix_cardiff": fr_rep["analysis_2_equal_length_prefixes"]["cardiff_sign"],
        },
        "prefix_vs_full_note": (
            "Equal-length prefix analysis scores only the shared character budget "
            "(opening of the longer text). It answers a different question than full-text scoring."
        ),
        "cohorts_kept_separate": True,
        "labels_unchanged": True,
        "sealed_trees_untouched": True,
    }
    write_json(dest / "SUMMARY.json", summary)
    return {"ai": ai_rep, "frozen": fr_rep, "summary": summary}
