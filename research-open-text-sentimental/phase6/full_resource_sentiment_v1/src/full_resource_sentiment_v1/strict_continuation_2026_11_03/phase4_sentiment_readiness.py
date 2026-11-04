"""Phase 4: transformer readiness inventory + matched VADER/TextBlob role sample."""

from __future__ import annotations

import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from full_resource_sentiment_v1.strict_continuation_2026_11_03 import SEED
from full_resource_sentiment_v1.strict_continuation_2026_11_03.common import (
    COMMENT_SCORES,
    DATASET_JSONL,
    FROZEN_DEF,
    OUT_ROOT,
    TEXT_MANIFEST,
    load_jsonl,
    utcnow,
    write_json,
)
from full_resource_sentiment_v1.master_research_corrections_2026_11_03.textblob_checks import (
    independent_stratified_mean_diff,
)


def _cache_inventory(model_dir_name: str, pinned_rev: str | None = None) -> dict[str, Any]:
    hub = Path.home() / ".cache/huggingface/hub" / model_dir_name
    snaps = []
    if (hub / "snapshots").exists():
        for s in sorted((hub / "snapshots").iterdir()):
            files = sorted(p.name for p in s.rglob("*") if p.is_file())
            snaps.append({"revision": s.name, "files": files, "n_files": len(files)})
    torch_ok = importlib.util.find_spec("torch") is not None
    tf_ok = importlib.util.find_spec("transformers") is not None
    return {
        "cache_root_exists": hub.exists(),
        "snapshots": snaps,
        "pinned_rev_requested": pinned_rev,
        "torch_installed": torch_ok,
        "transformers_installed": tf_ok,
        "runnable_now": False,
        "blocker": None
        if torch_ok and tf_ok
        else "torch/transformers not installed; large download/install requires Manu approval",
    }


def _score_textblob(text: str) -> float | None:
    from textblob import TextBlob

    if not (text or "").strip():
        return None
    return float(TextBlob(text).sentiment.polarity)


def run_phase4() -> dict[str, Any]:
    dest = OUT_ROOT / "phase4_sentiment_readiness"
    dest.mkdir(parents=True, exist_ok=True)

    cardiff = _cache_inventory(
        "models--cardiffnlp--twitter-roberta-base-sentiment-latest",
        pinned_rev="3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7",
    )
    # Completeness check: need tokenizer.json or vocab+merges and model weights
    cardiff_complete = False
    for snap in cardiff.get("snapshots") or []:
        files = set(snap["files"])
        has_weights = "pytorch_model.bin" in files or "model.safetensors" in files
        has_tok = "tokenizer.json" in files or ("vocab.json" in files and "merges.txt" in files)
        if has_weights and has_tok and "config.json" in files:
            cardiff_complete = True
            cardiff["selected_snapshot"] = snap["revision"]
            break
    cardiff["weights_tokenizer_apparently_complete"] = cardiff_complete
    cardiff["license_note"] = "CardiffNLP twitter-roberta-base-sentiment-latest — verify license before use"
    cardiff["est_ram_gb"] = "~2–4 GB inference once torch installed"
    cardiff["est_disk_extra_gb"] = "torch+transformers wheels often >1 GB; model already partially cached"

    bertweet = _cache_inventory(
        "models--finiteautomata--bertweet-base-sentiment-analysis",
        pinned_rev="924fc4c80bccb8003d21fe84dd92c7887717f245",
    )
    sst2 = _cache_inventory("models--distilbert-base-uncased-finetuned-sst-2-english")

    proposal = {
        "action_if_approved": "Install pinned torch+transformers into isolated venv; run 20-case canary then matched frozen TPS/GDS role texts with Cardiff pinned rev",
        "models": ["cardiffnlp/twitter-roberta-base-sentiment-latest", "optional BERTweet"],
        "affected_count_canary": 20,
        "affected_count_full_matched_frozen_posts": 3259,
        "estimated_time": "canary <10 min; full posts ~30–90 min sequential on CPU/M3",
        "estimated_disk": "1–3 GiB for torch stack if not present",
        "estimated_money": 0,
        "privacy": "local only; restricted text stays restricted",
        "rollback": "delete venv; do not write into frozen/accepted trees",
        "status": "AWAITING_USER_APPROVAL — not executed",
    }

    # Matched VADER/TextBlob on roles for a fixed-seed sample of frozen TPS/GDS with author>=1
    texts = {str(r["reddit_id"]): r for r in load_jsonl(TEXT_MANIFEST)}
    comments_by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in load_jsonl(COMMENT_SCORES):
        comments_by[str(c["reddit_id"])].append(c)

    eligible = []
    for r in load_jsonl(DATASET_JSONL):
        if r.get("final_relevance") != "relevant":
            continue
        if r.get("final_discussion") not in {"TPS", "GDS"}:
            continue
        if r.get("sentiment_definition") != FROZEN_DEF:
            continue
        sent = r.get("sentiment") or {}
        if int(sent.get("n_author_replies_nonempty") or 0) < 1:
            continue
        eligible.append(r)

    rng = np.random.default_rng(SEED)
    # Cap matched sample for TextBlob recompute cost while covering both labels
    gds = [r for r in eligible if r["final_discussion"] == "GDS"]
    tps = [r for r in eligible if r["final_discussion"] == "TPS"]
    take_gds = min(98, len(gds))
    take_tps = min(400, len(tps))
    sample_rows = list(rng.choice(gds, size=take_gds, replace=False)) + list(
        rng.choice(tps, size=take_tps, replace=False)
    )

    units = []
    for r in sample_rows:
        rid = str(r["reddit_id"])
        doc = texts.get(rid) or {"title": "", "body": ""}
        post_text = f"{doc.get('title') or ''}\n{doc.get('body') or ''}".strip()
        sent = r.get("sentiment") or {}
        clist = sorted(
            comments_by.get(rid, []),
            key=lambda c: (c.get("timestamp_epoch_seconds") is None, c.get("timestamp_epoch_seconds") or 0),
        )
        author = [c for c in clist if c.get("role") == "author" and c.get("compound") is not None]
        community = [c for c in clist if c.get("role") == "community" and c.get("compound") is not None]
        first_author = author[0] if author else None
        later_author = author[-1] if len(author) >= 2 else None
        first_comm = community[0] if community else None

        def pack(unit: str, vader: float | None, text: str | None):
            tb = _score_textblob(text or "") if text else None
            units.append(
                {
                    "reddit_id": rid,
                    "label": r["final_discussion"],
                    "unit": unit,
                    "vader_compound": vader,
                    "textblob_polarity": tb,
                    "text_chars": len(text or ""),
                    "polarity_sign_agree": (
                        None
                        if vader is None or tb is None
                        else (np.sign(vader) == np.sign(tb) or (abs(vader) < 0.05 and abs(tb) < 0.05))
                    ),
                }
            )

        pack("original_post", sent.get("combined_compound"), post_text)
        if first_author is not None:
            # comment text not in scores file — use compound only for VADER; TextBlob needs text
            # Without restricted comment bodies, TextBlob on replies is blocked unless capture has comments
            pack("first_author_reply_vader_only", float(first_author["compound"]), None)
        if later_author is not None:
            pack("later_author_reply_vader_only", float(later_author["compound"]), None)
        if first_comm is not None:
            pack("first_community_reply_vader_only", float(first_comm["compound"]), None)

    # Within-model post-level GDS−TPS on this sample for TextBlob vs VADER (posts only)
    post_units = [u for u in units if u["unit"] == "original_post"]
    contrasts = {
        "vader_post": independent_stratified_mean_diff(
            [u["vader_compound"] for u in post_units if u["label"] == "GDS" and u["vader_compound"] is not None],
            [u["vader_compound"] for u in post_units if u["label"] == "TPS" and u["vader_compound"] is not None],
            seed=SEED + 21,
        ),
        "textblob_post": independent_stratified_mean_diff(
            [u["textblob_polarity"] for u in post_units if u["label"] == "GDS" and u["textblob_polarity"] is not None],
            [u["textblob_polarity"] for u in post_units if u["label"] == "TPS" and u["textblob_polarity"] is not None],
            seed=SEED + 22,
        ),
    }
    agree_n = sum(1 for u in post_units if u["polarity_sign_agree"] is True)
    disagree_n = sum(1 for u in post_units if u["polarity_sign_agree"] is False)

    # Aspect / software-domain feasibility note
    aspect_note = {
        "status": "not_implemented",
        "requirement": "Defined aspect (e.g., update quality vs support) + independent evaluation evidence",
        "feasibility": "Document only — inventing an aspect model without evaluation evidence is out of scope",
    }

    report = {
        "utc": utcnow(),
        "cardiff": cardiff,
        "bertweet": bertweet,
        "sst2_distilbert": sst2,
        "transformer_execution": "STOPPED — install/download approval required",
        "approval_proposal": proposal,
        "matched_sample": {
            "seed": SEED,
            "n_posts": len(sample_rows),
            "n_gds": take_gds,
            "n_tps": take_tps,
            "eligibility": "frozen TPS/GDS with author>=1",
            "post_sign_agreement_vader_textblob": {"agree": agree_n, "disagree": disagree_n},
            "contrasts_gds_minus_tps_posts": contrasts,
            "reply_textblob": "not_scored — comment bodies not in score files; would need capture comment text join",
            "scale_warning": "Do not treat VADER compound and TextBlob polarity as identical measures",
        },
        "aspect_method": aspect_note,
    }
    write_json(dest / "SENTIMENT_READINESS.json", report)
    (dest / "FINDINGS.md").write_text(
        f"""# Phase 4 — sentiment readiness

**UTC:** {report['utc']}  
Transformers: **not run** (no torch/transformers). Cardiff cache partially present; proposal recorded for approval.

Matched post sample (author≥1): n={len(sample_rows)}; VADER/TextBlob sign agree={agree_n} disagree={disagree_n}.  
Reply TextBlob blocked without comment body join. Scales not commensurate.
""",
        encoding="utf-8",
    )
    (dest / "TRANSFORMER_APPROVAL_PROPOSAL.json").write_text(
        json.dumps(proposal, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
