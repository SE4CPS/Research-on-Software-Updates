"""Phase B: fixed-240 diagnostics + second rubric classifier (local, no paid API)."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from full_resource_sentiment_v1.master_research_2026_11_03.common import (
    CAPTURE_DIR,
    OUT_ROOT,
    PILOT_IDS,
    PILOT_STRATA,
    RESTRICTED_ROOT,
    V3_PREDICTIONS,
    load_jsonl,
    sha256_file,
    sha256_text,
    utcnow,
    write_json,
    write_jsonl,
)
from full_resource_sentiment_v1.paths import FRESH_500, RESERVED_TEST, WORKSPACE

UPDATE_CUES = re.compile(
    r"\b(update|upgrade|patch|rollback|firmware|version|changelog|release notes?|kernel)\b",
    re.I,
)
TROUBLE_CUES = re.compile(
    r"\b(error|crash|bug|fail(?:ed|ure)?|broken|fix|workaround|stacktrace|regression|not working)\b",
    re.I,
)
DISCONTENT_CUES = re.compile(
    r"\b(hate|ruined|awful|terrible|garbage|worst|disappointed|frustrated|angry|sucks)\b",
    re.I,
)
REMOVED = re.compile(r"^\[(removed|deleted)\]$", re.I)


def protected_ids() -> set[str]:
    out: set[str] = set()
    for path in (RESERVED_TEST, FRESH_500):
        for row in load_jsonl(path):
            rid = row.get("reddit_id") or row.get("id")
            if rid:
                out.add(str(rid))
    return out


def load_pilot() -> list[dict[str, Any]]:
    with PILOT_IDS.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_capture_texts_for_ids(ids: set[str]) -> dict[str, dict[str, Any]]:
    """Pilot 240 is drawn from the excluded pool, not the 6,425 frame.

    Title/body must come from broad capture pages, not the frame-only manifest.
    """
    texts: dict[str, dict[str, Any]] = {}
    remaining = set(ids)
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
            prev = texts.get(rid)
            if prev is None or len(body) > len(prev.get("body") or ""):
                texts[rid] = {
                    "title": title,
                    "body": body,
                    "snapshot": f"broad_capture:{path.name}",
                }
            remaining.discard(rid)
    return texts


def second_rubric_classifier(title: str, body: str) -> dict[str, Any]:
    """Genuinely distinct from Terra/v3: deterministic cue-based structured output.

    Not a human reference. For exploratory disagreement diagnostics only.
    """
    title = title or ""
    body = body or ""
    text = f"{title}\n{body}".strip()
    if not text or (REMOVED.match(title.strip()) and REMOVED.match(body.strip() or "[deleted]")):
        return {
            "relevance": "unassessable",
            "discussion": None,
            "evidence_quote": title.strip() or body.strip() or "[empty]",
            "reason": "No substantive saved title/body remains.",
            "confidence": "high",
            "classifier": "second_rubric_cue_v1",
        }
    updateish = bool(UPDATE_CUES.search(text))
    trouble = bool(TROUBLE_CUES.search(text))
    discontent = bool(DISCONTENT_CUES.search(text))
    if not updateish:
        ev = (title or body)[:120]
        return {
            "relevance": "not_relevant",
            "discussion": None,
            "evidence_quote": ev,
            "reason": "No clear software/firmware update cue in saved title/body.",
            "confidence": "medium",
            "classifier": "second_rubric_cue_v1",
        }
    # Relevant: choose purpose by dominant cues; ambiguous → Unclear
    if trouble and not discontent:
        purpose, reason = "TPS", "Update-related with troubleshooting/diagnostic cues dominant."
    elif discontent and not trouble:
        purpose, reason = "GDS", "Update-related with discontent cues and no clear troubleshooting objective."
    elif trouble and discontent:
        purpose, reason = "TPS", "Mixed cues; troubleshooting language present — default TPS under narrow rule."
    else:
        purpose, reason = "Other", "Update-related without clear TPS or strict GDS primary purpose cues."
    m = UPDATE_CUES.search(text)
    evidence = m.group(0) if m else title[:80]
    return {
        "relevance": "relevant",
        "discussion": purpose,
        "evidence_quote": evidence,
        "reason": reason,
        "confidence": "low",
        "classifier": "second_rubric_cue_v1",
        "cues": {"update": updateish, "trouble": trouble, "discontent": discontent},
    }


def run_phase_b() -> dict[str, Any]:
    dest = OUT_ROOT / "phase_b_classification_diagnostics"
    dest.mkdir(parents=True, exist_ok=True)
    pilot = load_pilot()
    assert len(pilot) == 240, f"expected 240 pilot ids, got {len(pilot)}"
    pilot_ids = [str(r["reddit_id"]) for r in pilot]
    assert len(set(pilot_ids)) == 240
    # Do not redraw: record frozen hash
    pilot_hash = sha256_file(PILOT_IDS)
    strata_hash = sha256_file(PILOT_STRATA)

    # Pilot is excluded-pool: load title/body from broad capture (not 6,425 frame manifest).
    text_index = load_capture_texts_for_ids(set(pilot_ids))
    missing_text = sorted(set(pilot_ids) - set(text_index))
    prot = protected_ids()
    overlap = sorted(set(pilot_ids) & prot)

    # Join v3 predictions
    v3: dict[str, dict[str, Any]] = {}
    if V3_PREDICTIONS.exists():
        for row in load_jsonl(V3_PREDICTIONS):
            rid = str(row.get("reddit_id") or row.get("id") or "")
            if rid in set(pilot_ids):
                v3[rid] = row

    packet = []
    second_preds = []
    restricted_packet = []
    for row in pilot:
        rid = str(row["reddit_id"])
        doc = text_index.get(rid) or {"title": "", "body": "", "snapshot": "missing_capture"}
        title, body = doc.get("title") or "", doc.get("body") or ""
        # Classifier input: no labels/sentiment
        packet.append(
            {
                "reddit_id": rid,
                "prediction_stratum": row.get("prediction_stratum"),
                "period": row.get("period"),
                "author_replies_ge4": row.get("author_replies_ge4"),
                "weight": row.get("inverse_probability_weight"),
                "title_sha256": sha256_text(title),
                "body_sha256": sha256_text(body),
                "title_chars": len(title),
                "body_chars": len(body),
                "text_source": doc.get("snapshot"),
            }
        )
        pred = second_rubric_classifier(title, body)
        second_preds.append({"reddit_id": rid, **pred, "weight": float(row.get("inverse_probability_weight") or 1)})
        restricted_packet.append(
            {
                "reddit_id": rid,
                "title": title,
                "body": body,
                "period": row.get("period"),
                "author_replies_ge4": row.get("author_replies_ge4"),
                "weight": row.get("inverse_probability_weight"),
                "text_source": doc.get("snapshot"),
            }
        )

    write_jsonl(dest / "pilot_240_public_packet.jsonl", packet)
    write_jsonl(RESTRICTED_ROOT / "pilot_240_inference_packet.jsonl", restricted_packet)
    write_jsonl(dest / "second_rubric_predictions.jsonl", second_preds)

    # Baseline distribution of second classifier (unweighted + IPW exploratory)
    rel_counts = Counter(r["relevance"] for r in second_preds)
    disc_counts = Counter(str(r.get("discussion")) for r in second_preds if r["relevance"] == "relevant")
    weighted_relevant = sum(r["weight"] for r in second_preds if r["relevance"] == "relevant")
    weighted_gds = sum(r["weight"] for r in second_preds if r.get("discussion") == "GDS")

    # Compare to v3 where available
    disagree = []
    agree = 0
    compared = 0
    for pred in second_preds:
        rid = pred["reddit_id"]
        if rid not in v3:
            continue
        compared += 1
        v = v3[rid]
        v_rel = None
        # exploratory v3 schema: predicted_label is primary (ai-relevance-exploratory-full)
        if isinstance(v.get("label"), dict):
            v_rel = v["label"].get("relevance") or v["label"].get("final_relevance")
        v_rel = (
            v_rel
            or v.get("predicted_label")
            or v.get("relevance")
            or v.get("final_relevance")
            or v.get("prediction")
        )
        if str(v_rel) == str(pred["relevance"]):
            agree += 1
        else:
            disagree.append(
                {
                    "reddit_id": rid,
                    "second_relevance": pred["relevance"],
                    "v3_relevance": v_rel,
                    "second_discussion": pred.get("discussion"),
                    "v3_model_id": v.get("model_id") or v.get("resolved_model"),
                    "v3_prompt_version": v.get("prompt_version"),
                }
            )

    n_nonempty = sum(1 for r in packet if (r["title_chars"] or 0) + (r["body_chars"] or 0) > 0)
    report = {
        "utc": utcnow(),
        "n_pilot": 240,
        "pilot_ids_sha256": pilot_hash,
        "pilot_strata_sha256": strata_hash,
        "protected_overlap_in_pilot": len(overlap),
        "text_source": "broad_capture_pages_for_excluded_pool_pilot",
        "n_with_capture_text": len(text_index),
        "n_missing_capture_text": len(missing_text),
        "n_nonempty_title_or_body": n_nonempty,
        "second_classifier": "second_rubric_cue_v1",
        "second_relevance_counts": dict(rel_counts),
        "second_discussion_among_relevant": dict(disc_counts),
        "exploratory_ipw_sum_relevant": weighted_relevant,
        "exploratory_ipw_sum_predicted_gds": weighted_gds,
        "v3_predictions_path_exists": V3_PREDICTIONS.exists(),
        "v3_compared_n": compared,
        "v3_agree_n": agree,
        "v3_disagree_n": len(disagree),
        "scientific_status": (
            "AI/rule agreement is NOT accuracy, NOT a verified false-negative rate, "
            "and NOT verified GDS yield. Human review of the 240 is required for those claims."
        ),
        "paid_api": False,
        "full_frame_reclassification": "not_run_awaiting_manu_approval",
    }
    write_json(dest / "DIAGNOSTICS.json", report)
    write_jsonl(dest / "v3_second_disagreements.jsonl", disagree[:500])
    write_json(
        dest / "FINDINGS.md".replace(".md", ".json") if False else dest / "PHASE_B_SUMMARY.json",
        report,
    )
    (dest / "FINDINGS.md").write_text(
        f"""# Phase B findings — classification diagnostics (exploratory)

**UTC:** {report['utc']}

- Fixed pilot N=**240** (not redrawn); protected overlap in pilot: **{len(overlap)}**
- Second local cue-based rubric classifier (`second_rubric_cue_v1`) relevance counts: `{dict(rel_counts)}`
- Among relevant, discussion cue counts: `{dict(disc_counts)}`
- Compared to existing exploratory v3 predictions: n={compared}, agree={agree}, disagree={len(disagree)}
- IPW sums are **model-output distributions only**, not verified yield.

## Blocked / not done
Paid API calls, full-frame reclassification, human validation of the 240.

## Reproduction note
Classifier inputs exclude labels and sentiment. See restricted inference packet (text) and public packet (hashes only).
""",
        encoding="utf-8",
    )
    (dest / "LIMITATIONS.md").write_text(
        """# Phase B limitations

| Limit | Implication |
|---|---|
| Cue classifier ≠ human rubric adjudication | Disagreement ≠ truth |
| No human labels on 240 | Cannot report accuracy or FN rate |
| IPW on model outputs | Not population verified GDS recovery |
| v3 join may be incomplete schema | Compared only where relevance field resolvable |
""",
        encoding="utf-8",
    )
    return report
