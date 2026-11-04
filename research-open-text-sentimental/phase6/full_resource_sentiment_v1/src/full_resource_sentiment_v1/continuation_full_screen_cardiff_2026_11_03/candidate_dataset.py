"""New versioned analytical union: original 6,425 preserved + AI-assisted candidates."""

from __future__ import annotations

from collections import Counter
from typing import Any

from full_resource_sentiment_v1.continuation_full_screen_cardiff_2026_11_03.common import (
    DATASET_JSONL,
    OUT_ROOT,
    POST_SCORES,
    RESTRICTED_ROOT,
    load_jsonl,
    protected_ids_only,
    sha256_file,
    utcnow,
    write_json,
    write_jsonl,
)

VERSION = "analytical_union_ai_assisted_v1_2026-11-03"


def run_candidate_dataset() -> dict[str, Any]:
    dest = OUT_ROOT / "phase2_candidate_dataset"
    dest.mkdir(parents=True, exist_ok=True)
    prot = protected_ids_only()

    screen_path = RESTRICTED_ROOT / "full_frame_screen_30998.jsonl"
    screened = load_jsonl(screen_path)
    post_meta = {str(r["reddit_id"]): r for r in load_jsonl(POST_SCORES)}

    original = load_jsonl(DATASET_JSONL)
    assert len(original) == 6425
    orig_ids = {str(r["reddit_id"]) for r in original}
    assert not (orig_ids & prot), "protected must not appear in original frame"

    # Preserve original rows verbatim + provenance wrapper fields
    union_rows = []
    for r in original:
        rid = str(r["reddit_id"])
        union_rows.append(
            {
                "dataset_version": VERSION,
                "record_origin": "original_6425_preserved",
                "reddit_id": rid,
                "final_relevance": r.get("final_relevance"),
                "final_discussion": r.get("final_discussion"),
                "decision_source": r.get("decision_source"),
                "decision_confidence": r.get("decision_confidence"),
                "decision_status": r.get("decision_status"),
                "selection_stratum": r.get("selection_stratum"),
                "sentiment_definition": r.get("sentiment_definition"),
                "model_version": r.get("model_version"),
                "label_overwritten": False,
                "ai_assisted_candidate": False,
                "human_validated": None,  # unknown / mixed historical — not claimed here
                "confirmed_gds_addition": False,
                "n_author_replies_nonempty": (r.get("sentiment") or {}).get("n_author_replies_nonempty"),
                "n_community_replies_nonempty": (r.get("sentiment") or {}).get("n_community_replies_nonempty"),
                "capture_source": (post_meta.get(rid) or {}).get("source"),
                "capture_page": (post_meta.get(rid) or {}).get("capture_page"),
                "created_month": (post_meta.get(rid) or {}).get("created_month"),
                "missing_post_scores": rid not in post_meta,
                "note": "Original label preserved; not modified by AI screen",
            }
        )

    # Selected AI-assisted records: TPS/GDS candidates only (strict definition retained)
    selected = [s for s in screened if s.get("recovery_status") == "candidate_ai_only"]
    overlap_with_orig = [s for s in selected if s["reddit_id"] in orig_ids]
    assert len(overlap_with_orig) == 0, "screened excluded must not overlap original"

    n_missing_text = 0
    n_missing_replies = 0
    for s in selected:
        rid = s["reddit_id"]
        if rid in prot:
            continue
        if int(s.get("text_chars") or 0) == 0:
            n_missing_text += 1
        if int(s.get("n_author") or 0) == 0 and int(s.get("n_community") or 0) == 0:
            n_missing_replies += 1
        union_rows.append(
            {
                "dataset_version": VERSION,
                "record_origin": "ai_assisted_screen_candidate",
                "reddit_id": rid,
                "final_relevance": "relevant",  # stage-A AI
                "final_discussion": s.get("stageB_purpose_pred"),
                "decision_source": "tfidf_two_stage_full_frame_v1_seed_20261103",
                "decision_confidence": s.get("stageB_purpose_prob"),
                "decision_status": "ai_candidate_unconfirmed",
                "selection_stratum": f"excluded_screen_{s.get('author_depth_bucket')}",
                "sentiment_definition": None,
                "model_version": s.get("screen_version"),
                "label_overwritten": False,
                "ai_assisted_candidate": True,
                "human_validated": False,
                "confirmed_gds_addition": False,
                "stageA_prob_relevant": s.get("stageA_prob_relevant"),
                "stageB_purpose_prob": s.get("stageB_purpose_prob"),
                "n_author_replies_nonempty": s.get("n_author"),
                "n_community_replies_nonempty": s.get("n_community"),
                "capture_source": s.get("source"),
                "capture_page": s.get("capture_page"),
                "created_month": s.get("created_month"),
                "created_utc": s.get("created_utc"),
                "missing_text": int(s.get("text_chars") or 0) == 0,
                "missing_timestamp": s.get("created_utc") in (None, ""),
                "in_prior_9882_screen": s.get("in_prior_9882_screen"),
                "in_fixed_pilot_240": s.get("in_fixed_pilot_240"),
                "note": "AI-assisted candidate only — NOT human-confirmed; NOT prevalence estimate",
            }
        )

    write_jsonl(RESTRICTED_ROOT / f"{VERSION}.jsonl", union_rows)

    # Public summary without full text
    by_origin = Counter(r["record_origin"] for r in union_rows)
    by_disc = Counter(
        (r["record_origin"], r.get("final_discussion")) for r in union_rows if r.get("final_discussion")
    )
    ai_gds = sum(
        1
        for r in union_rows
        if r["record_origin"] == "ai_assisted_screen_candidate" and r.get("final_discussion") == "GDS"
    )
    ai_tps = sum(
        1
        for r in union_rows
        if r["record_origin"] == "ai_assisted_screen_candidate" and r.get("final_discussion") == "TPS"
    )
    orig_gds = sum(
        1
        for r in union_rows
        if r["record_origin"] == "original_6425_preserved" and r.get("final_discussion") == "GDS"
    )
    orig_tps = sum(
        1
        for r in union_rows
        if r["record_origin"] == "original_6425_preserved" and r.get("final_discussion") == "TPS"
    )

    report = {
        "utc": utcnow(),
        "dataset_version": VERSION,
        "original_6425_path": str(DATASET_JSONL),
        "original_6425_sha256": sha256_file(DATASET_JSONL),
        "original_untouched": True,
        "n_union_rows": len(union_rows),
        "by_record_origin": dict(by_origin),
        "n_original_preserved": by_origin["original_6425_preserved"],
        "n_ai_assisted_added": by_origin["ai_assisted_screen_candidate"],
        "overlap_reddit_ids_original_vs_ai": 0,
        "protected_in_union": sum(1 for r in union_rows if r["reddit_id"] in prot),
        "original_tps_gds": {"TPS": orig_tps, "GDS": orig_gds},
        "ai_candidate_tps_gds": {"TPS": ai_tps, "GDS": ai_gds},
        "ai_gds_not_confirmed_prevalence": True,
        "missingness": {
            "ai_missing_text": n_missing_text,
            "ai_zero_author_and_community_replies": n_missing_replies,
            "ai_missing_timestamp": sum(
                1
                for r in union_rows
                if r["record_origin"] == "ai_assisted_screen_candidate" and r.get("missing_timestamp")
            ),
        },
        "by_origin_discussion": {f"{a}|{b}": n for (a, b), n in by_disc.items()},
        "strict_gds_definition": "unchanged — purpose label GDS from frozen rubric / AI stageB only as candidate",
        "scientific_status": (
            "Versioned AI-assisted analytical pool for exploratory work; "
            "do not cite AI GDS counts as confirmed prevalence or human accuracy"
        ),
        "restricted_file": f"{VERSION}.jsonl",
    }
    write_json(dest / "CANDIDATE_DATASET_MANIFEST.json", report)
    (dest / "LIMITATIONS.md").write_text(
        """# Candidate dataset limitations

- AI-assisted TPS/GDS rows are **candidates only**, not human-confirmed.
- Original 6,425 rows are preserved byte-for-byte in meaning (labels not overwritten).
- Do not treat AI GDS counts as prevalence or as validation of the purpose model.
- Protected evaluation isolation retained (protected IDs not in excluded screen).
- Strict GDS definition unchanged.
""",
        encoding="utf-8",
    )
    return report
