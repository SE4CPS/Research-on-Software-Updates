"""Pilot 240 reporting correction: keyword heuristics, not model truth / yield."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from full_resource_sentiment_v1.master_research_corrections_2026_11_03.common import (
    ORIG_OUT,
    OUT_ROOT,
    PILOT_IDS,
    PILOT_STRATA,
    sha256_file,
    utcnow,
    write_json,
)


def run_pilot_reporting() -> dict[str, Any]:
    dest = OUT_ROOT / "pilot_reporting"
    dest.mkdir(parents=True, exist_ok=True)
    diag_path = ORIG_OUT / "phase_b_classification_diagnostics/DIAGNOSTICS.json"
    diag = json.loads(diag_path.read_text(encoding="utf-8")) if diag_path.exists() else {}

    report = {
        "utc": utcnow(),
        "pilot_ids_path": str(PILOT_IDS),
        "pilot_ids_sha256": sha256_file(PILOT_IDS),
        "pilot_strata_sha256": sha256_file(PILOT_STRATA),
        "n_pilot": diag.get("n_pilot", 240),
        "ids_redrawn": False,
        "weights_redrawn": False,
        "protected_overlap_in_pilot": diag.get("protected_overlap_in_pilot"),
        "method_label": "keyword_cue_heuristic_second_rubric_cue_v1",
        "not_independent_model_truth": True,
        "not_human_reference": True,
        "relevance_counts_heuristic": diag.get("second_relevance_counts"),
        "discussion_among_heuristic_relevant": diag.get("second_discussion_among_relevant"),
        "v3_agree_n": diag.get("v3_agree_n"),
        "v3_disagree_n": diag.get("v3_disagree_n"),
        "v3_compared_n": diag.get("v3_compared_n"),
        "cue_predicted_gds_raw_count": (diag.get("second_discussion_among_relevant") or {}).get("GDS"),
        "exploratory_ipw_sum_predicted_gds": diag.get("exploratory_ipw_sum_predicted_gds"),
        "yield_claim_status": (
            "REJECTED_AS_YIELD_ESTIMATE — the weighted 66.8 figure derives from a single "
            "cue-labeled GDS case and is not a reliable recoverable-GDS yield estimate."
        ),
        "accuracy_or_fn_claim_status": (
            "NOT_ESTIMATED — no human labels on the 240; cue↔v3 agreement is method disagreement only."
        ),
        "paid_api": False,
        "full_frame_reclassification": "not_run",
        "source_diagnostics": str(diag_path),
    }
    write_json(dest / "PILOT_REPORTING.json", report)
    (dest / "FINDINGS.md").write_text(
        f"""# Pilot 240 reporting correction

**UTC:** {report['utc']}  
Pilot IDs/weights **fixed** (sha `{report['pilot_ids_sha256']}`).

## Correct labels for outputs
- `second_rubric_cue_v1` = **keyword/cue heuristic**, not an independent contextual model and not human truth.
- Cue vs v3: agree={report['v3_agree_n']} / disagree={report['v3_disagree_n']} of {report['v3_compared_n']} — **disagreement diagnostics only**.
- Cue-predicted GDS raw count: **{report['cue_predicted_gds_raw_count']}** (typically one case).
- Weighted sum 66.8: **{report['yield_claim_status']}**

## Forbidden claims (this packet)
- Human accuracy / false-negative rate on the 240
- Verified GDS population yield from cue IPW
- Paid inference or full-frame reclassification (not run)
""",
        encoding="utf-8",
    )
    (dest / "LIMITATIONS.md").write_text(
        """# Pilot reporting limitations

- Heuristic cues miss context and over/under-trigger on update vocabulary.
- Existing v3 labels are also AI exploratory predictions.
- Human review of the 240 remains required for accuracy/FN/yield.
""",
        encoding="utf-8",
    )
    return report
