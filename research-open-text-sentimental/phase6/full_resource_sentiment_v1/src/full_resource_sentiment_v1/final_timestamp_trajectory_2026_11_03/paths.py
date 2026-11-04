"""Paths for final timestamp trajectory package."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results"
RESTRICTED = ROOT / "restricted"

OUT = RESULTS / "final_timestamp_trajectory_2026-11-03"
OUT_RESTRICTED = RESTRICTED / "final_timestamp_trajectory_2026-11-03"

CORRECTIONS = RESULTS / "pre_manuscript_evidence_corrections_2026-11-03"
PLANNING = ROOT / "planning" / "final_comprehensive_consolidation_2026-11-03"

POST_SCORES = RESULTS / "broad_sentiment_2026-10-22-main" / "post_scores.jsonl"
THREAD_METRICS = RESULTS / "broad_sentiment_2026-10-22-main" / "thread_metrics.jsonl"
COMMENT_SCORES = RESULTS / "broad_sentiment_2026-10-22-main" / "comment_scores.jsonl"
CAPTURE_DIR = RESTRICTED / "broad_release_train" / "2026-10-22-main"

FULL_DATASET = RESULTS / "tps_gds_full_ai_dataset_2026-11-02" / "full_dataset.jsonl"
UNION_AI = (
    RESTRICTED
    / "correction_rq_cardiff_2026-11-03"
    / "analytical_union_ai_assisted_v2_corrected_removals_2026-11-03.jsonl"
)

FOUR_MODEL_AI = (
    RESTRICTED
    / "pre_manuscript_evidence_reconciliation_2026-11-03"
    / "ai_assisted_v2_four_model_matched_pairs.jsonl"
)
FOUR_MODEL_FROZEN = (
    RESTRICTED
    / "pre_manuscript_evidence_reconciliation_2026-11-03"
    / "original_frozen_four_model_matched_pairs.jsonl"
)

PROTECTED_IDS = (
    RESTRICTED / "relevance_audit_sample_2026-10-28" / "excluded_protected_ids.json"
)

TEXT_MANIFEST = RESTRICTED / "master_research_2026-11-03" / "text_thread_manifest_restricted.jsonl"

FROZEN_DEF = "frozen_broad_release_train_package_scores"

# Expected Stage-0 gates (fail if disagree; do not hard-code patches)
EXPECTED = {
    "frozen": {
        "TPS": 3028,
        "GDS": 231,
        "author_ge1": {"TPS": 956, "GDS": 98},
        "author_ge2": {"TPS": 437, "GDS": 66},
        "both_roles": {"TPS": 934, "GDS": 98},
    },
    "ai": {
        "TPS": 11054,
        "GDS": 999,
        "author_ge1": {"TPS": 5332, "GDS": 610},
        "author_ge2": {"TPS": 2529, "GDS": 410},
        "both_roles": {"TPS": 5271, "GDS": 608},
    },
    "union_n": 18478,
    "four_model_n": 6996,
    "protected_n": 693,
}
