"""Constants for the approved 200-case TPS/GDS validation draw."""

from __future__ import annotations

from pathlib import Path

BASE_SEED = "20261102"
RUN_ID = "validation-200-2026-11-02"

# Domain × predicted-category strata. Names are frozen into seed derivation.
STRATA: dict[str, dict[str, int]] = {
    "fresh_relevant_tps": {"n": 45, "expected_N": 1327},
    "fresh_relevant_gds": {"n": 50, "expected_N": 153},
    "fresh_relevant_other": {"n": 30, "expected_N": 1345},
    "fresh_not_relevant": {"n": 25, "expected_N": 1028},
    "fresh_unclear_unassessable": {"n": 10, "expected_N": 31},
    "covered_relevant_tps": {"n": 10, "expected_N": 1725},
    "covered_relevant_gds": {"n": 10, "expected_N": 89},
    "covered_relevant_other": {"n": 5, "expected_N": 297},
    "covered_not_relevant": {"n": 8, "expected_N": 324},
    "covered_unclear_unassessable": {"n": 7, "expected_N": 106},
}

EXPECTED_FRESH_N = 3884
EXPECTED_COVERED_N = 2541
EXPECTED_FRAME_N = 6425
EXPECTED_DRAW_FRESH = 160
EXPECTED_DRAW_COVERED = 40
EXPECTED_DRAW_TOTAL = 200

CATEGORIES = (
    "relevant_tps",
    "relevant_gds",
    "relevant_other",
    "not_relevant",
    "unclear_unassessable",
)

RELEVANCE_A = ("relevant", "not_relevant", "unclear", "unassessable")
DISCUSSION_B = ("TPS", "GDS", "Other", "Unclear", "not_applicable")
CONFIDENCE = ("high", "medium", "low")
RECOGNIZED = ("yes", "no", "unsure")


def phase6_root() -> Path:
    return Path(__file__).resolve().parents[3]


def repo_root() -> Path:
    return phase6_root().parents[1]
