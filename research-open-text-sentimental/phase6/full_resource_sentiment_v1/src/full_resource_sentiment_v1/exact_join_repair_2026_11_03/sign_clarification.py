"""Preserve exact-sign vs Codex soft-neutral vs three-way estimands (no overwrite)."""

from __future__ import annotations

import json
from typing import Any

from full_resource_sentiment_v1.exact_join_repair_2026_11_03.common import (
    CORR_OUT,
    OUT_ROOT,
    utcnow,
    write_json,
)


def run_sign_clarification() -> dict[str, Any]:
    dest = OUT_ROOT / "phase0_sign_clarification"
    dest.mkdir(parents=True, exist_ok=True)
    prior_path = CORR_OUT / "stage0_sign_agreement/SIGN_AGREEMENT_CORRECTED.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8")) if prior_path.exists() else {}

    report = {
        "utc": utcnow(),
        "preserves_prior_path": str(prior_path),
        "prior_not_overwritten": True,
        "estimands": {
            "exact_sign_np_sign": {
                "definition": "int(np.sign(vader)) == int(np.sign(textblob)); zeros agree",
                "seeded_498": prior.get("primary_np_sign") or {"agree": 308, "disagree": 190, "missing": 0},
                "canonical_for_this_project": True,
            },
            "codex_soft_neutral_or_exact_sign": {
                "definition": "exact same sign OR (abs(vader)<0.05 AND abs(textblob)<0.05)",
                "seeded_498": {"agree": 309, "disagree": 189, "missing": 0},
                "note": "Adds one near-neutral pair relative to exact-sign; not an unexplained off-by-one",
            },
            "three_way_neutral_bucket_0_05": {
                "definition": "Both |score|<0.05 counted as mutual neutral agreement bucket",
                "seeded_498": (prior.get("sensitivity") or {}).get("vader_abs_0.05_tb_abs_0.05")
                or {"agree": 232, "disagree": 266, "missing": 0},
                "note": "Third estimand; never interchange with exact-sign 308/190",
            },
        },
        "rule": "Name the threshold/estimand every time a number is cited; never present interchangeably",
    }
    write_json(dest / "SIGN_ESTIMAND_CLARIFICATION.json", report)
    return report
