"""v3 freeze stays distinct from the locked v1 prompt. No live API."""

from __future__ import annotations

import os

from full_resource_sentiment_v1.relevance_classify.scope_primary import SCOPE_VERSION
from full_resource_sentiment_v1.relevance_classify.spec import (
    PROMPT_VERSION,
    PROMPT_VERSION_ENV,
    PROMPT_VERSION_V3,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_V3,
    active_prompt_sha256,
    active_prompt_version,
    active_system_prompt,
    prompt_sha256,
    prompt_v3_sha256,
    schema_sha256,
)
from full_resource_sentiment_v1.relevance_classify.validation_weighted import score_weighted_validation

LOCKED_V1_PROMPT = "fb30113eb732f320018bc78420159feda1298899ac070991ca3b6cc9dcf67f85"
LOCKED_SCHEMA = "62bead564f30ff68d774a2ecaf766bcdc055d7d92374cda8b23e58735cd1f4bb"


def test_v1_prompt_hash_unchanged():
    assert prompt_sha256() == LOCKED_V1_PROMPT
    assert schema_sha256() == LOCKED_SCHEMA
    assert active_prompt_version() == PROMPT_VERSION
    assert active_system_prompt() is SYSTEM_PROMPT


def test_v3_prompt_is_frozen_and_distinct(monkeypatch):
    assert SYSTEM_PROMPT_V3 != SYSTEM_PROMPT
    digest = prompt_v3_sha256()
    assert len(digest) == 64
    assert digest != LOCKED_V1_PROMPT
    assert digest == prompt_v3_sha256()
    assert SCOPE_VERSION == "primary_update_change_v3"
    monkeypatch.setenv(PROMPT_VERSION_ENV, "v3")
    assert active_prompt_version() == PROMPT_VERSION_V3
    assert active_prompt_sha256() == digest
    assert "EXISTING software" in active_system_prompt()
    monkeypatch.delenv(PROMPT_VERSION_ENV, raising=False)
    assert active_prompt_sha256() == LOCKED_V1_PROMPT
    assert os.environ.get(PROMPT_VERSION_ENV) != "v3"


def test_weighted_eval_uses_inclusion_probabilities():
    rows = [
        {
            "case_code": "V-000001",
            "stratum": "october_2026",
            "ai_selected": True,
            "final_label": "relevant",
            "inclusion_probability": 0.5,
        },
        {
            "case_code": "V-000002",
            "stratum": "october_2026",
            "ai_selected": True,
            "final_label": "not_relevant",
            "inclusion_probability": 0.5,
        },
        {
            "case_code": "V-000003",
            "stratum": "older",
            "ai_selected": False,
            "final_label": "relevant",
            "inclusion_probability": 0.25,
        },
        {
            "case_code": "V-000004",
            "stratum": "older",
            "ai_selected": False,
            "final_label": "not_relevant",
            "inclusion_probability": 0.25,
        },
    ]
    scored = score_weighted_validation(rows)
    assert scored["unweighted_tp"] == 1
    assert scored["unweighted_fp"] == 1
    assert scored["unweighted_fn"] == 1
    assert scored["unweighted_tn"] == 1
    assert scored["weighted_precision"] == 0.5
    # selected relevant weight 2; human relevant weights 2 + 4 = 6 → recall 1/3
    assert abs(scored["weighted_recall"] - (2.0 / 6.0)) < 1e-12
    assert scored["used_protected_outcomes"] is False
