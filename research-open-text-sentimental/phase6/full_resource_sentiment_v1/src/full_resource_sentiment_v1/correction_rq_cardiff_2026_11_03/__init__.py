"""Correction + RQ1–RQ4 + Cardiff full scoring — 2026-11-03.

Preserves sealed continuation_full_screen_cardiff tree and original 6,425.
AI candidates ≠ confirmed. No manuscript.
"""

RUN_ID = "correction-rq-cardiff-2026-11-03"
SEED = 20261103
N_BOOT = 10000
FROZEN_DEF = "frozen_broad_release_train_package_scores"
DATASET_VERSION = "analytical_union_ai_assisted_v2_corrected_removals_2026-11-03"
CARDIFF_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
CARDIFF_REV = "3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7"

__all__ = [
    "RUN_ID",
    "SEED",
    "N_BOOT",
    "FROZEN_DEF",
    "DATASET_VERSION",
    "CARDIFF_MODEL",
    "CARDIFF_REV",
]
