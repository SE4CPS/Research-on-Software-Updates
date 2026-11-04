"""Continuation after Codex exact-join acceptance — 2026-11-03.

Preserves sealed exact_join_repair tree. Full-frame screen, candidate pool,
cohort reporting fix, Cardiff canary/pilot. No manuscript; AI ≠ confirmed.
"""

RUN_ID = "continuation-full-screen-cardiff-2026-11-03"
SEED = 20261103
N_BOOT = 5000
FROZEN_DEF = "frozen_broad_release_train_package_scores"
CARDIFF_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
CARDIFF_REV = "3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7"

__all__ = ["RUN_ID", "SEED", "N_BOOT", "FROZEN_DEF", "CARDIFF_MODEL", "CARDIFF_REV"]
