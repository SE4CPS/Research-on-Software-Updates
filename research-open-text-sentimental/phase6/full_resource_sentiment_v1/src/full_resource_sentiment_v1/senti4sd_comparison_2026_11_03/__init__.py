"""Senti4SD comparison against VADER/TextBlob/Cardiff on length-sensitivity units."""

RUN_ID = "senti4sd-comparison-2026-11-03"
SEED = 20261103
N_BOOT = 10000
LABEL_TO_SIGN = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}

__all__ = ["RUN_ID", "SEED", "N_BOOT", "LABEL_TO_SIGN"]
