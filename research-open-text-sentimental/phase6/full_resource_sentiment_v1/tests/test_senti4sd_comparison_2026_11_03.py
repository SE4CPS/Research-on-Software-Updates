"""Tests for Senti4SD comparison package (no full scoring)."""

from __future__ import annotations

from full_resource_sentiment_v1.senti4sd_comparison_2026_11_03 import LABEL_TO_SIGN
from full_resource_sentiment_v1.senti4sd_comparison_2026_11_03.common import (
    DATASET_JSONL,
    EXPECTED_JSONL,
    JAVA_BIN,
    LENGTH_OUT,
    PYSENTI,
    SEALED_AUDIT,
    VENV_SENTI,
    sha256_file,
)


def test_frozen_dataset_hash():
    assert sha256_file(DATASET_JSONL) == EXPECTED_JSONL


def test_length_sensitivity_seal_preserved():
    assert (LENGTH_OUT / "OUTPUT_HASHES_FINAL.json").exists()
    assert (SEALED_AUDIT / "OUTPUT_HASHES_FINAL.json").exists()


def test_senti4sd_assets_present():
    assert JAVA_BIN.exists()
    assert (PYSENTI / "Senti4SD.model").exists()
    assert (PYSENTI / "java/dsm.bin").exists()
    assert (PYSENTI / "java/Senti4SD-fast.jar").exists()
    assert (VENV_SENTI / "bin/python").exists()
    assert (PYSENTI / "java/dsm.bin").stat().st_size > 100_000_000


def test_polarity_sign_map():
    assert LABEL_TO_SIGN["negative"] == -1.0
    assert LABEL_TO_SIGN["neutral"] == 0.0
    assert LABEL_TO_SIGN["positive"] == 1.0
