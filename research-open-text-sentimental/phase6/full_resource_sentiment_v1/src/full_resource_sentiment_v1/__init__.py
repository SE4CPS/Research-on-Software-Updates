"""Workstream 1: normalization, package adapter, historical 542 comparison."""

from .records import (
    audit_normalized_records,
    build_normalized_record,
    choose_analysis_snapshot,
    iter_normalized_records,
)

__version__ = "0.1.0"

__all__ = [
    "audit_normalized_records",
    "build_normalized_record",
    "choose_analysis_snapshot",
    "iter_normalized_records",
]
