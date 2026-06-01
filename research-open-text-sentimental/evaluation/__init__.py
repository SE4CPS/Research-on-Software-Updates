"""
ICMLA 2026 — unified classification evaluation (TPS vs GDS).

Entry point: scripts/run_evaluation.py
"""

from evaluation.config import EvaluationConfig
from evaluation.runner import run_cross_validation

__all__ = ["EvaluationConfig", "run_cross_validation"]
