"""
Base interface for TPS/GDS classifiers (RoBERTa-ready).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class BaseDiscourseModel(ABC):
    """Sklearn-like contract for cross-validation runner."""

    name: str = "base"

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return (n, 2) probabilities [P(GDS), P(TPS)].
        Default: hard labels → near-one-hot.
        """
        pred = self.predict(X)
        out = np.zeros((len(pred), 2), dtype=float)
        out[:, 0] = (pred == 0).astype(float)
        out[:, 1] = (pred == 1).astype(float)
        return out

    def decision_function_tps(self, X: pd.DataFrame) -> np.ndarray:
        """Score for TPS (positive class); used for PR-AUC."""
        proba = self.predict_proba(X)
        return proba[:, 1]

    def get_params(self) -> dict[str, Any]:
        return {"name": self.name}
