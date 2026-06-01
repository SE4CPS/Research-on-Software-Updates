from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.models.base import BaseDiscourseModel


class MajorityModel(BaseDiscourseModel):
    name = "majority"

    def __init__(self) -> None:
        self.majority_class_: int = 0

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        values, counts = np.unique(y, return_counts=True)
        self.majority_class_ = int(values[np.argmax(counts)])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.majority_class_, dtype=int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        out = np.zeros((n, 2), dtype=float)
        out[:, self.majority_class_] = 1.0
        return out
