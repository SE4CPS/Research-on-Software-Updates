from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from evaluation.models.base import BaseDiscourseModel


class LogisticRegressionModel(BaseDiscourseModel):
    name = "logistic_regression"

    def __init__(
        self,
        text_column: str = "text",
        max_features: int = 12_000,
        random_state: int = 42,
    ) -> None:
        self.text_column = text_column
        self.max_features = max_features
        self.random_state = random_state
        self.pipe_: Pipeline | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        sw = compute_sample_weight("balanced", y)
        self.pipe_ = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=self.max_features,
                        ngram_range=(1, 2),
                        min_df=1,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=self.random_state,
                    ),
                ),
            ]
        )
        texts = X[self.text_column].astype(str)
        self.pipe_.fit(texts, y, clf__sample_weight=sw)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self.pipe_ is not None
        return self.pipe_.predict(X[self.text_column].astype(str))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert self.pipe_ is not None
        return self.pipe_.predict_proba(X[self.text_column].astype(str))

    def get_params(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "text_column": self.text_column,
            "max_features": self.max_features,
        }
