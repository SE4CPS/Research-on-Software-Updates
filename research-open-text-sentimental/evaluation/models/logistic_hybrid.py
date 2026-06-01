"""
Logistic regression on TF-IDF text + optional numeric trajectory / subreddit features.
Used for ablation D4 (text + trajectory) and D5 (+ subreddit dummies).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from evaluation.data_loader import TRAJECTORY_FEATURE_COLS
from evaluation.models.base import BaseDiscourseModel


class LogisticHybridModel(BaseDiscourseModel):
    name = "logistic_hybrid"

    def __init__(
        self,
        text_column: str = "text",
        max_features: int = 12_000,
        random_state: int = 42,
        use_trajectory: bool = False,
        use_subreddit: bool = False,
        trajectory_cols: list[str] | None = None,
    ) -> None:
        self.text_column = text_column
        self.max_features = max_features
        self.random_state = random_state
        self.use_trajectory = use_trajectory
        self.use_subreddit = use_subreddit
        self.trajectory_cols = trajectory_cols or list(TRAJECTORY_FEATURE_COLS)
        self._tfidf: TfidfVectorizer | None = None
        self._clf: LogisticRegression | None = None
        self._traj_scaler: StandardScaler | None = None
        self._sub_enc: OneHotEncoder | None = None
        self._traj_cols_used: list[str] = []

    def _build_sparse(self, X: pd.DataFrame, fit: bool) -> Any:
        assert self._tfidf is not None
        X_text = self._tfidf.transform(X[self.text_column].astype(str))
        parts: list[Any] = [X_text]

        if self.use_trajectory and self._traj_cols_used:
            traj = X[self._traj_cols_used].fillna(0.0).to_numpy(dtype=float)
            if fit:
                self._traj_scaler = StandardScaler()
                traj_scaled = self._traj_scaler.fit_transform(traj)
            else:
                assert self._traj_scaler is not None
                traj_scaled = self._traj_scaler.transform(traj)
            parts.append(csr_matrix(traj_scaled))

        if self.use_subreddit:
            sub = X[["subreddit"]].fillna("unknown").astype(str)
            if fit:
                self._sub_enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
                parts.append(self._sub_enc.fit_transform(sub))
            else:
                assert self._sub_enc is not None
                parts.append(self._sub_enc.transform(sub))

        return parts[0] if len(parts) == 1 else hstack(parts, format="csr")

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        self._tfidf = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        self._tfidf.fit(X[self.text_column].astype(str))

        self._traj_cols_used = [c for c in self.trajectory_cols if c in X.columns]
        if self.use_trajectory and not self._traj_cols_used:
            raise ValueError("use_trajectory=True but no trajectory columns in X")

        X_mat = self._build_sparse(X, fit=True)
        sw = compute_sample_weight("balanced", y)
        self._clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=self.random_state,
        )
        self._clf.fit(X_mat, y, sample_weight=sw)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self._clf is not None
        return self._clf.predict(self._build_sparse(X, fit=False))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert self._clf is not None
        return self._clf.predict_proba(self._build_sparse(X, fit=False))

    def get_params(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "text_column": self.text_column,
            "use_trajectory": self.use_trajectory,
            "use_subreddit": self.use_subreddit,
            "trajectory_cols": self._traj_cols_used,
        }
