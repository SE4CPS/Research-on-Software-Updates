"""
VADER + technical-cue rule baseline (matches train_vader_baseline.py).
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from evaluation.models.base import BaseDiscourseModel

_TECH_CUES = re.compile(
    r"\b("
    r"error|bug|bugs|fix|fixed|fixing|crash|crashed|crashing|failed|failing|failure|"
    r"exception|traceback|stack\s*trace|compile|compiling|build|building|install|"
    r"installation|uninstall|patch|patches|regression|reproduce|repro|workaround|"
    r"broken|not\s+working|doesn'?t\s+work|won'?t\s+work|does\s+not\s+work|"
    r"issue|issues|problem|problems|solve|solved|solution|debug|debugging|"
    r"version|update\s+broke|after\s+update|after\s+updating|upgrade|downgrade|"
    r"dependency|dependencies|config|configuration|logs?|terminal|command|kernel|"
    r"docker|container|npm|pip|gradle|cmake|build\s+error|runtime\s+error|"
    r"how\s+do\s+i|how\s+to|help\s+with|anyone\s+know|stuck\s+on"
    r")\b",
    re.IGNORECASE,
)


def has_technical_cues(text: str) -> bool:
    return bool(text and _TECH_CUES.search(text))


def vader_rule_predict(
    text_raw: str,
    analyzer: SentimentIntensityAnalyzer,
    neutral_band: float,
    rule_style: str,
) -> tuple[int, float]:
    text_raw = text_raw or ""
    compound = float(analyzer.polarity_scores(text_raw)["compound"])
    neutral = -neutral_band <= compound <= neutral_band
    strong_emotion = abs(compound) > neutral_band
    tech = has_technical_cues(text_raw)

    if rule_style == "technical_first":
        if compound < -0.6 and not tech:
            return 0, compound
        if tech:
            return 1, compound
        if strong_emotion:
            return 0, compound
        return 0, compound

    if strong_emotion:
        return 0, compound
    if neutral and tech:
        return 1, compound
    return 0, compound


class VaderRulesModel(BaseDiscourseModel):
    """Non-trainable baseline; fit() stores analyzer only."""

    name = "vader_rules"

    def __init__(
        self,
        text_column: str = "text_raw",
        neutral_band: float = 0.15,
        rule_style: str = "technical_first",
    ) -> None:
        self.text_column = text_column
        self.neutral_band = neutral_band
        self.rule_style = rule_style
        self.analyzer_: SentimentIntensityAnalyzer | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        self.analyzer_ = SentimentIntensityAnalyzer()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self.analyzer_ is not None
        out = []
        for t in X[self.text_column].astype(str):
            pred, _ = vader_rule_predict(
                t, self.analyzer_, self.neutral_band, self.rule_style
            )
            out.append(pred)
        return np.array(out, dtype=int)

    def decision_function_tps(self, X: pd.DataFrame) -> np.ndarray:
        """Continuous score in [0,1] for PR-AUC."""
        assert self.analyzer_ is not None
        scores = []
        for t in X[self.text_column].astype(str):
            pred, compound = vader_rule_predict(
                t, self.analyzer_, self.neutral_band, self.rule_style
            )
            tech = has_technical_cues(t)
            base = (compound + 1.0) / 2.0
            if pred == 1:
                scores.append(0.5 + 0.5 * base * (1.2 if tech else 1.0))
            else:
                scores.append(0.5 - 0.5 * abs(compound))
        return np.clip(np.array(scores, dtype=float), 0.0, 1.0)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        s = self.decision_function_tps(X)
        return np.column_stack([1.0 - s, s])

    def get_params(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "text_column": self.text_column,
            "neutral_band": self.neutral_band,
            "rule_style": self.rule_style,
        }
