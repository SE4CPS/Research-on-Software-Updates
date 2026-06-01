"""
VADER sentiment helpers (aligned with enhanced_automated_sentiment_analysis.py).
"""

from __future__ import annotations

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_COMPOUND_POS = 0.05
_COMPOUND_NEG = -0.05


def compound_score(analyzer: SentimentIntensityAnalyzer, text: str) -> float:
    if not text or not str(text).strip():
        return 0.0
    return float(analyzer.polarity_scores(str(text))["compound"])


def compound_label(compound: float) -> str:
    if compound >= _COMPOUND_POS:
        return "Positive"
    if compound <= _COMPOUND_NEG:
        return "Negative"
    return "Neutral"
