"""
Trajectory statistics: slope, volatility, early-late shift, divergence.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


def linear_slope(values: list[float]) -> float:
    """OLS slope over index 0..n-1; 0 if n < 2."""
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    num = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
    den = sum((i - x_mean) ** 2 for i in range(n))
    if den == 0:
        return 0.0
    return num / den


def mean_abs_step(values: list[float]) -> float:
    """Volatility: mean |delta| between consecutive compounds."""
    if len(values) < 2:
        return 0.0
    return sum(abs(values[i + 1] - values[i]) for i in range(len(values) - 1)) / (
        len(values) - 1
    )


def early_late_shift(values: list[float]) -> float:
    """Late-half mean minus early-half mean; 0 if len < 4."""
    if len(values) < 4:
        return 0.0
    mid = len(values) // 2
    early = sum(values[:mid]) / mid
    late = sum(values[mid:]) / (len(values) - mid)
    return late - early


def mean_or_zero(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


@dataclass
class TrajectoryMetrics:
    reddit_id: str
    subreddit: str
    corrected_label: int
    url: str

    title_compound: float
    n_author_replies: int
    n_community_comments: int
    n_comments_total: int

    author_mean_sentiment: float
    community_mean_sentiment: float
    sentiment_divergence: float
    author_more_negative: bool

    author_trend: float
    community_trend: float
    author_volatility: float
    community_volatility: float
    author_early_late_shift: float
    community_early_late_shift: float

    trajectory_eligible: bool
    trajectory_notes: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["author_more_negative"] = bool(d["author_more_negative"])
        return d
