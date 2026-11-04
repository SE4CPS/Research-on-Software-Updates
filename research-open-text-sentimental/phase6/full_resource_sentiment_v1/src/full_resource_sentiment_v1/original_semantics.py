"""Reconstruct original C1-prime semantics without writing to data/c1_prime."""

from __future__ import annotations

from typing import Any

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .comments import original_string_sort


def compound_score(analyzer: SentimentIntensityAnalyzer, text: str) -> float:
    if not text or not str(text).strip():
        return 0.0
    return float(analyzer.polarity_scores(str(text))["compound"])


def linear_slope(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    if denominator == 0:
        return 0.0
    return numerator / denominator


def mean_abs_step(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return sum(abs(values[i + 1] - values[i]) for i in range(len(values) - 1)) / (len(values) - 1)


def early_late_shift(values: list[float]) -> float:
    if len(values) < 4:
        return 0.0
    mid = len(values) // 2
    early = sum(values[:mid]) / mid
    late = sum(values[mid:]) / (len(values) - mid)
    return late - early


def mean_or_zero(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def analyze_post_trajectories_original(
    post: dict[str, Any],
    analyzer: SentimentIntensityAnalyzer,
) -> dict[str, Any]:
    """Match trajectory/build.py:analyze_post_trajectories, plus eligibility flags."""
    title = post.get("title") or ""
    title_compound = compound_score(analyzer, title)
    author_name = (post.get("author") or "").strip()
    raw_comments = post.get("comments") or []
    comments = original_string_sort(list(raw_comments)) if isinstance(raw_comments, list) else []

    author_traj: list[float] = []
    community_traj: list[float] = []
    body = (post.get("author_description") or "").strip()
    used_body = False
    if body:
        author_traj.append(compound_score(analyzer, body))
        used_body = True

    n_empty_skipped = 0
    for comment in comments:
        comment_body = (comment.get("body") or "").strip()
        if not comment_body:
            n_empty_skipped += 1
            continue
        compound = compound_score(analyzer, comment_body)
        is_author = bool(comment.get("is_submitter")) or (
            bool(author_name) and (comment.get("author") or "").strip() == author_name
        )
        if is_author:
            author_traj.append(compound)
        else:
            community_traj.append(compound)

    title_fallback = False
    if not author_traj:
        author_traj.append(title_compound)
        title_fallback = True

    return {
        "author_traj": author_traj,
        "community_traj": community_traj,
        "title_compound": title_compound,
        "used_body_as_first_author_point": used_body,
        "title_fallback": title_fallback,
        "n_empty_comment_bodies_skipped": n_empty_skipped,
        "n_author_track_points": len(author_traj),
        "n_community_track_points": len(community_traj),
        "author_mean_unrounded": mean_or_zero(author_traj),
        "community_mean_unrounded": mean_or_zero(community_traj),
        "community_mean_is_observed": len(community_traj) > 0,
        "author_mean_is_observed": True,
        "author_mean_observed_without_fallback": (not title_fallback),
    }


def compute_metrics_row_original(
    reddit_id: str,
    subreddit: str,
    corrected_label: int,
    url: str,
    analysis: dict[str, Any],
    min_author: int = 1,
    min_community: int = 1,
) -> dict[str, Any]:
    author_traj = analysis["author_traj"]
    community_traj = analysis["community_traj"]
    title_compound = analysis["title_compound"]
    author_mean = mean_or_zero(author_traj)
    community_mean = mean_or_zero(community_traj)
    divergence = abs(author_mean - community_mean)
    notes: list[str] = []
    eligible = True
    if len(author_traj) < min_author:
        notes.append(f"author_replies<{min_author}")
        eligible = False
    if len(community_traj) < min_community:
        notes.append(f"community_comments<{min_community}")
        eligible = False
    return {
        "reddit_id": reddit_id,
        "subreddit": subreddit,
        "corrected_label": int(corrected_label),
        "url": url,
        "title_compound": round(title_compound, 4),
        "n_author_replies": len(author_traj),
        "n_community_comments": len(community_traj),
        "n_comments_total": len(author_traj) + len(community_traj),
        "author_mean_sentiment": round(author_mean, 4),
        "community_mean_sentiment": round(community_mean, 4),
        "sentiment_divergence": round(divergence, 4),
        "author_more_negative": author_mean < community_mean,
        "author_trend": round(linear_slope(author_traj), 4),
        "community_trend": round(linear_slope(community_traj), 4),
        "author_volatility": round(mean_abs_step(author_traj), 4),
        "community_volatility": round(mean_abs_step(community_traj), 4),
        "author_early_late_shift": round(early_late_shift(author_traj), 4),
        "community_early_late_shift": round(early_late_shift(community_traj), 4),
        "trajectory_eligible": eligible,
        "trajectory_notes": "; ".join(notes) if notes else "ok",
        "used_body_as_first_author_point": analysis["used_body_as_first_author_point"],
        "title_fallback": analysis["title_fallback"],
        "community_mean_is_observed": analysis["community_mean_is_observed"],
        "author_early_late_shift_eligible": len(author_traj) >= 4,
        "community_early_late_shift_eligible": len(community_traj) >= 4,
    }
