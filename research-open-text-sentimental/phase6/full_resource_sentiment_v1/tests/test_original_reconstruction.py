from __future__ import annotations

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from full_resource_sentiment_v1.compare_historical import compare_original_to_saved, parse_saved_row
from full_resource_sentiment_v1.loaders import load_c1_prime
from full_resource_sentiment_v1.original_semantics import (
    analyze_post_trajectories_original,
    compute_metrics_row_original,
)


def test_march_post_original_metrics_match_saved_c1_prime(march_post, march_label):
    saved = parse_saved_row(load_c1_prime()["by_id"]["1o0nzld"])
    analysis = analyze_post_trajectories_original(march_post, SentimentIntensityAnalyzer())
    reconstructed = compute_metrics_row_original(
        reddit_id="1o0nzld",
        subreddit=str(march_label.get("subreddit") or ""),
        corrected_label=int(march_label["corrected_label"]),
        url=str(march_label.get("url") or ""),
        analysis=analysis,
    )
    compared = compare_original_to_saved(reconstructed, saved)
    assert compared["within_tolerance"], compared
    assert reconstructed["n_author_replies"] == 2
    assert reconstructed["n_community_comments"] == 2
    assert analysis["used_body_as_first_author_point"] is True
    assert analysis["title_fallback"] is False
