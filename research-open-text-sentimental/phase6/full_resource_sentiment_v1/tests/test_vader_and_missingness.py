from __future__ import annotations

import json
from pathlib import Path

from releasetrain_sentiment import score_post, trajectory_summary
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from full_resource_sentiment_v1.adapter import package_eligibility, to_package_post
from full_resource_sentiment_v1.harmonized import harmonized_measures
from full_resource_sentiment_v1.original_semantics import analyze_post_trajectories_original, mean_or_zero

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "synthetic_posts.json").read_text(encoding="utf-8"))
ANALYZER = SentimentIntensityAnalyzer()

# Independent VADER compounds for the hand-checked fixtures (vaderSentiment==3.3.2).
HANDCALC_POST_CONCAT = -0.4215
HANDCALC_AUTHOR_COMMENTS = [0.0, 0.4404]
HANDCALC_COMMUNITY_COMMENTS = [-0.4576, -0.4767]
GENUINE_ZERO_TEXT = "The table is made of wood."
GENUINE_ZERO = 0.0


def test_hand_calculated_vader_for_two_fixtures():
    thread = FIXTURES["handcalc_thread"]
    concat = f"{thread['title']}\n\n{thread['author_description']}".strip()
    assert ANALYZER.polarity_scores(concat)["compound"] == HANDCALC_POST_CONCAT
    author_means = [ANALYZER.polarity_scores(c["body"])["compound"] for c in thread["comments"] if c["is_submitter"]]
    community_means = [ANALYZER.polarity_scores(c["body"])["compound"] for c in thread["comments"] if not c["is_submitter"]]
    assert author_means == HANDCALC_AUTHOR_COMMENTS
    assert community_means == HANDCALC_COMMUNITY_COMMENTS
    expected_author_avg = sum(HANDCALC_AUTHOR_COMMENTS) / 2
    expected_community_avg = sum(HANDCALC_COMMUNITY_COMMENTS) / 2

    mapped = to_package_post(thread)
    scored = score_post(mapped)
    summary = trajectory_summary(scored)
    assert scored[0]["compound"] == HANDCALC_POST_CONCAT
    assert round(summary["author_avg"], 4) == round(expected_author_avg, 4)
    assert round(summary["community_avg"], 4) == round(expected_community_avg, 4)

    second = FIXTURES["genuine_neutral_zero"]
    assert ANALYZER.polarity_scores(GENUINE_ZERO_TEXT)["compound"] == GENUINE_ZERO
    mapped2 = to_package_post(second)
    scored2 = score_post(mapped2)
    assert scored2[0]["compound"] == GENUINE_ZERO
    assert scored2[1]["compound"] == GENUINE_ZERO
    summary2 = trajectory_summary(scored2)
    assert summary2["community_avg"] == GENUINE_ZERO
    elig2 = package_eligibility(mapped2)
    assert elig2["community_avg_is_observed"] is True


def test_genuine_zero_is_not_confused_with_no_observed_reply():
    zero_post = FIXTURES["genuine_neutral_zero"]
    none_post = FIXTURES["no_observed_reply"]
    zero_summary = trajectory_summary(score_post(to_package_post(zero_post)))
    none_mapped = to_package_post(none_post)
    none_summary = trajectory_summary(score_post(none_mapped))
    none_elig = package_eligibility(none_mapped)
    harm_none = harmonized_measures(ANALYZER, none_post)
    assert zero_summary["community_avg"] == 0.0
    assert none_summary["community_avg"] == 0.0
    assert none_elig["community_avg_is_observed"] is False
    assert harm_none["community_comment_mean"] is None
    assert harm_none["community_comment_mean_is_observed"] is False
    original = analyze_post_trajectories_original(none_post, ANALYZER)
    assert original["community_mean_is_observed"] is False
    assert mean_or_zero(original["community_traj"]) == 0.0


def test_shift_requires_four_role_specific_nonempty_comments():
    three = package_eligibility(to_package_post(FIXTURES["three_author_comments"]))
    four = package_eligibility(to_package_post(FIXTURES["four_author_comments"]))
    assert three["author_shift_eligible"] is False
    assert four["author_shift_eligible"] is True
    three_summary = trajectory_summary(score_post(to_package_post(FIXTURES["three_author_comments"])))
    four_native = package_eligibility(to_package_post(FIXTURES["four_author_comments"]))
    assert three_summary["author_shift"] == 0.0
    assert four["n_author_comments_nonempty"] == 4
    assert four_native["author_shift_eligible_package_native"] is True
    harm_three = harmonized_measures(ANALYZER, FIXTURES["three_author_comments"])
    harm_four = harmonized_measures(ANALYZER, FIXTURES["four_author_comments"])
    assert harm_three["author_shift_eligible"] is False
    assert harm_four["author_shift_eligible"] is True
