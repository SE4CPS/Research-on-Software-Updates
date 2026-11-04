"""Tests for corrections package — synthetic fixtures only (do not overwrite A–F outputs)."""

from __future__ import annotations

from full_resource_sentiment_v1.master_research_corrections_2026_11_03.near_dup import (
    NearDupPolicy,
    assert_no_cross_split_overlap,
    build_components,
    jaccard,
    normalize_ws,
    word_set,
)
from full_resource_sentiment_v1.scoring import early_late_or_none


def test_normalize_and_jaccard():
    assert normalize_ws("  Hello   WORLD ") == "hello world"
    assert jaccard(word_set("a b c"), word_set("b c d")) == 0.5


def test_exact_full_text_component():
    ids = ["1", "2", "3"]
    titles = ["Same title here long enough", "Same title here long enough", "Other"]
    bodies = ["body alpha beta", "body alpha beta", "different"]
    out = build_components(ids, titles, bodies)
    # 1 and 2 exact full text → same component
    assert out["component_ids"][0] == out["component_ids"][1]
    assert out["component_ids"][0] != out["component_ids"][2]


def test_title_body_near_dup_paraphrase_and_repost():
    """Shared nontrivial title + high body Jaccard merges; paraphrased different title does not."""
    ids = ["a", "b", "c", "d"]
    titles = [
        "App crashed after the latest update to 2.0",
        "App crashed after the latest update to 2.0",  # repost same title
        "Totally different subject line about cooking",
        "App crashed after the latest update to 2.0",
    ]
    bodies = [
        "After updating my phone the wifi driver fails and I need a workaround",
        "After updating my phone the wifi driver fails and I need a fix workaround",  # near-dup body
        "I baked bread today and it was delicious with butter",
        "Completely unrelated body text about gardening tomatoes and soil pH",  # same title, low jaccard
    ]
    policy = NearDupPolicy(title_min_chars=20, title_body_jaccard=0.55)
    out = build_components(ids, titles, bodies, policy=policy)
    # a and b should merge
    assert out["component_ids"][0] == out["component_ids"][1]
    # c separate
    assert out["component_ids"][2] != out["component_ids"][0]
    # d same title but low body overlap → separate under 0.55
    assert jaccard(word_set(bodies[0]), word_set(bodies[3])) < 0.55
    assert out["component_ids"][3] != out["component_ids"][0]


def test_cross_split_overlap_detector():
    comps = [0, 0, 1, 2]
    assert assert_no_cross_split_overlap(comps, [0], [1]) == 1  # same component
    assert assert_no_cross_split_overlap(comps, [0], [2]) == 0


def test_author_shift_definition_preserved():
    assert early_late_or_none([0.1, 0.2, 0.3]) is None
    assert early_late_or_none([0.0, 0.0, 1.0, 1.0]) == 1.0
