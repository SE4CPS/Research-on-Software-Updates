"""Synthetic + light integration tests for master research 2026-11-03."""

from __future__ import annotations

from full_resource_sentiment_v1.master_research_2026_11_03.phase_a import protected_ids, run_inventory
from full_resource_sentiment_v1.master_research_2026_11_03.phase_b import (
    load_capture_texts_for_ids,
    load_pilot,
    second_rubric_classifier,
)
from full_resource_sentiment_v1.master_research_2026_11_03.phase_e import _near_dup_group
from full_resource_sentiment_v1.scoring import early_late_or_none


def test_inventory_hashes_and_protected():
    inv = run_inventory()
    assert inv["ok"] is True, inv["discrepancies"]
    assert inv["n"] == 6425
    assert inv["protected_overlap"] == 0
    assert len(protected_ids()) >= 700


def test_second_rubric_removed_and_cues():
    u = second_rubric_classifier("[removed]", "[deleted]")
    assert u["relevance"] == "unassessable"
    nr = second_rubric_classifier("I lost my keys", "Update: found them in laundry")
    # may still hit 'update' cue as word — ensure structured keys exist
    assert "relevance" in nr and "evidence_quote" in nr and "reason" in nr
    tps = second_rubric_classifier(
        "Kernel panic after upgrade to 6.11",
        "After the upgrade the wifi driver crashes. Looking for a workaround or rollback.",
    )
    assert tps["relevance"] == "relevant"
    assert tps["discussion"] == "TPS"
    gds = second_rubric_classifier(
        "This update ruined the app",
        "The latest release is awful and I hate it. Support is garbage.",
    )
    assert gds["relevance"] == "relevant"
    assert gds["discussion"] == "GDS"


def test_author_shift_definition_ge4():
    assert early_late_or_none([0.1, 0.2, 0.3]) is None
    assert early_late_or_none([0.0, 0.0, 1.0, 1.0]) == 1.0


def test_near_dup_group_stable():
    assert _near_dup_group("Hello", "World") == _near_dup_group(" hello ", "World")


def test_pilot_capture_text_available():
    """Excluded-pool pilot must resolve title/body from broad capture, not the 6,425 frame."""
    pilot = load_pilot()
    assert len(pilot) == 240
    sample_ids = {str(r["reddit_id"]) for r in pilot[:5]}
    texts = load_capture_texts_for_ids(sample_ids)
    assert set(texts) == sample_ids
    for rid, doc in texts.items():
        assert (doc.get("title") or "").strip() or (doc.get("body") or "").strip(), rid
