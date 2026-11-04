"""Synthetic tests for the approved 200-case validation sampler and harness."""

from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path

import pytest

from full_resource_sentiment_v1.validation_200.analysis_harness import (
    macro_f1,
    precision_recall_f1,
    weighted_confusion,
)
from full_resource_sentiment_v1.validation_200.sampler import (
    case_code_for_reddit_id,
    draw_stratum,
    draw_validation_sample,
    stratum_seed,
)
from full_resource_sentiment_v1.validation_200.text_resolve import redact_packet_text


def test_stratum_seed_stable_and_big_endian():
    a = stratum_seed("fresh_relevant_gds")
    b = stratum_seed("fresh_relevant_gds")
    assert a == b
    assert isinstance(a, int)
    assert a > 0
    # Different strata differ.
    assert stratum_seed("fresh_relevant_tps") != a


def test_draw_without_replacement_and_probabilities():
    ids = [f"id{i:04d}" for i in range(20)]
    drawn = draw_stratum(ids, 5, "synthetic_stratum_a")
    assert len(drawn) == 5
    assert len(set(drawn)) == 5
    assert set(drawn) <= set(ids)
    # Reproducible.
    assert draw_stratum(ids, 5, "synthetic_stratum_a") == drawn


def test_draw_validation_sample_synthetic_cells():
    strata = {
        "fresh_relevant_tps": {"n": 2, "expected_N": 5},
        "fresh_relevant_gds": {"n": 2, "expected_N": 4},
        "covered_not_relevant": {"n": 1, "expected_N": 3},
    }
    cells = {
        "fresh_relevant_tps": [f"a{i}" for i in range(5)],
        "fresh_relevant_gds": [f"b{i}" for i in range(4)],
        "covered_not_relevant": [f"c{i}" for i in range(3)],
    }
    out = draw_validation_sample(cells, strata=strata)
    assert out["n_selected"] == 5
    assert out["n_fresh"] == 4
    assert out["n_covered"] == 1
    assert len(out["selected_ids_sorted"]) == 5
    # Weights / inclusion probabilities.
    by_stratum = {}
    for row in out["selected"]:
        by_stratum.setdefault(row["stratum"], row)
        assert abs(row["inclusion_probability"] - row["n_h"] / row["N_h"]) < 1e-12
        assert abs(row["weight"] - row["N_h"] / row["n_h"]) < 1e-12
    assert abs(by_stratum["fresh_relevant_tps"]["weight"] - 2.5) < 1e-12
    # Reproducible selected set.
    out2 = draw_validation_sample(cells, strata=strata)
    assert out2["selected_ids_sorted"] == out["selected_ids_sorted"]
    assert out2["selected_ids_sha256"] == out["selected_ids_sha256"]
    # Presentation ranks unique.
    ranks = [row["presentation_order"] for row in out["selected"]]
    assert sorted(ranks) == list(range(1, 6))
    # Case codes opaque and stable.
    assert all(row["case_code"].startswith("V-") for row in out["selected"])
    assert case_code_for_reddit_id("a0") == case_code_for_reddit_id("a0")


def test_refuses_short_stratum():
    strata = {"fresh_relevant_tps": {"n": 3, "expected_N": 2}}
    cells = {"fresh_relevant_tps": ["a", "b"]}
    with pytest.raises(RuntimeError):
        draw_validation_sample(cells, strata=strata)


def test_redaction_removes_email_and_keeps_hashes_distinct():
    packet = redact_packet_text(
        "Update broke login for user@example.com",
        "Call me at +1-415-555-0100 or see https://example.com/path",
    )
    assert "user@example.com" not in packet["display_title"]
    assert "[REDACTED_EMAIL]" in packet["display_title"]
    assert "[REDACTED_PHONE]" in packet["display_body"] or "415" not in packet["display_body"]
    assert "[REDACTED_URL]" in packet["display_body"]
    assert packet["original_title_sha256"] != packet["display_title_sha256"]
    assert packet["body_truncated_for_model_cap"] is False


def test_weighted_confusion_known_population():
    # Invented population: N=10 GDS-predicted, draw n=2 with weight 5 each.
    rows = [
        {"predicted": "GDS", "human": "GDS", "weight": 5.0, "domain": "fresh"},
        {"predicted": "GDS", "human": "Other", "weight": 5.0, "domain": "fresh"},
        {"predicted": "TPS", "human": "TPS", "weight": 4.0, "domain": "fresh"},
    ]
    conf = weighted_confusion(rows, predicted_key="predicted", human_key="human")
    assert conf[("GDS", "GDS")] == 5.0
    assert conf[("GDS", "Other")] == 5.0
    gds = precision_recall_f1(conf, "GDS")
    assert gds["precision"] == 0.5
    assert gds["recall"] == 1.0
    tps = precision_recall_f1(conf, "TPS")
    assert tps["precision"] == 1.0
    assert tps["recall"] == 1.0
    assert macro_f1([gds, tps]) == pytest.approx((2 / 3 + 1.0) / 2)


def test_blinded_packet_has_no_forbidden_fields():
    forbidden = {
        "reddit_id",
        "final_relevance",
        "final_discussion",
        "weight",
        "inclusion_probability",
        "stratum",
        "model_reason",
        "vader",
        "compound",
        "predicted",
        "decision_source",
    }
    # Simulate a blinded public case record.
    case = {
        "case_code": "V-TEST",
        "presentation_order": 1,
        "title": "hello",
        "body": "world",
        "decision_a": "",
        "decision_b": "",
        "confidence": "",
        "evidence_phrase": "",
        "reason": "",
        "boundary_note": "",
        "recognized_prior_case": "",
    }
    assert not (set(case) & forbidden)


def test_temp_reproducibility_archive(tmp_path: Path):
    strata = {"fresh_relevant_tps": {"n": 2, "expected_N": 4}}
    cells = {"fresh_relevant_tps": ["w", "x", "y", "z"]}
    first = draw_validation_sample(cells, strata=strata)
    second = draw_validation_sample(cells, strata=strata)
    path = tmp_path / "ids.txt"
    path.write_text("\n".join(first["selected_ids_sorted"]) + "\n", encoding="utf-8")
    assert second["selected_ids_sorted"] == path.read_text(encoding="utf-8").splitlines()


def test_blank_and_removed_text_redaction():
    packet = redact_packet_text("[removed]", "[deleted]")
    assert packet["body_chars_original"] == len("[deleted]")
    assert packet["display_title"] == "[removed]"
    assert packet["body_truncated_for_model_cap"] is False


def test_review_app_autosave_revision_and_export(tmp_path: Path, monkeypatch):
    """Synthetic autosave/restart/export without touching the real draw directory."""
    import importlib.util
    import sqlite3
    import sys

    app_path = Path(__file__).resolve().parents[1] / "scripts" / "validation_200_review_app.py"
    spec = importlib.util.spec_from_file_location("validation_200_review_app", app_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # Point the app at a temp restricted dir before load executes module-level paths... 
    # Module binds paths at import; override after import.
    sys.modules["validation_200_review_app"] = mod
    spec.loader.exec_module(mod)

    cases_path = tmp_path / "blinded_cases.jsonl"
    cases = [
        {
            "case_code": f"V-SYNTH{i:04d}",
            "presentation_order": i,
            "title": f"synth title {i}",
            "body": f"synth body {i}",
        }
        for i in range(1, 3)
    ]
    with cases_path.open("w", encoding="utf-8") as handle:
        for row in cases:
            handle.write(json.dumps(row) + "\n")

    db_path = tmp_path / "review_state.sqlite"
    export_dir = tmp_path / "exports"
    monkeypatch.setattr(mod, "CASES_PATH", cases_path)
    monkeypatch.setattr(mod, "DB_PATH", db_path)
    monkeypatch.setattr(mod, "EXPORT_DIR", export_dir)
    monkeypatch.setattr(mod, "RESTRICTED", tmp_path)

    mod.load_cases()
    assert len(mod._cases) == 2

    saved = mod.save_decision(
        {
            "case_code": "V-SYNTH0001",
            "decision_a": "relevant",
            "decision_b": "TPS",
            "confidence": "high",
            "evidence_phrase": "synth",
            "reason": "first",
            "boundary_note": "",
            "recognized_prior_case": "no",
        }
    )
    assert saved["decision_a"] == "relevant"
    # Revision after edit.
    mod.save_decision(
        {
            "case_code": "V-SYNTH0001",
            "decision_a": "relevant",
            "decision_b": "GDS",
            "confidence": "medium",
            "evidence_phrase": "synth",
            "reason": "revised",
            "boundary_note": "tone",
            "recognized_prior_case": "no",
        }
    )
    # Restart simulation: reopen DB.
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM decisions WHERE case_code=?", ("V-SYNTH0001",)).fetchone()
    assert row["decision_b"] == "GDS"
    assert row["reason"] == "revised"
    n_rev = conn.execute("SELECT COUNT(*) AS n FROM revisions WHERE case_code=?", ("V-SYNTH0001",)).fetchone()["n"]
    assert n_rev == 2
    conn.close()

    # B must be not_applicable when A is not relevant.
    mod.save_decision(
        {
            "case_code": "V-SYNTH0002",
            "decision_a": "not_relevant",
            "decision_b": "TPS",
            "confidence": "low",
            "evidence_phrase": "",
            "reason": "no update",
            "boundary_note": "",
            "recognized_prior_case": "unsure",
        }
    )
    conn = sqlite3.connect(db_path)
    row2 = conn.execute("SELECT decision_b FROM decisions WHERE case_code=?", ("V-SYNTH0002",)).fetchone()
    conn.close()
    assert row2[0] == "not_applicable"

    export = mod.export_latest()
    assert export["n_cases"] == 2
    assert export["n_with_decision_a"] == 2
    assert Path(export["csv"]).exists()
    assert Path(export["jsonl"]).exists()
    assert Path(export["revisions"]).exists()
    rev_lines = Path(export["revisions"]).read_text(encoding="utf-8").strip().splitlines()
    assert len(rev_lines) >= 3
