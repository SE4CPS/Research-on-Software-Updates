"""Focused tests for the bounded broad-capture page contract."""

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "collect_releasetrain_broad.py"
spec = importlib.util.spec_from_file_location("collect_releasetrain_broad", SCRIPT)
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


def test_requires_exact_ids_and_array():
    with pytest.raises(ValueError, match="data array"):
        collector.safe_rows({"rows": []})
    with pytest.raises(ValueError, match="Reddit ID"):
        collector.safe_rows({"data": [{"title": "x"}]})
    assert collector.safe_rows({"data": [{"redditId": "abc"}]})[0]["redditId"] == "abc"


def test_page_audit_counts_presence_without_exporting_text():
    rows = [{"redditId": "abc", "title": "t", "author_description": "b", "created_utc": "2026-10-01",
             "author": "op", "source": "reddit", "comments": [
                 {"author": "op", "body": "reply", "created_utc": "2026-10-02", "is_submitter": True},
                 {"author": "other", "body": "", "created_utc": None, "is_submitter": False},
             ]}]
    ids, counts = collector.audit_page(rows)
    assert ids == ["abc"]
    assert counts["stored_comment_turns"] == 2
    assert counts["nonempty_comment_turns"] == 1
    assert counts["is_submitter_true_turns"] == 1
    assert "reply" not in str(counts)
