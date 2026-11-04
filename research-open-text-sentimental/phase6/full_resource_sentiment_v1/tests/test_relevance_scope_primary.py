"""Synthetic tests for the original-scope second pass. No live API, no second-review labels."""

from __future__ import annotations

from full_resource_sentiment_v1.relevance_classify.scope_primary import apply_second_pass, scope_flags


def _pred(label="relevant", evidence="after the update Safari crashed", reason="Named version change."):
    return {
        "predicted_label": label,
        "response_status": "ok",
        "evidence_phrase": evidence,
        "decision_reason": reason,
    }


def test_keeps_identifiable_existing_update():
    out = apply_second_pass(
        _pred(),
        "iOS 18.2 made Safari crash",
        "After the update Safari crashes on launch.",
    )
    assert out["predicted_label"] == "relevant"
    assert out["second_pass_applied"] is False
    assert out["scope_flags"]["strong_existing_change"] is True


def test_excludes_news_update():
    out = apply_second_pass(
        _pred(evidence="daily news update", reason="Mentions update."),
        "Daily news update: local election results",
        "Here is this morning's news update. No software is discussed.",
    )
    assert out["predicted_label"] == "not_relevant"
    assert out["second_pass_reason"] == "news_or_status_update"


def test_excludes_sql_dml_not_cyberpanel_upgrade():
    sql = apply_second_pass(
        _pred(evidence="whether the UPDATE actually worked", reason="Database UPDATE."),
        "Agent tools often can't tell you if they actually succeeded",
        "I was testing whether the UPDATE actually worked or matched zero row.",
    )
    assert sql["predicted_label"] == "not_relevant"
    keep = apply_second_pass(
        _pred(evidence="upgrade to the latest version", reason="Upgrade broke phpMyAdmin."),
        "Cyberpanel error after upgrade",
        "After I upgrade to the latest version phpMyAdmin is a blank page.",
    )
    assert keep["predicted_label"] == "relevant"


def test_excludes_first_ever_app_without_prior_version():
    out = apply_second_pass(
        _pred(evidence="Its in beta v1.5", reason="Beta and weekly updates."),
        "Im 13 and I made my first ever app :D",
        "Its in beta v1.5. I will add weekly content updates.",
    )
    assert out["predicted_label"] == "not_relevant"
    assert out["scope_flags"]["first_release_boundary"] is True


def test_flags_os_migration_without_demoting():
    out = apply_second_pass(
        _pred(evidence="dual boot", reason="Linux dual boot."),
        "Cons of Linux Dual boot with windows laptop.",
        "I am thinking about dual boot. No version change named.",
    )
    assert out["scope_flags"]["os_migration_boundary"] is True
    assert out["predicted_label"] == "relevant"
    assert out["second_pass_applied"] is False


def test_excludes_hardware_revision_and_prices_and_advisory_only():
    board = apply_second_pass(
        _pred(evidence="released the latest revision", reason="Hardware revision."),
        "ArduLora board",
        "I’ve just released the latest revision of this development board.",
    )
    assert board["second_pass_reason"] == "hardware_or_network"
    price = apply_second_pass(
        _pred(evidence="rising prices", reason="Prices changed."),
        "New to ollama cloud and found I got higher prices",
        "Prices are in the middle of rising prices.",
    )
    assert price["second_pass_reason"] == "pricing_change"
    cve = apply_second_pass(
        _pred(evidence="version 1.26.0", reason="Vulnerability in a version."),
        "Critical 9.8 vulnerability in Unbound up to and including version 1.26.0",
        "Advisory text names an affected version. Nobody discusses installing a fix.",
    )
    assert cve["second_pass_reason"] == "advisory_only"


def test_does_not_convert_errors():
    row = {
        "predicted_label": None,
        "response_status": "evidence_fail",
        "evidence_phrase": None,
        "decision_reason": None,
    }
    out = apply_second_pass(row, "title", "body")
    assert out["response_status"] == "evidence_fail"
    assert out["predicted_label"] is None


def test_keeps_firmware_and_version_to_version_upgrade():
    fw = apply_second_pass(
        _pred(evidence="update the firmware via iCUE", reason="Firmware update."),
        "Corsair hub on Linux",
        "I had to update the firmware via iCUE.",
    )
    assert fw["predicted_label"] == "relevant"
    ver = apply_second_pass(
        _pred(evidence="upgraded from 2026.7.4 to 2026.8.2", reason="HA upgrade."),
        "Update to 2026.8.2 broke remote access",
        "I've recently upgraded from 2026.7.4 to 2026.8.2.",
    )
    assert ver["predicted_label"] == "relevant"
    assert ver["scope_flags"]["strong_existing_change"] is True
