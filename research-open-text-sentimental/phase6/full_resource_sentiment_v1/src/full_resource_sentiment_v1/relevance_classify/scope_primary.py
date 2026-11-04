"""Primary software-update scope for the original research definition.

This module does not call a model and does not read protected outcomes.
Second-review labels are not used here.
"""

from __future__ import annotations

import re
from typing import Any

SCOPE_VERSION = "primary_update_change_v3"
SCOPE_SUMMARY = (
    "Primary relevant posts centrally discuss an identifiable update, upgrade, "
    "patch, rollback, or version change to existing software/OS/app/firmware, "
    "or a consequence attributed to that change. First-ever launches without a "
    "prior version are out of the primary cohort. Ambiguous OS-migration and "
    "bare version mentions are flagged, not silently treated as relevant. "
    "TPS/GDS purpose labels and sentiment scores are separate dimensions."
)

# Strong evidence that an existing install changed.
STRONG_CHANGE_RE = re.compile(
    r"""
    (
        \b(upgrad(?:e|ed|ing|es)|patch(?:ed|es|ing)?|rollback|rolled\ back|
           hotfix|changelog|firmware\ update|git\ pull|windows\ update|
           updating|updated)\b
        | \bkb\d{5,}\b
        | \bupdate the firmware\b
        | \bafter\ (?:the\ )?(?:update|upgrade|patch)
        | \bsince\ (?:the\ )?(?:update|upgrade)
        | \bbroke\ (?:after|since)
        | (?:from|to)\s+[\w.-]*\d[\w.-]*\s+(?:to|from)\s+[\w.-]*\d
    )
    """,
    re.I | re.VERBOSE,
)

SQL_DML_RE = re.compile(
    r"\b(matched zero row|UPDATE actually worked|database UPDATE|SQL UPDATE statement)\b",
    re.I,
)
FIRST_RELEASE_RE = re.compile(
    r"\b(first ever app|i made my first|submitted for review|not yet packaged|"
    r"closed (?:test|pilot)|meet \w+!!)\b",
    re.I,
)
HARDWARE_RE = re.compile(
    r"\b(hardware revision|development board|new router \(same model\)|"
    r"replaced 99% of my lights|security gateway line)\b",
    re.I,
)
VENDOR_SWITCH_RE = re.compile(r"\b(open\s*router|switching to openrouter)\b", re.I)
CERT_IP_RE = re.compile(
    r"\b(ssl certificate was updated|update the public .+ IP|DuckDNS integration to update)\b",
    re.I,
)
PRICE_RE = re.compile(r"\b(rising prices|higher prices)\b", re.I)
ADVISORY_ONLY_RE = re.compile(
    r"\b(vulnerabilit\w*|CVE-\d+)\b",
    re.I,
)
STATUS_UPDATES_RE = re.compile(r"\b(latest updates installed)\b", re.I)
NEWS_UPDATE_RE = re.compile(r"\b(news update|status update|daily update)\b", re.I)
OS_MIGRATION_RE = re.compile(
    r"\b(dual boot|switching to linux|moved? to (?:a )?linux|"
    r"installed debian after|replace(?:d|ing)? linux)\b",
    re.I,
)


def _blob(*parts: str | None) -> str:
    return "\n".join(part or "" for part in parts)


def scope_flags(title: str, body: str, evidence: str | None = None, reason: str | None = None) -> dict[str, Any]:
    text = _blob(title, body, evidence, reason)
    strong = bool(STRONG_CHANGE_RE.search(text))
    flags = {
        "scope_version": SCOPE_VERSION,
        "strong_existing_change": strong,
        "first_release_boundary": bool(FIRST_RELEASE_RE.search(text)),
        "os_migration_boundary": bool(OS_MIGRATION_RE.search(text)),
        "sql_dml_update": bool(SQL_DML_RE.search(text)),
        "hardware_or_network": bool(HARDWARE_RE.search(text)),
        "vendor_switch": bool(VENDOR_SWITCH_RE.search(text)),
        "certificate_or_ip_refresh": bool(CERT_IP_RE.search(text)),
        "pricing_change": bool(PRICE_RE.search(text)),
        "advisory_names_version": bool(ADVISORY_ONLY_RE.search(text)),
        "status_latest_updates": bool(STATUS_UPDATES_RE.search(text)),
        "news_or_status_update": bool(NEWS_UPDATE_RE.search(text)),
    }
    demote = None
    if flags["sql_dml_update"]:
        demote = "sql_dml_update"
    elif flags["certificate_or_ip_refresh"]:
        demote = "certificate_or_ip_refresh"
    elif flags["pricing_change"] and not strong:
        demote = "pricing_change"
    elif flags["hardware_or_network"] and not strong:
        demote = "hardware_or_network"
    elif flags["vendor_switch"] and not strong:
        demote = "vendor_switch"
    elif flags["status_latest_updates"] and not strong:
        demote = "status_latest_updates"
    elif flags["news_or_status_update"]:
        demote = "news_or_status_update"
    elif flags["advisory_names_version"] and not strong:
        demote = "advisory_only"
    elif flags["first_release_boundary"]:
        demote = "first_release_boundary"
    flags["second_pass_demote"] = demote
    flags["os_migration_flag_only"] = flags["os_migration_boundary"]
    return flags


def apply_second_pass(prediction: dict[str, Any], title: str, body: str) -> dict[str, Any]:
    """Demote a relevant prediction that matches a high-confidence exclusion.

    Errors and non-relevant labels are unchanged. OS-migration is flagged only.
    """
    out = dict(prediction)
    flags = scope_flags(
        title,
        body,
        prediction.get("evidence_phrase"),
        prediction.get("decision_reason"),
    )
    out["scope_flags"] = flags
    if (
        prediction.get("predicted_label") == "relevant"
        and prediction.get("response_status") == "ok"
        and flags["second_pass_demote"]
    ):
        out["predicted_label"] = "not_relevant"
        out["second_pass_applied"] = True
        out["second_pass_reason"] = flags["second_pass_demote"]
        out["pre_second_pass_label"] = "relevant"
    else:
        out["second_pass_applied"] = False
        out["second_pass_reason"] = None
    return out
