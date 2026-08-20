"""
classify_breaking_type.py

Keyword-based classifier for update impact / breaking-change severity.

NOTE ON NAMING: the paper (Berhe et al., "Triage Software Update Impact via
Release Notes Classification," Procedia Computer Science, 2024) reports a
"Release Type" category. This script's current name in production is
"Breaking Type" — it is the closest live equivalent, but has not been
confirmed to use the exact same label taxonomy as the paper's original
"Release Type". Treat the mapping as provisional; see README.

Pure classification logic, no external dependencies — pass any release-note
text to classify_text() for the keyword-category labels, or
detect_breaking_version() to flag a major version bump (x.0 / x.0.0) as a
"Breaking Update" independent of keyword matching.
"""

import re

# Define `breakingType` categories (user-friendly labels)
categories_breaking = {
    "Critical Failure": [  # Complete failures, critical breakdowns, and fatal errors
        "crash", "failure", "downtime", "outage", "broken",
        "error", "unavailable", "access denied", "refused",
        "timeout", "halted", "stuck", "frozen", "terminated",
        "shutdown", "rollback", "corrupted", "deadlock", "aborted",
        "disconnected", "expired", "reboot", "unauthorized",
        "forbidden", "unreachable", "exception", "denial",

        "denial of service", "critical failure", "fatal error",
        "severe disruption", "total outage", "system crash",
        "server down", "permanent failure", "critical bug",

        "400 bad request", "401 unauthorized", "403 forbidden",
        "404 not found", "405 method not allowed", "408 request timeout",
        "429 too many requests", "500 internal server error",
        "501 not implemented", "502 bad gateway", "503 service unavailable",
        "504 gateway timeout", "505 http version not supported",
        "506 variant also negotiates", "507 insufficient storage",
        "508 loop detected", "510 not extended", "511 network authentication required",

        "DoS", "DDoS", "RCE", "SQLi", "CSRF", "XSS", "TLS",
        "RBAC", "SSO", "LDAP", "MITM", "zero-day", "exploit"
    ],

    "Limited Functionality": [  # Partial failures, degraded performance, non-fatal issues
        "unstable", "deprecated", "removed feature", "overload",
        "disrupted", "unresponsive", "buggy", "instability",
        "overrun", "hanging", "lagging", "restart", "glitch",
        "delayed", "reset", "locked", "exhausted", "truncated",
        "retrying", "patchy", "recovered", "degraded", "drained",

        "service degradation", "temporary outage", "unstable connection",
        "intermittent failure", "partial downtime", "delayed response",
        "feature restriction", "partial loss of service",

        "307 temporary redirect", "308 permanent redirect",
        "422 unprocessable entity", "423 locked", "424 failed dependency",
        "425 too early", "426 upgrade required"
    ],

    "Performance Issues": [  # Issues impacting speed, responsiveness, and efficiency
        "slow", "lag", "delayed", "congested", "spiking",
        "bottlenecked", "blocked", "saturated", "throttled",
        "stalled", "backlogged", "starved", "leaking",
        "underperforming", "exhausted", "fragmented", "running out",

        "slow response", "increased latency", "throughput reduction",
        "memory bloat", "cpu spike", "disk I/O bottleneck",
        "excessive garbage collection", "network congestion", "packet loss",
        "cache miss", "high CPU usage", "memory fragmentation",
        "database lock", "queue overflow", "inefficient queries",
        "thread contention", "process hang", "concurrent bottleneck",

        "query performance degradation", "server resource exhaustion",
        "excessive load time", "delayed execution", "poor response time",
        "load balancing issue", "under-provisioned resources",
        "rate limiting hit", "API throttling", "bandwidth congestion"
    ],

    "Compatibility Issues": [
        "incompatible", "unsupported", "mismatch", "deprecated",
        "legacy", "obsoleted", "breaking", "versioning", "portability",
        "formatting"
    ],

    "Configuration Errors": [
        "misconfigured", "invalid", "incorrect", "unset",
        "missing", "malformed", "defaulted", "uninitialized",
        "overridden", "conflicting"
    ],

    "Resource Exhaustion": [
        "starved", "exhausted", "overloaded", "depleted",
        "drained", "consumed", "saturated", "throttled",
        "fragmented", "strained"
    ],

    "Network Issues": [
        "disconnected", "unreachable", "timeout", "dropped",
        "congested", "reset", "refused", "latency",
        "interrupted", "denied"
    ],

    "Hardware Failures": [
        "overheating", "corrupted", "damaged", "faulty",
        "unstable", "crashed", "failed", "glitching",
        "degraded", "unresponsive"
    ],

    "Dependency Failures": [
        "unavailable", "unresolved", "missing", "mismatched",
        "broken", "outdated", "conflicting", "deprecated",
        "failing", "malfunctioning"
    ],

    "Concurrency Issues": [
        "deadlock", "race", "stalled", "blocked",
        "overlapping", "starved", "timeout", "desynchronized",
        "contention", "interleaved"
    ],

    "Data Integrity Issues": [
        "corrupt", "duplicate", "inconsistent", "stale",
        "missing", "truncated", "mismatched", "tampered",
        "drifted", "invalid"
    ],

    "Logging & Monitoring Failures": [
        "silent", "unlogged", "omitted", "missing",
        "suppressed", "lagging", "overloaded", "inaccurate",
        "delayed", "disconnected"
    ]
}


def classify_text(text):
    """Return (labels, highlighted_text) for the Breaking Type keyword categories."""
    if not text:
        return [], text

    words = re.findall(r'\b\w+\b', text.lower())
    matches = set()
    highlighted_text = text

    for word in words:
        for category, keywords in categories_breaking.items():
            if word.lower() in (kw.lower() for kw in keywords):
                matches.add(category)
                highlighted_text = re.sub(r'\b' + re.escape(word) + r'\b', f"**{word}**", highlighted_text, flags=re.IGNORECASE)

    return list(matches) if matches else [], highlighted_text


def detect_breaking_version(text):
    """Detects major breaking versions (x.0 or x.0.0) while avoiding x.y.0, x.0.y, and false positives."""
    match_x00 = re.search(r'\b\d{1,2}\.0\.0\b', text)
    match_x0 = re.search(r'\b\d{1,2}\.0\b', text)
    reject_x0y = re.search(r'\b\d{1,2}\.0\.\d+\b', text)  # Reject x.0.y

    if match_x00:
        return True
    if match_x0 and not reject_x0y:
        return True
    return False


if __name__ == "__main__":
    sample = "Upgrading to version 3.0 causes a full crash and total outage on startup."
    labels, highlighted = classify_text(sample)
    print(labels)
    print(highlighted)
    print("Breaking version detected:", detect_breaking_version(sample))
