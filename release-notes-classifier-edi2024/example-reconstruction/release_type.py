"""
Release Type classification by semantic-versioning position: MAJOR / MINOR /
PATCH, inferred directly from each note's own version number (no version
history needed. This is a heuristic based on trailing zeros, matching
common semver convention: a release ending in .0.0 is treated as a major
bump, ending in .0 (but not .0.0) as minor, anything else as patch).
"""

import re

VERSION_RE = re.compile(r"\b\d{1,4}(?:\.\d{1,4}){1,3}\b")


def extract_version(text):
    """Return the last version-like token in the text, or None."""
    matches = VERSION_RE.findall(text)
    return matches[-1] if matches else None


def release_type(text):
    version = extract_version(text)
    if not version:
        return "UNKNOWN"

    parts = [int(p) for p in version.split(".")]

    if len(parts) >= 3:
        # x.0.0(...) -> MAJOR ; x.y.0 -> MINOR ; x.y.z (z != 0) -> PATCH
        if all(p == 0 for p in parts[1:]):
            return "MAJOR"
        if parts[-1] == 0:
            return "MINOR"
        return "PATCH"
    else:
        # two-part version, e.g. "6.2"
        if parts[-1] == 0:
            return "MAJOR"
        return "MINOR"
