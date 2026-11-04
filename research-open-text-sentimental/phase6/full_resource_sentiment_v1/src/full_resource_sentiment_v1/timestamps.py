"""Parse comment timestamps; never treat index order as elapsed time."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def parse_created_utc(value: Any) -> dict[str, Any]:
    """Return parse flags. Naive ISO strings are ordered as UTC for comparability only."""
    raw = "" if value is None else str(value).strip()
    if value is None or raw == "":
        return {
            "raw": raw,
            "valid": False,
            "parsed_utc": None,
            "epoch_seconds": None,
            "reason": "missing",
        }
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _from_unix(float(value), raw)
    if _looks_like_unix(raw):
        try:
            return _from_unix(float(raw), raw)
        except ValueError:
            pass
    iso = raw.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(iso)
    except ValueError:
        return {
            "raw": raw,
            "valid": False,
            "parsed_utc": None,
            "epoch_seconds": None,
            "reason": "unparseable",
        }
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return {
        "raw": raw,
        "valid": True,
        "parsed_utc": parsed.isoformat(),
        "epoch_seconds": parsed.timestamp(),
        "reason": "ok",
    }


def _looks_like_unix(raw: str) -> bool:
    if "T" in raw or "-" in raw or ":" in raw:
        return False
    try:
        float(raw)
    except ValueError:
        return False
    return True


def _from_unix(epoch: float, raw: str) -> dict[str, Any]:
    try:
        parsed = datetime.fromtimestamp(epoch, tz=timezone.utc)
    except (OSError, OverflowError, ValueError):
        return {
            "raw": raw,
            "valid": False,
            "parsed_utc": None,
            "epoch_seconds": None,
            "reason": "unparseable",
        }
    return {
        "raw": raw,
        "valid": True,
        "parsed_utc": parsed.isoformat(),
        "epoch_seconds": float(epoch),
        "reason": "ok",
    }
