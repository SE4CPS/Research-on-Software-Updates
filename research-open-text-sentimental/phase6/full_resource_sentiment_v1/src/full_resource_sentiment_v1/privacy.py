"""Reject public exports that contain raw text, usernames, or URLs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

FORBIDDEN_KEY_EXACT = {
    "title",
    "body",
    "text",
    "author",
    "username",
    "url",
    "author_description",
    "selftext",
    "description",
    "snippet",
    "post_body",
    "comment_body",
    "permalink",
}

FORBIDDEN_KEY_SUBSTRINGS = (
    "username",
    "author_name",
    "raw_text",
    "snippet",
    "permalink",
)

FORBIDDEN_VALUE_MARKERS = (
    "reddit.com/",
    "http://",
    "https://",
)


def forbidden_key(key: str, value: Any) -> bool:
    lowered = key.lower()
    if any(part in lowered for part in FORBIDDEN_KEY_SUBSTRINGS):
        return True
    if lowered not in FORBIDDEN_KEY_EXACT:
        return False
    if isinstance(value, str) and value.strip():
        return True
    if isinstance(value, list) and any(isinstance(item, str) and item.strip() for item in value):
        return True
    return False


def _scan(obj: Any, *, path: str, hits: list[str]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            here = f"{path}.{key}" if path else str(key)
            if forbidden_key(str(key), value):
                hits.append(here)
            _scan(value, path=here, hits=hits)
        return
    if isinstance(obj, list):
        for index, value in enumerate(obj[:50]):
            _scan(value, path=f"{path}[{index}]", hits=hits)
        if len(obj) > 50:
            _scan(obj[-1], path=f"{path}[-1]", hits=hits)
        return
    if isinstance(obj, str):
        lowered = obj.lower()
        if any(marker in lowered for marker in FORBIDDEN_VALUE_MARKERS):
            hits.append(path or "<string>")


def scan_obj(obj: Any) -> list[str]:
    hits: list[str] = []
    _scan(obj, path="", hits=hits)
    return hits


def scan_json_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        hits: list[str] = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            hits.extend(f"{path.name}:{line_no}:{hit}" for hit in scan_obj(json.loads(line)))
        return hits
    return [f"{path.name}:{hit}" for hit in scan_obj(json.loads(text))]


def scan_paths(paths: list[Path]) -> list[str]:
    hits: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        hits.extend(scan_json_file(path))
    return hits
