import json
import os
from pathlib import Path
from typing import Optional

CONFIG_PATH = Path.home() / ".codesnip" / "config.json"


def _load() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {}


def _save(data: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(data, f, indent=2)


def set_key(name: str, value: str) -> None:
    data = _load()
    data[name] = value
    _save(data)


def get_key(name: str) -> Optional[str]:
    return _load().get(name)


def get_all() -> dict:
    return _load()


def get_ollama_url() -> str:
    """Returns saved config → env var → hardcoded default. Never requires user to set."""
    return get_key("ollama_url") or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def get_ollama_model() -> str:
    """Returns saved config → env var → hardcoded default. Never requires user to set."""
    return get_key("ollama_model") or os.getenv("OLLAMA_MODEL", "llama3")


def get_github_token() -> Optional[str]:
    return get_key("github_token") or os.getenv("GITHUB_TOKEN")
