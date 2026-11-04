from __future__ import annotations

import sys
from pathlib import Path

import pytest

WORKSPACE = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKSPACE.parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))
sys.path.insert(0, str(REPO_ROOT))

from full_resource_sentiment_v1.inventory import load_all_sources  # noqa: E402
from full_resource_sentiment_v1.loaders import (  # noqa: E402
    load_historical_labels,
    load_historical_raw,
    load_oct5_pages,
)

MARCH_ID = "1o0nzld"
OCT5_ID = "1wyg8ze"


@pytest.fixture(scope="session")
def march_post() -> dict:
    raw = load_historical_raw()
    return raw["merged_by_id_original_rule"][MARCH_ID]


@pytest.fixture(scope="session")
def oct5_post() -> dict:
    pages = load_oct5_pages()
    return pages["by_id"][OCT5_ID]


@pytest.fixture(scope="session")
def march_label() -> dict:
    return load_historical_labels()["by_id"][MARCH_ID]


@pytest.fixture(scope="session")
def loaded_sources() -> dict:
    return load_all_sources()
