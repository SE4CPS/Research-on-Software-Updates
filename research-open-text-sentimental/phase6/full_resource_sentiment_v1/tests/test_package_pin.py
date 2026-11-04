from __future__ import annotations

from importlib.metadata import version

import releasetrain_sentiment

from full_resource_sentiment_v1.hashing import sha256_file
from full_resource_sentiment_v1.paths import PACKAGE_TARBALL

EXPECTED_TARBALL_SHA256 = "251b36225f53772bb5cdd3bd126e9142b00bd89f228f175be89c53e82ed94b77"


def test_package_version_and_saved_tarball_hash():
    assert version("releasetrain-sentiment") == "0.2.1"
    assert releasetrain_sentiment.__version__ == "0.2.1"
    assert version("vaderSentiment") == "3.3.2"
    assert sha256_file(PACKAGE_TARBALL) == EXPECTED_TARBALL_SHA256
    assert "score_post" in dir(releasetrain_sentiment)
    assert "trajectory_summary" in dir(releasetrain_sentiment)
