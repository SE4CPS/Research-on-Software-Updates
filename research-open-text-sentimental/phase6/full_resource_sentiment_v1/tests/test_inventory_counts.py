from __future__ import annotations

from full_resource_sentiment_v1.inventory import build_overlap_matrix
from full_resource_sentiment_v1.paths import EXPECTED_COUNTS, HISTORICAL_LABEL_COUNTS


def test_source_unique_id_and_overlap_counts(loaded_sources):
    loaded = loaded_sources
    overlap = build_overlap_matrix(loaded)
    assert overlap["matches_expected"], overlap["mismatches"]
    for key, expected in EXPECTED_COUNTS.items():
        assert overlap["actual"][key] == expected
    labels = loaded["historical_labels"]["corrected_label_counts"]
    assert int(labels["1"]) == HISTORICAL_LABEL_COUNTS["tps_corrected_1"]
    assert int(labels["0"]) == HISTORICAL_LABEL_COUNTS["gds_corrected_0"]
    assert loaded["master"]["n_uncertain_review"] == 5
    assert loaded["reserved"]["ids_only"] is True
    assert loaded["fresh"]["ids_only"] is True
    files = loaded["file_manifest"]
    assert files["n_missing"] == 0
    assert loaded["oct5"]["by_id"]["1wyg8ze"]["author_description"].strip()
