"""Offline archive-label version tests; never call a model."""

from copy import deepcopy
import json

import pytest

from evaluations import rescore_ai_labels as rescore


@pytest.fixture
def label_data():
    labels, sources, _, _ = rescore.load_labels()
    return labels, sources


def u006_row():
    return {"id": "U006", "expected_label": "uncertain", "predicted_label": "not_supported",
            "elapsed_seconds": 1.0, "error": None}


def test_u006_scores_against_two_versions_without_mutating_archive(label_data):
    labels, sources = label_data
    rows = [u006_row()]
    before = deepcopy(rows)
    result = rescore.score_rows(rows, labels, sources, {"case_count": 1})
    assert rows == before
    assert result["original_labels"]["accuracy"] == 0
    assert result["ai_suggested_labels"]["accuracy"] == 1
    assert result["changed_agreement_rows"][0]["id"] == "U006"
    assert "latency_seconds" not in result["original_labels"]
    assert result["historical_latency_seconds"]["total"] == 1


def test_unknown_or_omitted_case_rejected(label_data):
    labels, sources = label_data
    with pytest.raises(ValueError, match="missing"):
        rescore.score_rows([dict(u006_row(), id="UNKNOWN")], labels, sources, {})
    with pytest.raises(ValueError, match="omit"):
        rescore.score_rows([u006_row()], labels, sources, {"selected_ids": ["U006", "S001"]})
    with pytest.raises(ValueError, match="case_count"):
        rescore.score_rows([u006_row()], labels, sources, {"case_count": 2})


def test_repeated_archives_are_kept_separate_not_pooled(tmp_path, label_data):
    labels, sources = label_data
    paths = []
    for index in range(2):
        path = tmp_path / f"attempt-{index}.json"
        path.write_text(json.dumps({"selected_ids": ["U006"],
                                  "v1": {"rows": [u006_row()]},
                                  "v2": {"rows": [u006_row()]}}))
        paths.append(path)
    result = rescore.rescore_archives(paths, labels, sources)
    assert len(result) == 2
    assert result[0]["strategy_scores"]["v1"]["case_count"] == 1
    assert result[1]["strategy_scores"]["v1"]["case_count"] == 1
    assert all("reuse_note" in archive for archive in result)


def test_failures_still_count_and_claim_or_evidence_changes_rejected(label_data):
    labels, sources = label_data
    failed = dict(u006_row(), error={"type": "TimeoutExpired"})
    result = rescore.score_rows([failed], labels, sources, {})
    assert result["ai_suggested_labels"]["error_count"] == 1
    assert result["ai_suggested_labels"]["accuracy"] == 0
    with pytest.raises(ValueError, match="claim differs"):
        rescore.score_rows([dict(u006_row(), claim="changed")], labels, sources, {})
    with pytest.raises(ValueError, match="evidence differs"):
        rescore.score_rows([dict(u006_row(), documents=[])], labels, sources, {})
