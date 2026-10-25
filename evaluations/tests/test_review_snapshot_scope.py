"""The label audit must protect evidence while exposing runtime source drift."""

import copy
import hashlib
import json

import pytest

from evaluations import review_suggestions_check as checks


@pytest.fixture
def tiny_snapshot(tmp_path, monkeypatch):
    paths = checks.REQUIRED_INPUTS | checks.RUNTIME_SNAPSHOT_FILES | {checks.OUTPUT_RELATIVE}
    for relative in paths:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original", encoding="utf-8")
    sha = hashlib.sha256(b"original").hexdigest()
    manifest = {"schema_version": "1.0", "generated_on": checks.DATE,
                "time_zone": "Asia/Shanghai", "ai_suggestions_only": True,
                "human_review_completed": False,
                "inputs": [{"path": path, "sha256": sha} for path in sorted(paths - {checks.OUTPUT_RELATIVE})],
                "output": {"path": checks.OUTPUT_RELATIVE, "sha256": sha}}
    source = tmp_path / "manifest.json"
    source.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(checks, "REPO", tmp_path)
    monkeypatch.setattr(checks, "MANIFEST", source)
    return tmp_path, copy.deepcopy(manifest)


def test_runtime_drift_is_reported_and_strict_snapshot_still_rejects(tiny_snapshot):
    root, original = tiny_snapshot
    (root / "app/agent.py").write_text("authorised runtime edit", encoding="utf-8")
    result = checks.verify_manifest(strict_snapshot=False)
    assert result["snapshot_validation"]["label_and_evidence_inputs_match"] is True
    assert result["snapshot_validation"]["all_historical_snapshot_files_match"] is False
    assert result["snapshot_validation"]["runtime_source_changes"][0]["path"] == "app/agent.py"
    assert checks.read_json(checks.MANIFEST) == original
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        checks.verify_manifest()


@pytest.mark.parametrize("relative", sorted(checks.REQUIRED_INPUTS | {checks.OUTPUT_RELATIVE}))
def test_runtime_mode_never_allows_evidence_or_label_drift(tiny_snapshot, relative):
    root, _ = tiny_snapshot
    (root / relative).write_text("tampered evidence or labels", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        checks.verify_manifest(strict_snapshot=False)


def test_public_scope_reports_a_missing_private_archive_without_claiming_full_snapshot(tiny_snapshot):
    _, original = tiny_snapshot
    original["inputs"].append({"path": "evaluations/results/private-run/report.json", "sha256": "0" * 64})
    checks.MANIFEST.write_text(json.dumps(original), encoding="utf-8")
    result = checks.verify_manifest(strict_snapshot=False)["snapshot_validation"]
    assert result["validation_scope"] == "public_evidence_and_label_inputs"
    assert result["label_and_evidence_inputs_match"] is True
    assert result["all_historical_snapshot_files_match"] is False
    assert result["unavailable_historical_files"][0]["path"].endswith("private-run/report.json")
    with pytest.raises(ValueError, match="Missing file"):
        checks.verify_manifest(strict_snapshot=True)


def test_public_scope_records_readme_drift_but_strict_mode_keeps_original_hash(tiny_snapshot):
    root, original = tiny_snapshot
    path = root / "README.md"
    path.write_text("historical instructions")
    original["inputs"].append({"path": "README.md", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    checks.MANIFEST.write_text(json.dumps(original), encoding="utf-8")
    path.write_text("portable installation instructions")
    result = checks.verify_manifest(strict_snapshot=False)["snapshot_validation"]
    assert any(item["path"] == "README.md" for item in result["runtime_source_changes"])
    assert result["all_historical_snapshot_files_match"] is False
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        checks.verify_manifest(strict_snapshot=True)
