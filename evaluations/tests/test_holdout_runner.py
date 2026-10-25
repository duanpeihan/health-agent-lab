"""The paired holdout harness is tested without network or model calls."""

import copy
import json

import pytest

from evaluations import holdout_runner
from evaluations.dataset import load_dataset
from evaluations.public_fixtures import materialize_historical_source_root, public_fixture_path


@pytest.fixture
def historical_source_root(tmp_path, monkeypatch):
    """Replay the old source guard using the original run's saved source bytes."""
    # Explicit metadata fixture: no Ollama service or local model is needed.
    monkeypatch.setattr(holdout_runner, "MODEL_MANIFEST", public_fixture_path("model_manifest"))
    return materialize_historical_source_root(tmp_path)


def test_frozen_holdout_and_model_configuration_verify_without_model_calls(monkeypatch, historical_source_root):
    freeze = json.loads(holdout_runner.DEFAULT_FREEZE.read_text())
    # Runtime reliability changes must not silently reuse the historical freeze.
    with pytest.raises(ValueError, match="frozen source changed: app/claim_verifier.py"):
        holdout_runner.verify_freeze(freeze)
    monkeypatch.setattr(holdout_runner, "ROOT", historical_source_root)
    dataset = holdout_runner.verify_freeze(freeze)
    assert len(dataset["cases"]) == 8
    assert freeze["model"]["name"] == "qwen3:4b"
    assert freeze["model"]["options"] == {"temperature": 0}
    assert freeze["model"]["think"] is False
    altered = copy.deepcopy(freeze)
    altered["source_sha256"]["evaluations/claims.holdout.v2.json"] = "0" * 64
    with pytest.raises(ValueError, match="frozen source changed"):
        holdout_runner.verify_freeze(altered)


def test_paired_harness_passes_identical_claim_and_evidence_to_both_strategies():
    freeze = json.loads(holdout_runner.DEFAULT_FREEZE.read_text())
    # This test exercises pairing with fixed responses; source validation is above.
    dataset = load_dataset(holdout_runner.DEFAULT_DATASET)
    calls = []
    def fake_invoke(strategy, claim, documents, timeout_seconds):
        calls.append((strategy, claim, copy.deepcopy(documents), timeout_seconds))
        quote = documents[0]["facts"][0]
        id_for_claim = next(case["id"] for case in dataset["cases"] if case["claim"] == claim)
        expected = next(case["expected_label"] for case in dataset["cases"] if case["id"] == id_for_claim)
        if strategy == "v1":
            verification = {"verdict": expected, "reason": "fixed test response",
                            "evidence_quote": quote if expected in ("supported", "contradicted") else ""}
        else:
            refs = [{"id": 1, "document_id": documents[0]["id"], "fact_index": 1, "quote": quote}]
            if expected == "not_supported":
                refs = []
            verification = {"verdict": expected, "reason": "fixed test response",
                            "evidence_quote": refs[0]["quote"] if refs else "",
                            "evidence_references": refs, "numeric_check": "none", "quantifier_check": "none"}
        return {"strategy": strategy, "model": "qwen3:4b", "model_call_count": 1,
                "model_calls": [{"completed": True}], "verification": verification}
    report = holdout_runner.run_paired(dataset, freeze, 30, invoke=fake_invoke)
    assert report["execution_mode"] == "test_double"
    assert len(calls) == 16
    assert {strategy: sum(call[0] == strategy for call in calls) for strategy in ("v1", "v2")} == {"v1": 8, "v2": 8}
    assert all(call[3] == 30 for call in calls)
    for case in dataset["cases"]:
        matching = [call for call in calls if call[1] == case["claim"]]
        assert len(matching) == 2 and matching[0][2] == matching[1][2]
        assert "expected_label" not in str(matching[0][2])
    for strategy in ("v1", "v2"):
        part = report["strategies"][strategy]
        assert part["metrics"]["accuracy"] == 1
        assert part["observed_model_call_count"] == 8
        assert part["citations"]["citation_validity_rate"] == 1


def test_worker_request_contains_only_strategy_claim_and_evidence(monkeypatch):
    seen = []
    def subprocess_boundary(command, **kwargs):
        seen.append((command, kwargs))
        class Completed:
            stdout = '{"ok":true}'
        return Completed()
    monkeypatch.setattr(holdout_runner.subprocess, "run", subprocess_boundary)
    assert holdout_runner.invoke_worker("v2", "claim", [{"id": "A", "facts": ["fact"]}], 5) == {"ok": True}
    command, kwargs = seen[0]
    assert command[-1] == "evaluations.holdout_worker"
    assert kwargs["timeout"] == 5
    assert json.loads(kwargs["input"]) == {"strategy": "v2", "claim": "claim",
                                           "documents": [{"id": "A", "facts": ["fact"]}]}
