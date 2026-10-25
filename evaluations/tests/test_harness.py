"""Deterministic checks of the evaluator, never model-quality measurements.

Predictions below are explicitly specified test inputs. No fixture writes a
benchmark report or uses its expected labels to manufacture model predictions.
"""

import copy
import hashlib
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from evaluations.dataset import LABELS, load_dataset, resolve_documents
from evaluations.metrics import summarize
from evaluations import run


@pytest.fixture
def dataset_files(tmp_path):
    knowledge = [{
        "id": "KB1", "topic": "fixture", "source": "test fixture",
        "facts": ["成年人通常建议每晚睡眠至少7小时。", "睡眠质量也很重要。"],
    }]
    knowledge_path = tmp_path / "knowledge.json"
    knowledge_path.write_text(json.dumps(knowledge, ensure_ascii=False), encoding="utf-8")
    common = {
        "rationale": "A deliberately small independent schema fixture.",
        "tags": ["fixture"], "label_status": "pending_human_review", "reviewer": None,
        "evidence_origin": "knowledge_base",
        "evidence_refs": [{"document_id": "KB1", "fact_indices": [1]}],
        "evidence_fixture": [], "expected_evidence_quotes": [],
    }
    cases = []
    specifications = [
        ("case-1", "成年人每晚通常应睡够7小时。", "supported"),
        ("case-2", "成年人每晚只需睡3小时。", "contradicted"),
        ("case-3", "睡眠能治愈所有疾病。", "not_supported"),
        ("case-4", "每个人均应采用同样作息。", "uncertain"),
        ("case-5", "没有材料可以判断这项疗效。", "not_supported"),
    ]
    for case_id, claim, label in specifications:
        case = dict(copy.deepcopy(common), id=case_id, claim=claim, expected_label=label)
        if label in ("supported", "contradicted"):
            case["expected_evidence_quotes"] = [knowledge[0]["facts"][0]]
        if case_id == "case-4":
            case.update(
                evidence_origin="synthetic_ambiguous", evidence_refs=[],
                evidence_fixture=[{
                    "id": "SYN1", "source": "synthetic_ambiguous",
                    "facts": ["部分人在特定情境下可能适合这一作息。"],
                }],
            )
        if case_id == "case-5":
            case.update(evidence_origin="empty_evidence", evidence_refs=[])
        cases.append(case)
    payload = {
        "schema_version": "1.0", "dataset_id": "independent-unit-fixture",
        "label_status": "pending_human_review",
        "knowledge_sha256": hashlib.sha256(knowledge_path.read_bytes()).hexdigest(),
        "cases": cases,
    }
    dataset_path = tmp_path / "claims.json"
    dataset_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload, dataset_path, knowledge_path


@pytest.fixture
def dataset(dataset_files, monkeypatch):
    _, dataset_path, knowledge_path = dataset_files
    # The Git command is unrelated to these unit assertions and need not run.
    monkeypatch.setattr(run, "_git_state", lambda: {"head": "unit-fixture", "status": "test"})
    return load_dataset(dataset_path, knowledge_path)


def _write_payload(payload, path):
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _response(verdict, quote="", reason="Explicit unit-test response"):
    return {"model": "unit-test-double", "verification": {
        "verdict": verdict, "reason": reason, "evidence_quote": quote,
    }}


def test_hand_calculated_metrics_and_latency():
    # Eight manually fixed observations include both a disagreement and a failure.
    pairs = [
        ("supported", "supported"), ("supported", "not_supported"),
        ("not_supported", "not_supported"), ("not_supported", "contradicted"),
        ("contradicted", "contradicted"), ("contradicted", "supported"),
        ("uncertain", "uncertain"), ("uncertain", "error"),
    ]
    rows = [{"expected_label": expected, "predicted_label": predicted,
             "elapsed_seconds": float(i), "error": None}
            for i, (expected, predicted) in enumerate(pairs, 1)]
    metrics = summarize(rows)
    assert (metrics["total"], metrics["correct"], metrics["error_count"]) == (8, 4, 1)
    assert metrics["accuracy"] == 0.5
    assert metrics["macro_f1"] == pytest.approx(13 / 24)
    for label in ("supported", "not_supported", "contradicted"):
        assert metrics["per_class"][label] == {
            "precision": 0.5, "recall": 0.5, "f1": 0.5,
            "support": 2, "tp": 1, "fp": 1, "fn": 1,
        }
    assert metrics["per_class"]["uncertain"] == {
        "precision": 1.0, "recall": 0.5, "f1": pytest.approx(2 / 3),
        "support": 2, "tp": 1, "fp": 0, "fn": 1,
    }
    assert metrics["confusion_matrix"] == {
        "supported": {"supported": 1, "not_supported": 1, "contradicted": 0, "uncertain": 0, "error": 0},
        "not_supported": {"supported": 0, "not_supported": 1, "contradicted": 1, "uncertain": 0, "error": 0},
        "contradicted": {"supported": 1, "not_supported": 0, "contradicted": 1, "uncertain": 0, "error": 0},
        "uncertain": {"supported": 0, "not_supported": 0, "contradicted": 0, "uncertain": 1, "error": 1},
    }
    assert metrics["latency_seconds"] == {
        "total": 36.0, "mean": 4.5, "p50": 4.5, "p95": 8.0, "max": 8.0,
    }


def test_empty_metrics_are_defined_without_nan():
    metrics = summarize([])
    assert metrics["accuracy"] == metrics["macro_f1"] == 0
    assert metrics["total"] == metrics["correct"] == metrics["error_count"] == 0
    assert all(value == 0 for value in metrics["latency_seconds"].values())
    assert all(item["support"] == item["precision"] == item["recall"] == item["f1"] == 0
               for item in metrics["per_class"].values())


def test_load_dataset_pins_bytes_and_preserves_review_status(dataset_files):
    payload, path, knowledge_path = dataset_files
    loaded = load_dataset(path, knowledge_path)
    assert LABELS == ("supported", "not_supported", "contradicted", "uncertain")
    assert loaded["cases"] == payload["cases"]
    assert loaded["label_status"] == "pending_human_review"
    assert loaded["_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_knowledge_drift_requires_label_review(dataset_files):
    _, path, knowledge_path = dataset_files
    knowledge_path.write_text(knowledge_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_dataset(path, knowledge_path)


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(schema_version="2.0"),
    lambda d: d.update(dataset_id=""),
    lambda d: d.update(label_status="reviewed"),
    lambda d: d.update(cases=d["cases"][:2]),
    lambda d: d["cases"][1].update(id="case-1"),
    lambda d: d["cases"][0].update(expected_label="unknown"),
    lambda d: d["cases"][0].update(claim=" "),
    lambda d: d["cases"][0].update(rationale=""),
    lambda d: d["cases"][0].update(tags=[]),
    lambda d: d["cases"][0].update(reviewer="Already reviewed"),
    lambda d: d["cases"][0].update(evidence_origin="unmarked_synthetic"),
    lambda d: d["cases"][0]["evidence_refs"][0].update(document_id="missing"),
    lambda d: d["cases"][0]["evidence_refs"][0].update(fact_indices=[0]),
    lambda d: d["cases"][0]["evidence_refs"][0].update(fact_indices=[3]),
    lambda d: d["cases"][0]["evidence_refs"][0].update(fact_indices=[True]),
    lambda d: d["cases"][0]["evidence_refs"][0].update(fact_indices=[1, 1]),
    lambda d: d["cases"][0].update(expected_evidence_quotes=["成年人应该睡7小时。"]),
    lambda d: d["cases"][2].update(expected_evidence_quotes=["成年人通常建议每晚睡眠至少7小时。"]),
    lambda d: d["cases"][3]["evidence_fixture"][0].update(source="unmarked"),
    lambda d: d["cases"][4].update(evidence_fixture=[{"facts": ["unexpected"]}]),
])
def test_rejects_invalid_annotations_and_unpinned_evidence(dataset_files, mutate):
    payload, path, knowledge_path = dataset_files
    mutate(payload)
    _write_payload(payload, path)
    with pytest.raises(ValueError):
        load_dataset(path, knowledge_path)


@pytest.mark.parametrize("replacement", [None, [], "invalid", 1])
def test_dataset_root_must_be_an_object(dataset_files, replacement):
    _, path, knowledge_path = dataset_files
    _write_payload(replacement, path)
    with pytest.raises(ValueError):
        load_dataset(path, knowledge_path)


@pytest.mark.parametrize("location", ["case", "reference"])
def test_nested_annotation_objects_are_validated(dataset_files, location):
    payload, path, knowledge_path = dataset_files
    if location == "case":
        payload["cases"][0] = None
    else:
        payload["cases"][0]["evidence_refs"][0] = "not an object"
    _write_payload(payload, path)
    with pytest.raises(ValueError):
        load_dataset(path, knowledge_path)


def test_resolve_only_selected_facts_without_mutating_knowledge(dataset):
    knowledge = dataset["_knowledge"]
    before = copy.deepcopy(knowledge)
    documents = resolve_documents(dataset["cases"][0], knowledge)
    assert documents[0]["facts"] == ["成年人通常建议每晚睡眠至少7小时。"]
    documents[0]["facts"][0] = "mutation"
    assert knowledge == before
    synthetic = resolve_documents(dataset["cases"][3], knowledge)
    assert synthetic[0]["source"] == "synthetic_ambiguous"
    synthetic[0]["facts"].append("mutation")
    assert len(dataset["cases"][3]["evidence_fixture"][0]["facts"]) == 1
    assert resolve_documents(dataset["cases"][4], knowledge) == []


def test_smoke_is_three_fixed_first_cases_and_full_is_all(dataset):
    assert [case["id"] for case in run.select_cases(dataset, "smoke")] == ["case-1", "case-2", "case-3"]
    assert len(run.select_cases(dataset, "full")) == 5
    with pytest.raises(ValueError):
        run.select_cases(dataset, "quick")


def test_injected_evaluation_is_explicitly_a_test_double_and_audits_quotes(dataset):
    # Deliberate fixed responses: a correct label with a bad quote, a correct
    # contradiction, and a wrong label. This is not a model benchmark.
    responses = iter([
        _response("supported", "paraphrase is not an evidence quote"),
        _response("contradicted", "成年人通常建议每晚睡眠至少7小时。"),
        _response("supported", "成年人通常建议每晚睡眠至少7小时。"),
    ])
    calls, observed = [], []
    def invoke(claim, documents, timeout_seconds):
        calls.append((claim, copy.deepcopy(documents), timeout_seconds))
        return next(responses)
    report = run.run_evaluation(dataset, "smoke", 12, invoke=invoke, on_result=observed.append)
    assert report["execution_mode"] == "test_double"
    assert report["model"] == ["unit-test-double"]
    assert report["provisional"] is True
    assert report["case_count"] == len(calls) == len(observed) == 3
    assert report["metrics"]["accuracy"] == pytest.approx(2 / 3)
    assert [row["evidence_quote_matches_expected"] for row in report["results"]] == [False, True, False]
    assert [row["id"] for row in report["errors"]] == ["case-1", "case-3"]
    assert report["metrics"]["by_evidence_origin"]["knowledge_base"]["total"] == 3
    assert all(timeout == 12 for _, _, timeout in calls)
    assert all("expected_label" not in doc for _, docs, _ in calls for doc in docs)
    assert all(row["elapsed_seconds"] >= 0 for row in report["results"])


def test_invocation_failures_remain_in_accuracy_and_class_recall_denominators(dataset):
    outcomes = iter([
        RuntimeError("unit fixture failure"),
        subprocess.TimeoutExpired("unit worker", 1),
        _response("not_supported"),
    ])
    def invoke(*args):
        item = next(outcomes)
        if isinstance(item, Exception):
            raise item
        return item
    report = run.run_evaluation(dataset, "smoke", 1, invoke=invoke)
    assert report["metrics"]["total"] == 3
    assert report["metrics"]["error_count"] == 2
    assert report["metrics"]["accuracy"] == pytest.approx(1 / 3)
    assert report["metrics"]["per_class"]["supported"]["fn"] == 1
    assert report["metrics"]["per_class"]["contradicted"]["fn"] == 1
    assert report["results"][1]["error"] == {
        "type": "TimeoutExpired", "message": "Verifier worker exceeded 1 seconds",
    }


@pytest.mark.parametrize("response", [
    None, {}, {"verification": {}},
    {"model": "unit-test-double", "verification": {"verdict": "other"}},
    {"model": "unit-test-double", "verification": {"verdict": "supported", "reason": None, "evidence_quote": ""}},
    {"model": "", "verification": {"verdict": "supported", "reason": "test", "evidence_quote": ""}},
])
def test_malformed_worker_responses_are_errors_not_silent_predictions(dataset, response):
    report = run.run_evaluation(dataset, "smoke", 1, invoke=lambda *args: response)
    assert report["metrics"]["accuracy"] == 0
    assert report["metrics"]["error_count"] == 3
    assert all(row["predicted_label"] == "error" and row["error"]["type"] == "ValueError"
               for row in report["results"])


def test_empty_evidence_path_is_visible_and_model_mode_is_not_inferred_from_verdicts(dataset):
    responses = iter([
        _response("not_supported"), _response("not_supported"),
        _response("not_supported"), _response("uncertain"), _response("not_supported"),
    ])
    report = run.run_evaluation(dataset, "full", 1, invoke=lambda *args: next(responses))
    assert report["execution_mode"] == "test_double"
    assert report["model_requests_expected"] == 4
    assert report["observed_model_call_count"] == 0
    assert report["unknown_model_call_count_cases"] == 5
    assert report["no_evidence_shortcut_count"] == 1
    assert report["results"][-1]["model_call_expected"] is False
    assert report["results"][-1]["evidence_quote_matches_expected"] is True
    assert report["metrics"]["by_evidence_origin"]["empty_evidence"]["total"] == 1


def test_default_invocation_path_selects_real_worker_mode_without_calling_a_model(dataset, monkeypatch):
    # Replace the process boundary solely to verify default dispatch and its
    # mode flag. The test produces no on-disk report and makes no quality claim.
    calls = []
    def worker_boundary(*args):
        calls.append(args)
        raise RuntimeError("unit-test blocked worker boundary")
    monkeypatch.setattr(run, "invoke_verifier", worker_boundary)
    report = run.run_evaluation(dataset, "smoke", 1)
    assert report["execution_mode"] == "real_model"
    assert len(calls) == 3
    assert report["model"] == []
    assert report["metrics"]["error_count"] == 3


def test_default_path_counts_observed_calls_instead_of_expected_requests(dataset, monkeypatch):
    def worker_boundary(claim, documents, timeout_seconds):
        response = _response("not_supported")
        count = int(bool(documents))
        response.update(model_call_count=count, model_calls=[{
            "model": "unit-test-double", "completed": True, "elapsed_seconds": 0.01,
        }] if count else [], ollama_python_version="unit-test", model_digest=None)
        return response
    monkeypatch.setattr(run, "invoke_verifier", worker_boundary)
    report = run.run_evaluation(dataset, "full", 1)
    assert report["execution_mode"] == "real_model"
    assert report["model_requests_expected"] == report["observed_model_call_count"] == 4
    assert report["unknown_model_call_count_cases"] == 0
    assert [row["observed_model_call_count"] for row in report["results"]] == [1, 1, 1, 1, 0]
    assert report["metrics"]["observed_model_cases"]["total"] == 4


@pytest.mark.parametrize("observation", [
    {}, {"model_call_count": "1", "model_calls": []},
    {"model_call_count": 2, "model_calls": []},
    {"model_call_count": 1, "model_calls": [{"model": "unit-test-double", "completed": False}]},
    {"model_call_count": 1, "model_calls": ["invalid observation"]},
])
def test_default_path_rejects_bad_model_observations_per_case(dataset, monkeypatch, observation):
    response = dict(_response("not_supported"), **observation)
    monkeypatch.setattr(run, "invoke_verifier", lambda *args: copy.deepcopy(response))
    report = run.run_evaluation(dataset, "smoke", 1)
    assert report["metrics"]["error_count"] == 3
    assert report["unknown_model_call_count_cases"] == 3
    assert report["observed_model_call_count"] == 0


def test_worker_process_receives_only_claim_and_evidence_and_has_hard_timeout(monkeypatch):
    calls = []
    def subprocess_boundary(command, **kwargs):
        calls.append((command, kwargs))
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])
    monkeypatch.setattr(run.subprocess, "run", subprocess_boundary)
    with pytest.raises(subprocess.TimeoutExpired):
        run.invoke_verifier("fixture claim", [{"facts": ["fixture evidence"]}], 0.25)
    command, kwargs = calls[0]
    assert command == [sys.executable, "-m", "evaluations.worker"]
    assert kwargs["timeout"] == 0.25
    assert kwargs["check"] is True and kwargs["capture_output"] is True
    assert json.loads(kwargs["input"]) == {
        "claim": "fixture claim", "documents": [{"facts": ["fixture evidence"]}],
    }


def test_worker_process_decodes_explicit_json_response(monkeypatch):
    monkeypatch.setattr(run.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(stdout=json.dumps(_response("uncertain"))))
    assert run.invoke_verifier("fixture", [], 1)["verification"]["verdict"] == "uncertain"


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_timeout_is_rejected_before_invocation(dataset, timeout):
    def unexpected(*args):
        pytest.fail("invalid timeout must not invoke a worker")
    with pytest.raises(ValueError, match="finite and positive"):
        run.run_evaluation(dataset, "smoke", timeout, invoke=unexpected)


def test_cli_validate_and_help_never_call_verifier(dataset, monkeypatch, capsys):
    def unexpected(*args, **kwargs):
        pytest.fail("validation/help must not invoke a model")
    monkeypatch.setattr(run, "load_dataset", lambda *args: dataset)
    monkeypatch.setattr(run, "run_evaluation", unexpected)
    monkeypatch.setattr(run, "invoke_verifier", unexpected)
    assert run.main(["--validate-only"]) == 0
    validation = json.loads(capsys.readouterr().out)
    assert validation["model_calls"] == 0 and validation["cases"] == 5
    with pytest.raises(SystemExit) as exit_info:
        run.main(["--help"])
    assert exit_info.value.code == 0


def test_cli_requires_explicit_mode(monkeypatch):
    monkeypatch.setattr(run, "load_dataset", lambda *args: pytest.fail("no mode must stop before dataset loading"))
    with pytest.raises(SystemExit) as exit_info:
        run.main([])
    assert exit_info.value.code == 2
