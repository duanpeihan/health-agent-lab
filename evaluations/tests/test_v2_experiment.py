"""Offline checks for V2 citation mapping, bounded comparison and holdout isolation."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluations import compare_v2, strategy, v2_strategy
from evaluations.dataset import DEFAULT_DATASET, load_dataset
from evaluations.public_fixtures import historical_source_bytes


def _model_result(**overrides):
    value = {"verdict": "supported", "reason": "两项事实共同支持", "numeric_check": "compatible",
             "quantifier_check": "none", "evidence_ids": [1, 2]}
    value.update(overrides)
    return SimpleNamespace(message=SimpleNamespace(content=json.dumps(value, ensure_ascii=False)))


def test_v2_passes_only_claim_and_numbered_facts_then_maps_structured_citations(monkeypatch):
    calls = []
    def fake_chat(**kwargs):
        calls.append(kwargs)
        return _model_result()
    monkeypatch.setattr(v2_strategy, "chat", fake_chat)
    docs = [{"id": "A", "facts": ["第一条事实。"]}, {"id": "B", "facts": ["第二条事实。"]}]
    result = v2_strategy.verify_claim_v2("受控主张。", docs)
    assert result["evidence_references"] == [
        {"id": 1, "document_id": "A", "fact_index": 1, "quote": "第一条事实。"},
        {"id": 2, "document_id": "B", "fact_index": 1, "quote": "第二条事实。"},
    ]
    assert result["evidence_quote"] == "第一条事实。"
    assert calls[0]["model"] == v2_strategy.MODEL
    assert calls[0]["options"] == {"temperature": 0}
    assert "[1] 第一条事实。" in calls[0]["messages"][1]["content"]
    assert "expected_label" not in str(calls[0])


@pytest.mark.parametrize("bad", [
    {"evidence_ids": [0]}, {"evidence_ids": [2]}, {"evidence_ids": [1, 1]},
    {"evidence_ids": []}, {"evidence_ids": [True]},
    {"numeric_check": "direct_conflict", "verdict": "supported"},
])
def test_v2_rejects_unverifiable_or_inconsistent_model_outputs(monkeypatch, bad):
    monkeypatch.setattr(v2_strategy, "chat", lambda **kwargs: _model_result(evidence_ids=[1], **bad)
                        if "evidence_ids" not in bad else _model_result(**bad))
    with pytest.raises(ValueError):
        v2_strategy.verify_claim_v2("主张", [{"id": "A", "facts": ["事实"]}])


def test_v2_empty_evidence_short_circuit_does_not_call_model(monkeypatch):
    monkeypatch.setattr(v2_strategy, "chat", lambda **kwargs: pytest.fail("no model call expected"))
    result = v2_strategy.verify_claim_v2("主张", [])
    assert result["verdict"] == "not_supported"
    assert result["evidence_references"] == []


def test_offline_strategy_switch_keeps_production_v1():
    from app.claim_verifier import verify_claim
    assert strategy.get_verifier("v1") is verify_claim
    assert strategy.get_verifier("v2") is v2_strategy.verify_claim_v2
    with pytest.raises(ValueError):
        strategy.get_verifier("v3")


def test_holdout_is_valid_and_disjoint_from_original_seed():
    v1 = load_dataset(DEFAULT_DATASET)
    holdout = load_dataset(Path(DEFAULT_DATASET).with_name("claims.holdout.v2.json"))
    assert len(holdout["cases"]) == 8
    assert {item["id"] for item in holdout["cases"]}.isdisjoint({item["id"] for item in v1["cases"]})
    assert {item["claim"] for item in holdout["cases"]}.isdisjoint({item["claim"] for item in v1["cases"]})
    assert {item["expected_label"] for item in holdout["cases"]} == set(v2_strategy.LABELS)


def test_citation_audit_rejects_fabricated_and_accepts_real_references():
    docs = [{"id": "D", "facts": ["原文"]}]
    ref = {"id": 1, "document_id": "D", "fact_index": 1, "quote": "原文"}
    good = compare_v2.citation_audit("uncertain", {"evidence_quote": "原文", "evidence_references": [ref]}, docs, "v2")
    bad = compare_v2.citation_audit("uncertain", {"evidence_quote": "伪造", "evidence_references": [dict(ref, quote="伪造")]}, docs, "v2")
    assert good["valid"] is True and good["required_citation_satisfied"] is True
    assert bad["valid"] is False and bad["required_citation_satisfied"] is False


def test_bounded_comparison_reuses_v1_archive_and_calls_only_v2(monkeypatch, tmp_path):
    dataset = load_dataset(DEFAULT_DATASET)
    baseline = json.loads(compare_v2.DEFAULT_BASELINE.read_text())
    calls = []
    def fake_invoke(claim, documents, timeout):
        calls.append(claim)
        quote = documents[0]["facts"][0]
        return {"model": "test-double", "model_call_count": 1,
                "model_calls": [{"completed": True}],
                "verification": {"verdict": "supported", "reason": "固定响应", "evidence_quote": quote,
                                 "evidence_references": [{"id": 1, "document_id": documents[0]["id"],
                                                          "fact_index": 1, "quote": quote}],
                                 "numeric_check": "none", "quantifier_check": "none"}}
    with pytest.raises(ValueError, match="V1 production verifier has changed"):
        compare_v2.compare(dataset, baseline, ["S001"], 10, invoke=fake_invoke)
    assert calls == []
    source = tmp_path / "app/claim_verifier.py"
    source.parent.mkdir()
    source.write_bytes(historical_source_bytes("app/claim_verifier.py"))
    # Fixed model test against the original source identity; real guards stay strict.
    monkeypatch.setattr(compare_v2, "ROOT", tmp_path)
    report = compare_v2.compare(dataset, baseline, ["S001"], 10, invoke=fake_invoke)
    assert len(calls) == 1
    assert report["v1"]["observed_model_call_count"] == 1
    assert report["v2"]["citations"]["citation_validity_rate"] == 1
    with pytest.raises(ValueError, match="1 to 8"):
        compare_v2.compare(dataset, baseline, ["S001"] * 9, 10, invoke=fake_invoke)
