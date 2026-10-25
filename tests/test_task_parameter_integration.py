"""Offline release-gate regressions; frozen JSON is replayed, never regenerated."""

from functools import lru_cache
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app import agent, evidence_checker
from app.evidence_checker import check_evidence
from app.knowledge import load_knowledge_base
from app.writer import SAFE_STOP
from evaluations.public_fixtures import load_writer_report


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = [doc for doc in load_knowledge_base() if doc["id"] == "CDC001"]
TASK = "请依据睡眠资料评估睡前拉伸20分钟能否让所有人的入睡时间减半；如果资料不足，请明确说明无法确认，不补充外部结论。"
BAD = "资料未提及睡前拉伸20分钟的效果，因此无法确认睡前拉伸2.0分钟能否让所有人的入睡时间减半。[1]"
CORRECT = "资料未提及睡前拉伸20分钟的效果，因此无法确认睡前拉伸20分钟能否让所有人的入睡时间减半。[1]"
OMITTED = "现有资料不足，无法确认该拉伸方案能否让所有人的入睡时间减半。[1]"
API_FIELDS = {"status", "answer", "evidence_count", "revision_count", "duration_seconds"}
REPORTS = {
    "R-INSUFFICIENT": "evidence-limit-r-insufficient-20261024T165857Z-9816f8c4",
    "R-NORMAL": "three-fixes-e2e-20261024T162132Z-a6b5bdac",
    "R-INJECT": "abnormal-r-inject-20261024T164222Z-6e8f7d76",
}


def verdict(label="supported"):
    return {"verdict": label, "reason": "offline fixture, not medical validation", "evidence_quote": ""}


@lru_cache(maxsize=None)
def frozen_report(case_id):
    return load_writer_report(REPORTS[case_id])


def response(article):
    return SimpleNamespace(
        message=SimpleNamespace(content=json.dumps({"article": article}, ensure_ascii=False)),
        done=True, done_reason="stop", eval_count=30, prompt_eval_count=100,
    )


def install_graph_doubles(monkeypatch, articles, verifier=None):
    """Trace the real graph's Checker while replacing only model transport."""
    chat = Mock(side_effect=[response(article) for article in articles])
    verify = verifier if verifier is not None else Mock(
        side_effect=AssertionError("A pure evidence-limit answer must not undergo entailment"))
    checks = []

    def observed_check(state, verify_fn):
        update = check_evidence(state, verify_fn)
        checks.append(update)
        return update

    monkeypatch.setenv("INJECT_BAD_FACT", "0")
    monkeypatch.setenv("INJECT_FALSE_CLAIM", "0")
    monkeypatch.setattr(agent, "chat", chat)
    monkeypatch.setattr(agent, "verify_claim", verify)
    monkeypatch.setattr(agent, "check_evidence", observed_check)
    return chat, verify, checks


def test_recorded_twenty_to_two_decimal_draft_cannot_be_published():
    saved = frozen_report("R-INSUFFICIENT")
    article = json.loads(saved["writer_runs"][-1]["raw_content"])["article"]
    assert "20分钟" in article and "2.0分钟" in article
    verifier = Mock(return_value=verdict())
    result = check_evidence({"task": saved["freeze"]["case"]["task"],
                             "draft": article, "evidence": EVIDENCE}, verifier)
    details = result["checker_details"]
    assert result["status"] == "needs_revision" and result["answer"] == ""
    assert details["task_parameter_fidelity"]["passed"] is False
    assert details["task_parameter_fidelity"]["issues"]
    limits = [segment for segment in details["segments"] if "2.0分钟" in segment["text"]]
    assert len(limits) == 1 and limits[0]["kind"] == "evidence_limit"
    assert all("2.0分钟" not in call.args[0] for call in verifier.call_args_list)
    assert all(not item["quantity_issues"] for item in details["claims"])


@pytest.mark.parametrize("repaired", [CORRECT, OMITTED])
def test_one_reflection_can_correct_or_omit_the_user_parameter(monkeypatch, repaired):
    chat, verifier, checks = install_graph_doubles(monkeypatch, [BAD, repaired])
    result = agent.run_health_agent(TASK)
    assert set(result) == API_FIELDS
    assert result["status"] == "completed" and result["revision_count"] == 1
    assert chat.call_count == 2 and len(checks) == 2
    assert checks[0]["checker_details"]["task_parameter_fidelity"]["passed"] is False
    assert checks[0]["answer"] == ""
    assert checks[1]["checker_details"]["task_parameter_fidelity"]["passed"] is True
    assert checks[1]["issues"] == []
    assert result["answer"].startswith(repaired)
    assert "2.0分钟" not in result["answer"]
    verifier.assert_not_called()


def test_repeated_parameter_error_safely_stops_after_only_one_reflection(monkeypatch):
    chat, verifier, checks = install_graph_doubles(monkeypatch, [BAD, BAD])
    result = agent.run_health_agent(TASK)
    assert set(result) == API_FIELDS
    assert result["status"] == "evaluation_failed" and result["revision_count"] == 1
    assert result["answer"] == SAFE_STOP and "2.0分钟" not in result["answer"]
    assert chat.call_count == 2 and len(checks) == 2
    assert all(check["checker_details"]["task_parameter_fidelity"]["passed"] is False for check in checks)
    assert all(check["answer"] == "" and check["status"] != "completed" for check in checks)
    verifier.assert_not_called()


@pytest.mark.parametrize("verifier_label", ["supported", "not_supported"])
def test_copying_a_user_hypothesis_correctly_does_not_make_it_evidence(verifier_label):
    article = "睡前拉伸20分钟能让所有人的入睡时间减半。[1]"
    verifier = Mock(return_value=verdict(verifier_label))
    result = check_evidence({"task": TASK, "draft": article, "evidence": EVIDENCE}, verifier)
    details = result["checker_details"]
    assert details["task_parameter_fidelity"]["passed"] is True
    assert result["status"] == "needs_revision" and result["answer"] == ""
    assert "quantity_mismatch" in details["issue_codes"]
    assert details["segments"][0]["kind"] == "medical_claim"
    verifier.assert_called_once_with(article.removesuffix("。[1]"), EVIDENCE)


@pytest.mark.parametrize("case_id", ["R-NORMAL", "R-INJECT"])
def test_existing_successful_production_paths_survive_the_new_release_gate(monkeypatch, case_id):
    saved = frozen_report(case_id)
    articles = [json.loads(run["raw_content"])["article"] for run in saved["writer_runs"]]
    verifier = Mock(side_effect=lambda claim, evidence: verdict("not_supported" if "牛奶" in claim else "supported"))
    chat, _, checks = install_graph_doubles(monkeypatch, articles, verifier)
    if case_id == "R-INJECT":
        monkeypatch.setenv("INJECT_FALSE_CLAIM", "1")
    result = agent.run_health_agent(saved["freeze"]["case"]["task"])
    assert set(result) == API_FIELDS
    assert result["status"] == "completed" and result["revision_count"] == 1
    assert result["answer"] == saved["production_result"]["answer"]
    assert chat.call_count == 2 and len(checks) == 2
    assert checks[-1]["checker_details"]["task_parameter_fidelity"]["passed"] is True
    assert checks[-1]["checker_details"]["complete"] is True
    assert checks[-1]["checker_details"]["unchecked_claims"] == []
    assert checks[-1]["issues"] == []
    if case_id == "R-INJECT":
        assert sum("牛奶" in call.args[0] for call in verifier.call_args_list) == 1
        assert "牛奶" not in result["answer"]


def test_a_parameter_gate_exception_is_a_safe_failure_without_reflection(monkeypatch):
    chat, verifier, checks = install_graph_doubles(monkeypatch, [CORRECT])
    monkeypatch.setattr(evidence_checker, "check_task_parameter_fidelity",
                        Mock(side_effect=RuntimeError("synthetic parameter gate failure")))
    result = agent.run_health_agent(TASK)
    assert set(result) == API_FIELDS
    assert result["status"] == "evaluation_failed" and result["revision_count"] == 0
    assert result["answer"] == SAFE_STOP
    assert chat.call_count == 1 and len(checks) == 1
    assert checks[0]["checker_details"]["complete"] is False
    assert "task_parameter_check_error" in checks[0]["checker_details"]["issue_codes"]
    verifier.assert_not_called()
