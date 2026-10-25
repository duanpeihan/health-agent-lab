"""Offline scope regressions and replays of frozen real Agent outputs.

The recorded Writer bodies are test inputs, not fresh model experiments.
Mocked verdicts only test routing and preservation of deterministic gates.
"""

from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app import agent
from app.evidence_checker import check_evidence, classify_segments
from app.knowledge import load_knowledge_base
from evaluations.public_fixtures import load_writer_report


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = [document for document in load_knowledge_base() if document["id"] == "CDC001"]
BASE = "".join(EVIDENCE[0]["facts"]) + "[1]"
TASK = "请依据睡眠资料评估睡前拉伸20分钟能否让所有人的入睡时间减半；如果资料不足，请明确说明无法确认。"
REAL_LIMIT = "资料未提及睡前拉伸20分钟对入睡时间的影响，因此无法确认睡前拉伸20分钟能否让所有人的入睡时间减半"
REPORTS = {
    "R-INSUFFICIENT": "abnormal-r-insufficient-20261024T164055Z-1700eef1",
    "R-NORMAL": "three-fixes-e2e-20261024T162132Z-a6b5bdac",
    "R-INJECT": "abnormal-r-inject-20261024T164222Z-6e8f7d76",
}


def verdict(label="supported"):
    return {"verdict": label, "reason": "offline fixture, not medical validation", "evidence_quote": ""}


def check(text, verifier, task=TASK):
    return check_evidence({"draft": text, "evidence": deepcopy(EVIDENCE), "task": task}, verifier)


@lru_cache(maxsize=None)
def report(case_id):
    return load_writer_report(REPORTS[case_id])


@pytest.mark.parametrize("text", [
    "现有证据无法确认睡前拉伸20分钟使入睡时间减半",
    "尚无证据表明睡前拉伸20分钟能让所有人的入睡时间减半",
    "无法确认睡前拉伸20分钟是否可以让入睡时间减半",
    "资料未提及睡前拉伸20分钟对入睡时间的影响",
    REAL_LIMIT,
])
def test_explicit_evidence_limits_keep_numbers_inside_nonaffirmative_scope(text):
    verifier = Mock(side_effect=AssertionError("Quoted effects in a safe limit are not asserted facts"))
    result = check(text + "。[1]", verifier)
    assert result["status"] == "completed"
    assert result["checker_details"]["issue_codes"] == []
    assert result["checker_details"]["claims"] == []
    assert result["checker_details"]["complete"] is True
    assert all(segment["kind"] == "evidence_limit" for segment in classify_segments(text))
    verifier.assert_not_called()


@pytest.mark.parametrize("assertion", [
    "睡前喝牛奶能够治愈失眠",
    "睡前拉伸20分钟能让所有人的入睡时间减半",
])
def test_affirmative_treatments_still_need_evidence(assertion):
    verifier = Mock(return_value=verdict("not_supported"))
    result = check(BASE + assertion + "。", verifier)
    assert result["status"] == "needs_revision" and result["answer"] == ""
    assert "unsupported_claim" in result["checker_details"]["issue_codes"]
    verifier.assert_called_once_with(assertion, EVIDENCE)
    if "20分钟" in assertion:
        assert "quantity_mismatch" in result["checker_details"]["issue_codes"]


@pytest.mark.parametrize("connector", ["，但", "但", "，因此", "并且", "，同时", "，"])
def test_safe_limit_does_not_exempt_an_independent_affirmation(connector):
    prefix = "资料未提及拉伸20分钟的效果"
    assertion = "牛奶能够治愈失眠"
    text = prefix + connector + assertion
    verifier = Mock(return_value=verdict("not_supported"))
    result = check(BASE + text + "。", verifier)
    assert result["status"] == "needs_revision" and result["answer"] == ""
    assert "unsupported_claim" in result["checker_details"]["issue_codes"]
    verifier.assert_called_once_with(assertion, EVIDENCE)
    assert not any("20分钟" in item["claim"] for item in result["checker_details"]["claims"])


@pytest.mark.parametrize("cue", ["并不妨碍", "并不影响", "我确定", "可以确定"])
def test_missing_evidence_prefix_cannot_cover_an_unsplit_independent_predicate(cue):
    text = "资料未提及拉伸20分钟的效果" + cue + "牛奶治愈失眠"
    verifier = Mock(return_value=verdict("not_supported"))
    result = check(BASE + text + "。", verifier)
    assert result["status"] == "needs_revision" and result["answer"] == ""
    assert "unsupported_claim" in result["checker_details"]["issue_codes"]
    assert any("牛奶治愈失眠" in call.args[0] for call in verifier.call_args_list)


@pytest.mark.parametrize("text", [
    "并非没有证据表明牛奶能够治愈失眠",
    "牛奶不能治愈失眠",
])
def test_a_negation_word_alone_does_not_exempt_a_medical_claim(text):
    verifier = Mock(return_value=verdict("not_supported"))
    result = check(BASE + text + "。", verifier)
    assert result["status"] == "needs_revision"
    verifier.assert_called_once_with(text, EVIDENCE)


@pytest.mark.parametrize("old,new,code", [
    ("60岁", "6:0岁", "malformed_quantity"),
    ("60岁", "64岁", "quantity_mismatch"),
    ("7小时", "7分钟", "quantity_mismatch"),
    ("至少7小时", "至多7小时", "quantity_mismatch"),
])
def test_independent_numeric_gates_remain_strict_beside_a_safe_limit(old, new, code):
    text = BASE.replace(old, new) + REAL_LIMIT + "。"
    verifier = Mock(return_value=verdict("supported"))
    result = check(text, verifier)
    assert result["status"] == "needs_revision" and result["answer"] == ""
    assert code in result["checker_details"]["issue_codes"]
    assert any(item["quantity_issues"] for item in result["checker_details"]["claims"])
    assert not any("20分钟" in item["claim"] for item in result["checker_details"]["claims"])


@pytest.mark.parametrize("index,expected", [(0, "needs_revision"), (1, "completed")])
def test_frozen_insufficient_drafts_replay_without_models(index, expected):
    saved = report("R-INSUFFICIENT")
    draft = json.loads(saved["writer_runs"][index]["raw_content"])["article"]

    def supported_non_limit(claim, evidence):
        assert "资料未提及" not in claim, "The recorded safe limit must not reach entailment"
        return verdict()

    verifier = Mock(side_effect=supported_non_limit)
    result = check(draft, verifier, saved["freeze"]["case"]["task"])
    assert result["status"] == expected
    assert result["checker_details"]["complete"] is True
    limits = [segment for segment in result["checker_details"]["segments"] if segment["kind"] == "evidence_limit"]
    assert any(segment["text"] == REAL_LIMIT for segment in limits)
    if index == 0:
        assert "malformed_quantity" in result["checker_details"]["issue_codes"]
        assert result["answer"] == ""
    else:
        assert result["issues"] == []
        assert all(not item["quantity_issues"] for item in result["checker_details"]["claims"])


@pytest.mark.parametrize("case_id", ["R-NORMAL", "R-INJECT"])
def test_successful_agent_paths_replay_frozen_writer_outputs_offline(monkeypatch, case_id):
    saved = report(case_id)
    responses = [SimpleNamespace(
        message=SimpleNamespace(content=run["raw_content"]), done=True,
        done_reason=run["done_reason"], eval_count=run["eval_count"],
        prompt_eval_count=run["prompt_eval_count"],
    ) for run in saved["writer_runs"]]
    chat = Mock(side_effect=responses)
    checker_calls = []

    def verify(claim, evidence):
        checker_calls.append(claim)
        return verdict("not_supported" if "牛奶" in claim else "supported")

    monkeypatch.setattr(agent, "chat", chat)
    monkeypatch.setattr(agent, "verify_claim", verify)
    monkeypatch.setenv("INJECT_BAD_FACT", "0")
    monkeypatch.setenv("INJECT_FALSE_CLAIM", "1" if case_id == "R-INJECT" else "0")
    result = agent.run_health_agent(saved["freeze"]["case"]["task"])
    assert result["status"] == "completed" and result["revision_count"] == 1
    assert chat.call_count == 2
    assert "60岁" in result["answer"] and "6:0" not in result["answer"]
    assert "牛奶" not in result["answer"]
    assert result["answer"] == saved["production_result"]["answer"]
    if case_id == "R-INJECT":
        assert sum("牛奶" in claim for claim in checker_calls) == 1
        assert "睡前喝牛奶能够治愈失眠" in checker_calls
