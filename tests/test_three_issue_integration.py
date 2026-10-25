"""Offline integration: short completeness, quantity fidelity, and whole sentences."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app import agent
from app.evidence_checker import check_evidence
from app.knowledge import load_knowledge_base


TASK = "请依据给定资料写18至60岁成年人睡眠科普，说明推荐时长与睡眠质量，避免增加无证据结论。"
EVIDENCE = [doc for doc in load_knowledge_base() if doc["id"] == "CDC001"]
SHORT = "18至60岁成年人每晚建议睡眠时长至少7小时[1]。健康睡眠需同时满足充足时长与良好质量。"
SUPPORTED = {"verdict": "supported", "reason": "fixed offline fixture", "evidence_quote": ""}


def check(text, verify=None, task=TASK):
    return check_evidence({"task": task, "draft": text, "evidence": EVIDENCE},
                          verify or Mock(return_value=SUPPORTED))


def test_short_complete_answer_has_no_artificial_minimum_length():
    assert len(SHORT) < 60
    verify = Mock(return_value=SUPPORTED)
    result = check(SHORT, verify)
    assert result["status"] == "completed"
    assert result["issues"] == []
    assert result["checker_details"]["content_assessment"]["missing"] == []
    assert verify.call_args_list[-1].args[0] == "健康睡眠需同时满足充足时长与良好质量"
    assert "健康睡眠需" not in [call.args[0] for call in verify.call_args_list]


@pytest.mark.parametrize("text", [
    SHORT.replace("60岁", "6:0岁"),
    SHORT.replace("60岁", "64岁"),
    SHORT.replace("7小时", "7分钟"),
    SHORT.replace("至少7小时", "至多7小时"),
])
def test_numeric_gate_blocks_even_when_the_model_says_supported(text):
    result = check(text)
    assert result["status"] != "completed"
    assert result["answer"] == ""
    assert any(item["quantity_issues"] for item in result["checker_details"]["claims"])
    assert text != SHORT  # The checker never silently repairs the source text.


def test_missing_requested_quality_is_incomplete_not_just_short():
    result = check("18至60岁成年人通常建议每晚睡眠至少7小时。[1]")
    assert result["status"] == "needs_revision"
    content = result["checker_details"]["content_assessment"]
    assert content["missing"] == ["sleep_quality"]
    assert content["terminal"] is False


def test_long_form_with_only_summary_evidence_stops_before_any_model(monkeypatch):
    writer = Mock(side_effect=AssertionError("Insufficient evidence must not prompt padding"))
    monkeypatch.setattr(agent, "chat", writer)
    result = agent.run_health_agent("请依据睡眠资料写一篇2000字长文，详细说明时长与质量。")
    assert result["status"] == "insufficient_content"
    assert result["revision_count"] == 0
    assert "不足" in result["answer"] and result["evidence_count"] == 1
    writer.assert_not_called()
    assert set(result) == {"status", "answer", "evidence_count", "revision_count", "duration_seconds"}


def test_long_form_terminal_status_survives_stop_node():
    result = check(SHORT, task="请用睡眠资料写1000字长文。")
    assert result["status"] == "insufficient_content"
    assert agent.route_checker({**result, "revision_count": 0}) == "stop"
    assert agent.stop_node(result) == {"status": "insufficient_content", "answer": result["answer"]}


def test_entire_compound_claim_reaches_verifier_and_unsupported_tail_blocks():
    tail = "健康睡眠需同时满足充足时长与良好质量，但牛奶能够治愈失眠"
    verify = Mock(side_effect=lambda claim, evidence: {**SUPPORTED,
        "verdict": "not_supported" if "牛奶" in claim else "supported"})
    result = check("18至60岁成年人每晚建议睡眠至少7小时。[1]" + tail + "。", verify)
    assert result["status"] == "needs_revision"
    assert tail in [call.args[0] for call in verify.call_args_list]


def test_graph_accepts_short_complete_article_with_unchanged_api(monkeypatch):
    response = SimpleNamespace(message=SimpleNamespace(content=json.dumps({"article": SHORT}, ensure_ascii=False)),
                               done=True, done_reason="stop", eval_count=42)
    writer = Mock(return_value=response)
    monkeypatch.setattr(agent, "chat", writer)
    monkeypatch.setattr(agent, "verify_claim", Mock(return_value=SUPPORTED))
    result = agent.run_health_agent(TASK)
    assert result["status"] == "completed" and result["revision_count"] == 0
    assert result["answer"].startswith(SHORT)
    writer.assert_called_once()
