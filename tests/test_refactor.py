"""Offline compatibility and graph-flow regression tests, not medical evaluation."""

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
import pytest

from app import agent, api, claim_verifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = {
    "id": "TEST001",
    "topic": "流程测试资料",
    "keywords": ["睡眠"],
    "facts": ["测试资料建议保持规律作息，并关注安静舒适的休息环境。"],
    "source": "离线测试资料",
    "url": "https://example.invalid/test-evidence",
}
DRAFT = (
    "关于睡眠，建议保持规律作息，并关注安静舒适的休息环境。"
    "生活中可以记录自己的作息习惯，结合实际感受了解休息情况，"
    "并在需要时咨询专业人员。以上内容依据所提供的测试资料整理。[1]"
)
EXPECTED_CLAUSES = {
    "关于睡眠，建议保持规律作息，并关注安静舒适的休息环境",
    "生活中可以记录自己的作息习惯，结合实际感受了解休息情况，并在需要时咨询专业人员",
}


def verified_clauses(mock):
    return [call.kwargs["messages"][-1]["content"].split("待核查主张：", 1)[1]
            for call in mock.call_args_list]


def model_response(content):
    return SimpleNamespace(message=SimpleNamespace(content=content), done=True, done_reason="stop")


def writer_response(content):
    return model_response(json.dumps({"article": content}, ensure_ascii=False))


@pytest.fixture
def mock_models(monkeypatch):
    monkeypatch.setattr(agent, "knowledge_base", [DOCUMENT])
    monkeypatch.delenv("INJECT_BAD_FACT", raising=False)
    monkeypatch.delenv("INJECT_FALSE_CLAIM", raising=False)
    writer = Mock(return_value=writer_response(DRAFT))
    verifier = Mock(return_value=model_response(json.dumps({
        "verdict": "supported",
        "reason": "离线流程测试的固定分类",
        "evidence_id": 1,
    })))
    monkeypatch.setattr(agent, "chat", writer)
    monkeypatch.setattr(claim_verifier, "chat", verifier)
    return writer, verifier


@pytest.mark.parametrize("legacy_name, modular_name", [
    ("api", "app.api"),
    ("health_content_agent_v3", "app.agent"),
    ("claim_verifier", "app.claim_verifier"),
    ("evaluate_claims", "app.evaluation"),
])
def test_legacy_imports_share_implementation(legacy_name, modular_name):
    assert importlib.import_module(legacy_name) is importlib.import_module(modular_name)


def test_api_preserves_complete_payload_and_legacy_patch():
    result = {
        "status": "completed",
        "answer": "离线测试回答。[1]",
        "evidence_count": 1,
        "revision_count": 1,
        "duration_seconds": 12.5,
    }
    with patch("api.run_health_agent", return_value=result) as run_agent:
        response = TestClient(api.app).post(
            "/generate", json={"task": "请写一篇关于睡眠的科普文章"}
        )
    assert response.status_code == 200
    assert response.json() == result
    run_agent.assert_called_once_with("请写一篇关于睡眠的科普文章")


def test_api_hides_internal_errors():
    with patch("api.run_health_agent", side_effect=RuntimeError("private error detail")):
        response = TestClient(api.app).post(
            "/generate", json={"task": "请写一篇关于睡眠的科普文章"}
        )
    assert response.status_code == 503
    assert response.json() == {"detail": "Agent执行失败，请检查服务日志及本地模型"}
    assert "private error detail" not in response.text


def test_graph_without_evidence_skips_models(mock_models):
    writer, verifier = mock_models
    result = agent.run_health_agent("没有匹配关键词的主题")
    assert result["status"] == "no_evidence"
    assert result["evidence_count"] == result["revision_count"] == 0
    assert result["answer"] == "知识库中没有找到相关资料，暂不生成内容。"
    writer.assert_not_called()
    verifier.assert_not_called()


def test_graph_success_uses_real_verifier_with_mock_model(mock_models):
    writer, verifier = mock_models
    result = agent.run_health_agent("请写一篇睡眠科普文章")
    assert result["status"] == "completed"
    assert result["evidence_count"] == 1
    assert result["revision_count"] == 0
    assert result["answer"].startswith(DRAFT)
    assert "[1] 离线测试资料：https://example.invalid/test-evidence" in result["answer"]
    assert "不代替专业医疗建议" in result["answer"]
    assert result["duration_seconds"] >= 0
    writer.assert_called_once()
    # Every substantive clause is covered, including those without old keywords.
    assert set(verified_clauses(verifier)) == EXPECTED_CLAUSES


def test_graph_revises_after_checker_feedback(mock_models):
    writer, verifier = mock_models
    writer.side_effect = [writer_response("内容太短。[1]"), writer_response(DRAFT)]
    result = agent.run_health_agent("请写一篇睡眠科普文章")
    assert result["status"] == "completed"
    assert result["revision_count"] == 1
    assert writer.call_count == 2
    assert "占位文字或未完成陈述" in writer.call_args.kwargs["messages"][1]["content"]
    assert set(verified_clauses(verifier)) == EXPECTED_CLAUSES | {"内容太短"}


def test_graph_stops_after_revision_limit(mock_models):
    writer, verifier = mock_models
    verifier.return_value = model_response(json.dumps({
        "verdict": "not_supported", "reason": "固定测试反馈", "evidence_id": 0,
    }))
    result = agent.run_health_agent("请写一篇睡眠科普文章")
    assert result["status"] == "evaluation_failed"
    assert result["revision_count"] == agent.MAX_REVISIONS
    assert writer.call_count == agent.MAX_REVISIONS + 1 == 2
    assert set(verified_clauses(verifier)) == EXPECTED_CLAUSES
    assert verifier.call_count == len(EXPECTED_CLAUSES) * 2
    assert "不输出未经核查的科普内容" in result["answer"]
    assert DRAFT not in result["answer"]


def test_imports_and_knowledge_loading_from_another_directory(tmp_path):
    # A subprocess verifies fresh imports, without depending on sys.modules caches.
    script = """
import json
import socket

def reject_connection(*args, **kwargs):
    raise AssertionError('Imports must not connect to Ollama')

socket.socket.connect = reject_connection
socket.socket.connect_ex = reject_connection
import api
import app.api
import health_content_agent_v3
import claim_verifier
import evaluate_claims
from app.knowledge import KNOWLEDGE_PATH, load_knowledge_base

assert api is app.api
assert callable(health_content_agent_v3.main)
assert callable(claim_verifier.main)
assert callable(evaluate_claims.main)
print(json.dumps({'path': str(KNOWLEDGE_PATH), 'documents': load_knowledge_base()}))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    completed = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, env=env,
        capture_output=True, text=True, check=True, timeout=30,
    )
    loaded = json.loads(completed.stdout)
    expected_path = PROJECT_ROOT / "data" / "health_knowledge.json"
    assert Path(loaded["path"]) == expected_path
    assert loaded["documents"] == json.loads(expected_path.read_text(encoding="utf-8"))
    assert not list(tmp_path.iterdir())  # Importing evaluation must not write a report.
