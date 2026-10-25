"""Offline Writer protocol regressions; no response is an independent medical review."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app import writer


BODY = ("18至60岁成年人通常建议每晚睡眠至少7小时。[1]"
        "不同年龄人群的推荐睡眠时长不同。健康睡眠不仅需要足够的睡眠时长，也需要良好的睡眠质量。[1]")
EXPECTED_OPTIONS = {"temperature": 0, "seed": 42, "num_predict": 512, "num_ctx": 4096}


def state(revision=0):
    return {"task": "请依据睡眠资料写科普。", "evidence": [{"facts": [BODY]}],
            "draft": BODY, "issues": ["删除无依据的治疗效果"], "revision_count": revision}


def article_json(article=BODY):
    return json.dumps({"article": article}, ensure_ascii=False)


def reply(content=None, **kwargs):
    return SimpleNamespace(message=SimpleNamespace(content=article_json() if content is None else content),
                           **{"done": True, "done_reason": "stop", **kwargs})


def assert_safe_failure(result):
    assert result["status"] == "draft_generation_failed"
    assert result["generation_failed"] is True
    assert result["draft"] == ""
    assert result["answer"] == writer.SAFE_STOP
    assert BODY not in result["answer"]
    assert result["writer_runs"][-1]["error"]


def test_short_json_prompt_uses_same_fixed_budget_for_initial_and_repair():
    first, repair = (writer.build_messages(state(index)) for index in (0, 1))
    assert [message["role"] for message in first] == ["system", "user"]
    assert [message["role"] for message in repair] == ["system", "user"]
    assert all("<think>" not in message["content"] and "</think>" not in message["content"]
               for messages in (first, repair) for message in messages)
    assert "/no_think" in first[1]["content"] and "/no_think" in repair[1]["content"]
    assert "上一版文章" not in first[1]["content"]
    assert "发现的问题" in repair[1]["content"]
    assert "删除无依据的治疗效果" in repair[1]["content"]
    assert len(repair[1]["content"]) < 1000
    assert "120至220字" not in first[0]["content"]
    assert "200至300字" not in first[0]["content"]

    requests = []
    for revision in (0, 1):
        model = Mock(return_value=reply(eval_count=86, prompt_eval_count=210))
        result = writer.generate_article(state(revision), model)
        model.assert_called_once()
        requests.append(model.call_args.kwargs)
        assert result["draft"] == BODY
        # Generation alone has not yet passed the Evidence Checker.
        assert result["status"] == "drafted" and result["status"] != "completed"
        assert not result["generation_failed"]
    for request in requests:
        assert request["options"] == EXPECTED_OPTIONS
        assert request["think"] is False
        assert request["model"] == writer.MODEL
        schema = request["format"]
        assert schema["type"] == "object"
        assert schema["required"] == ["article"]
        assert schema["properties"]["article"]["type"] == "string"
        assert schema["additionalProperties"] is False


def test_json_article_and_original_raw_response_and_token_metrics_are_preserved():
    raw = json.dumps({"article": BODY}, ensure_ascii=False, indent=2) + "\n"
    model = Mock(return_value=reply(raw, eval_count=91, prompt_eval_count=183))
    result = writer.generate_article(state(), model)
    assert result["draft"] == BODY
    metadata = result["writer_runs"][0]
    assert metadata["raw_content"] == raw
    assert metadata["eval_count"] == 91
    assert metadata["prompt_eval_count"] == 183
    assert metadata["done"] is True and metadata["done_reason"] == "stop"
    assert metadata["response_received"] is True
    assert metadata["truncated"] is False and metadata["error"] is None
    assert metadata["protocol"] == "json_article_v1"
    assert metadata["assistant_prefill"] is None
    assert metadata["elapsed_seconds"] >= 0


@pytest.mark.parametrize("done,reason", [
    (False, "stop"), (None, "stop"), (0, "stop"), (1, "stop"), ("true", "stop"),
    (True, None), (True, "load"), (True, "error"), (True, "unknown"),
])
def test_only_explicit_successful_finish_metadata_accepts_a_valid_json_body(done, reason):
    model = Mock(return_value=reply(done=done, done_reason=reason))
    result = writer.generate_article(state(), model)
    model.assert_called_once()
    assert_safe_failure(result)


def test_missing_finish_metadata_is_not_success():
    response = SimpleNamespace(message=SimpleNamespace(content=article_json()))
    result = writer.generate_article(state(), Mock(return_value=response))
    assert_safe_failure(result)
    metadata = result["writer_runs"][0]
    assert metadata["done"] is None and metadata["done_reason"] is None


def test_length_finish_rejects_even_syntactically_complete_json_and_keeps_original_tokens():
    raw = article_json()
    model = Mock(return_value=reply(raw, done_reason="length", eval_count=512, prompt_eval_count=190))
    result = writer.generate_article(state(), model)
    model.assert_called_once()
    assert_safe_failure(result)
    metadata = result["writer_runs"][0]
    assert metadata["truncated"] is True
    assert metadata["raw_content"] == raw
    assert metadata["eval_count"] == 512 and metadata["prompt_eval_count"] == 190


@pytest.mark.parametrize("raw", [
    "", " ", BODY, '{"article":', '```json\n{"article":"正文"}\n```',
    '[]', 'null', '{}', '{"article":null}', '{"article":42}', '{"article":[]}',
    json.dumps({"article": BODY, "analysis": "extra output"}, ensure_ascii=False),
    article_json() + " trailing explanation",
])
def test_malformed_json_or_non_schema_payload_is_not_salvaged(raw):
    result = writer.generate_article(state(), Mock(return_value=reply(raw)))
    assert_safe_failure(result)
    assert result["writer_runs"][0]["raw_content"] == raw


@pytest.mark.parametrize("article", ["", " \n\t", "字" * (writer.MAX_ARTICLE_CHARS + 1)])
def test_empty_or_oversized_article_is_rejected(article):
    result = writer.generate_article(state(), Mock(return_value=reply(article_json(article))))
    assert_safe_failure(result)


@pytest.mark.parametrize("article", [
    "<think>unfinished", "<think>internal</think>" + BODY, "</think>" + BODY,
    BODY + "<think>unfinished", BODY + "<think>internal</think>",
    "分析：先核对证据。" + BODY, "思考：先核对证据。" + BODY,
    "首先，用户要求我写文章。", "我需要生成一篇正文。", "我应该确保文章有依据。",
])
def test_thinking_tags_and_explicit_analysis_inside_article_are_rejected(article):
    raw = article_json(article)
    result = writer.generate_article(state(), Mock(return_value=reply(raw)))
    assert_safe_failure(result)
    metadata = result["writer_runs"][0]
    assert metadata["analysis_detected"] is True
    assert metadata["raw_content"] == raw


def test_timeout_is_one_attempt_and_keeps_unknown_token_metadata():
    model = Mock(side_effect=TimeoutError("fixed request deadline"))
    result = writer.generate_article(state(1), model)
    model.assert_called_once()
    assert_safe_failure(result)
    metadata = result["writer_runs"][0]
    assert metadata["phase"] == "repair"
    assert metadata["error"]["type"] == "TimeoutError"
    assert metadata["eval_count"] is None and metadata["prompt_eval_count"] is None
    assert metadata["done"] is None and metadata["done_reason"] is None
    assert metadata["response_received"] is False
    assert metadata["elapsed_seconds"] >= 0


def test_repair_appends_a_separate_record_without_rewriting_initial_observation():
    initial_model = Mock(return_value=reply(eval_count=81, prompt_eval_count=170))
    initial = writer.generate_article(state(), initial_model)
    initial_record = dict(initial["writer_runs"][0])
    repair_state = {**state(1), "writer_runs": initial["writer_runs"]}
    repair_model = Mock(return_value=reply(eval_count=89, prompt_eval_count=244))
    repaired = writer.generate_article(repair_state, repair_model)
    assert repaired["writer_runs"][0] == initial_record
    assert len(repaired["writer_runs"]) == 2
    assert repaired["writer_runs"][1]["phase"] == "repair"
    assert repaired["writer_runs"][1]["eval_count"] == 89
    assert initial_model.call_args.kwargs["options"] == repair_model.call_args.kwargs["options"] == EXPECTED_OPTIONS
    initial_model.assert_called_once()
    repair_model.assert_called_once()
