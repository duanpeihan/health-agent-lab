"""Offline flow tests for independent OFF/ON runs and the one-repair intervention.

Fixed model responses exercise safety gates and accounting, not medical quality.
The parent evaluations conftest forbids every network connection.
"""

import copy
import json
import os
import subprocess
from types import SimpleNamespace

import pytest

from app import agent, claim_verifier, writer
from evaluations.reflection_ab import core


GOOD_DRAFT = (
    "这篇睡眠说明依据当前给定资料整理，关注资料允许表达的范围，也保留年龄范围和不同条件。"
    "18至60岁成年人通常建议每晚睡眠至少7小时。"
    "健康睡眠不仅需要足够的睡眠时长，也需要良好的睡眠质量。"
    "以上内容仅限所提供资料，未引入其他干预效果。[1]"
)
BAD_CLAIM = "睡前喝牛奶能够治愈失眠。[1]"


def response(content, **extra):
    return SimpleNamespace(message=SimpleNamespace(content=content),
                           **{"done": True, "done_reason": "stop", **extra})


def request_stage(request):
    properties = request.get("format", {}).get("properties", {})
    return "Writer" if "article" in properties else "Verifier"


def task_case(case_id="TEST", *, task="请写成年人睡眠科普。", fault=None):
    return {"id": case_id, "task": task,
            "fault": fault or {"mode": "none", "text": ""}}


class FixedModel:
    def __init__(self, drafts=None, fail_stage=None):
        self.drafts = list(drafts or [GOOD_DRAFT])
        self.fail_stage = fail_stage
        self.calls = []
        self.writer_calls = 0

    def __call__(self, *args, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        stage = request_stage(kwargs)
        if self.fail_stage == stage:
            raise RuntimeError(f"Fixed {stage} failure")
        if stage == "Writer":
            index = min(self.writer_calls, len(self.drafts) - 1)
            self.writer_calls += 1
            return response(json.dumps({"article": self.drafts[index]}, ensure_ascii=False))
        claim = kwargs["messages"][-1]["content"].split("待核查主张：", 1)[1]
        verdict = "not_supported" if "牛奶" in claim else "supported"
        return response(json.dumps({"verdict": verdict,
                                    "reason": "离线流程测试固定分类，不代表医学判断",
                                    "evidence_id": 0 if verdict == "not_supported" else 1},
                                   ensure_ascii=False))


def assert_accounting(result, model):
    calls = result["model_calls"]
    ids = {item["id"] for item in calls}
    a, b = result["rows"]["A"], result["rows"]["B"]
    assert result["physical_model_call_count"] == len(calls) == len(ids) == len(model.calls)
    assert set(a["model_call_ids"]) | set(b["model_call_ids"]) == ids
    assert set(a["model_call_ids"]) & set(b["model_call_ids"]) == set()
    assert result["shared_model_call_ids"] == []
    assert a["model_call_count"] + b["model_call_count"] == len(ids)
    for row in (a, b):
        assert row["model_call_count"] == len(row["model_call_ids"])
        assert row["shared_model_call_count"] == 0
        assert row["model_call_count"] == row["exclusive_model_call_count"]
        assert set(row["stage_seconds"]) == {"Retriever", "Writer", "Verifier", "Reflection"}
        assert all(seconds >= 0 for seconds in row["stage_seconds"].values())
        assert row["total_seconds"] >= 0
        assert row["quality_status"] == "pending_independent_human_review"
    for observed, request in zip(calls, model.calls):
        assert request["model"] == "qwen3:4b"
        assert request["think"] is False
        expected_options = writer.OPTIONS if request_stage(request) == "Writer" else {"temperature": 0}
        assert request["options"] == expected_options == observed["options"]
        assert observed["elapsed_seconds"] >= 0


def test_suite_is_exactly_four_representative_tasks_with_only_one_fault():
    suite = core.load_cases()
    assert len(suite["cases"]) == len({case["id"] for case in suite["cases"]}) == 4
    assert {case["category"] for case in suite["cases"]} == {
        "normal_sleep", "no_evidence", "unsupported_claim_injection", "insufficient_evidence"}
    injected = [case for case in suite["cases"] if case["fault"]["mode"] != "none"]
    assert len(injected) == 1
    assert injected[0]["fault"] == {"mode": "append", "text": BAD_CLAIM}


def test_no_evidence_stops_both_branches_without_any_model_call():
    model = FixedModel()
    result = core.run_pair(task_case(task="请写科学补钙科普。"), model)
    assert result["execution_mode"] == "test_double"
    assert result["design"] == "independent_end_to_end_paired"
    assert result["physical_model_call_count"] == 0
    assert result["shared_model_call_ids"] == []
    for row in result["rows"].values():
        assert row["status"] == "no_evidence"
        assert row["no_evidence_safe_stop"] is True
        assert row["revision_count"] == row["model_call_count"] == 0
        assert row["draft"] == ""
        assert row["check_history"] == []
        assert row["stop_reason"] == "no_evidence"
        assert row["answer"]
    assert_accounting(result, model)


def test_clean_initial_draft_passes_checker_without_reflection_in_either_branch():
    model = FixedModel()
    result = core.run_pair(task_case(), model)
    assert result["raw_initial_draft"] == GOOD_DRAFT
    for row in result["rows"].values():
        assert row["status"] == "completed"
        assert row["revision_count"] == 0
        assert row["final_errors"] == row["detected_errors"] == []
        assert row["stop_reason"] == "checker_passed"
        assert row["no_evidence_safe_stop"] is None
        assert len(row["check_history"]) == 1
        assert row["check_history"][0]["claims"]
        assert all(item["method"] == "exact_fact" for item in row["check_history"][0]["claims"])
        assert row["check_history"][0]["unchecked_candidate_claims"] == []
        assert row["final_checker_completed"] is True
    assert model.writer_calls == 2
    assert result["physical_model_call_count"] == 2
    assert_accounting(result, model)


def test_independent_initial_runs_use_identical_production_requests_without_shared_calls():
    model = FixedModel()
    result = core.run_pair(task_case(fault={"mode": "append", "text": BAD_CLAIM}), model)
    initial = [call for call in result["model_calls"] if call["stage"] == "Writer" and call["pass"] == 0]
    assert len(initial) == 2
    assert {call["phase"] for call in initial} == {"A", "B"}
    assert initial[0]["request"] == initial[1]["request"]
    assert initial[0]["request"]["options"] == writer.OPTIONS
    assert initial[0]["request"]["format"] == writer.OUTPUT_SCHEMA
    verifier_calls = [call for call in result["model_calls"] if call["stage"] == "Verifier"]
    assert len(verifier_calls) == 2
    assert all(call["request"]["options"] == {"temperature": 0} for call in verifier_calls)
    assert_accounting(result, model)


def test_off_failure_does_not_replace_the_on_independent_initial_draft():
    model = FixedModel([BAD_CLAIM, GOOD_DRAFT])
    result = core.run_pair(task_case(), model)
    a, b = result["rows"]["A"], result["rows"]["B"]
    assert a["initial_draft"] == BAD_CLAIM
    assert b["initial_draft"] == GOOD_DRAFT
    assert a["status"] == "evaluation_failed" and b["status"] == "completed"
    assert a["revision_count"] == b["revision_count"] == 0
    assert a["initial_draft_sha256"] != b["initial_draft_sha256"]
    assert_accounting(result, model)


def test_task_parameter_gate_is_kept_for_off_and_on_evidence_limit_answers():
    case = next(case for case in core.load_cases()["cases"] if case["id"] == "R-INSUFFICIENT")
    bad = "无法确认睡前拉伸2.0分钟能否使所有人的入睡时间减半。[1]"
    good = "无法确认睡前拉伸20分钟能否使所有人的入睡时间减半。[1]"
    model = FixedModel([bad, bad, good])
    result = core.run_pair(case, model)
    a, b = result["rows"]["A"], result["rows"]["B"]
    assert a["status"] == "evaluation_failed" and a["revision_count"] == 0
    assert b["status"] == "completed" and b["revision_count"] == 1
    assert a["check_history"][-1]["checker_details"]["task_parameter_fidelity"]["passed"] is False
    assert b["check_history"][-1]["checker_details"]["task_parameter_fidelity"]["passed"] is True
    assert all(check["checker_details"]["segments"][0]["kind"] == "evidence_limit"
               for row in (a, b) for check in row["check_history"])
    assert "2.0分钟" not in a["answer"] and "2.0分钟" not in b["answer"]
    assert all(call["stage"] == "Writer" for call in result["model_calls"])
    assert_accounting(result, model)


def test_content_capacity_terminal_state_is_preserved_without_writer_in_either_group():
    model = FixedModel()
    result = core.run_pair(task_case(task="请依据睡眠资料写一篇2000字长文。"), model)
    assert result["physical_model_call_count"] == model.writer_calls == 0
    for row in result["rows"].values():
        assert row["status"] == "insufficient_content"
        assert row["revision_count"] == 0 and row["check_history"] == []
        assert row["draft"] == "" and row["evidence"]
        assert row["stop_reason"] == "insufficient_content"
        assert "不足" in row["answer"]
    assert_accounting(result, model)


def test_off_still_checks_and_blocks_while_on_repairs_once_and_rechecks():
    model = FixedModel([GOOD_DRAFT, GOOD_DRAFT])
    case = task_case(fault={"mode": "append", "text": BAD_CLAIM})
    events = []
    result = core.run_pair(case, model, event_sink=events.append)
    a, b = result["rows"]["A"], result["rows"]["B"]
    assert a["initial_draft"] == b["initial_draft"] == GOOD_DRAFT + "\n" + BAD_CLAIM
    assert a["initial_draft_sha256"] == b["initial_draft_sha256"]
    assert a["input_evidence_sha256"] == b["input_evidence_sha256"]
    # Equal fixed inputs yield equal decisions, but both checks run independently.
    assert a["check_history"][0]["checker_details"] == b["check_history"][0]["checker_details"]
    assert any("牛奶" in item["claim"] for item in a["check_history"][0]["claims"])
    assert a["status"] == "evaluation_failed"
    assert a["revision_count"] == 0
    assert a["exclusive_model_call_count"] > 0
    assert a["stop_reason"] == "reflection_disabled"
    assert a["final_errors"] and a["detected_errors"]
    assert BAD_CLAIM not in a["answer"]
    assert b["status"] == "completed"
    assert b["revision_count"] == 1
    assert b["final_errors"] == [] and b["detected_errors"]
    assert len(b["check_history"]) == 2
    assert "牛奶" not in b["draft"]
    repair_prompts = [message["content"] for call in model.calls if request_stage(call) == "Writer"
                      for message in call["messages"] if message["role"] == "user"
                      and "上一版文章" in message["content"]]
    assert len(repair_prompts) == 1 and "发现的问题" in repair_prompts[0]
    assert all(call["messages"][-1]["role"] == "user"
               for call in model.calls if request_stage(call) == "Writer")
    assert all(call["format"]["properties"]["article"]["type"] == "string"
               for call in model.calls if request_stage(call) == "Writer")
    assert model.writer_calls == 3
    assert sum(stage["stage"] == "Reflection" for stage in result["stages"]) == 1
    assert any(event["event"] == "model_call_started" for event in events)
    assert sum(event["event"] == "model_call_finished" for event in events) == len(model.calls)
    assert events[-1]["event"] == "pair_finished"
    assert_accounting(result, model)


def test_failed_repair_remains_blocked_without_a_second_repair():
    model = FixedModel([GOOD_DRAFT, GOOD_DRAFT, GOOD_DRAFT + "\n" + BAD_CLAIM])
    result = core.run_pair(task_case(fault={"mode": "append", "text": BAD_CLAIM}), model)
    b = result["rows"]["B"]
    assert model.writer_calls == 3
    assert b["revision_count"] == 1
    assert b["status"] == "evaluation_failed"
    assert b["stop_reason"] == "revision_limit_1"
    assert b["final_errors"]
    assert len(b["check_history"]) == 2
    assert BAD_CLAIM not in b["answer"]
    assert_accounting(result, model)


def test_ambient_fault_environment_is_ignored_and_all_scoped_patches_restore(monkeypatch):
    monkeypatch.setenv("INJECT_BAD_FACT", "1")
    monkeypatch.setenv("INJECT_FALSE_CLAIM", "1")
    initial_objects = (agent.chat, claim_verifier.chat, agent.verify_claim, agent.knowledge_base,
                       agent.MAX_REVISIONS)
    model = FixedModel()
    result = core.run_pair(task_case(), model)
    assert result["raw_initial_draft"] == GOOD_DRAFT
    assert result["rows"]["A"]["initial_draft"] == GOOD_DRAFT
    assert "12小时" not in result["rows"]["A"]["initial_draft"]
    assert "牛奶" not in result["rows"]["A"]["initial_draft"]
    assert os.environ["INJECT_BAD_FACT"] == os.environ["INJECT_FALSE_CLAIM"] == "1"
    after_objects = (agent.chat, claim_verifier.chat, agent.verify_claim, agent.knowledge_base,
                     agent.MAX_REVISIONS)
    assert all(before is after for before, after in zip(initial_objects, after_objects))
    assert_accounting(result, model)


@pytest.mark.parametrize("stage", ["Writer", "Verifier"])
def test_model_exception_fails_closed_with_attempt_and_elapsed_accounting(stage):
    model = FixedModel(fail_stage=stage)
    result = core.run_pair(task_case(fault={"mode": "append", "text": BAD_CLAIM}), model)
    for row in result["rows"].values():
        assert row["status"] == "evaluation_failed"
        assert row["stop_reason"] == ("writer_generation_failed" if stage == "Writer"
                                      else "infrastructure_error")
        assert row["error"]["type"] == "RuntimeError"
        assert row["answer"] == core.SAFE_STOP
        assert row["revision_count"] == 0
        assert row["final_checker_completed"] is False
        if stage == "Writer":
            assert row["check_history"] == []
            assert row["draft"] == ""
    assert model.writer_calls <= 2
    failed = result["model_calls"][-1]
    assert failed["completed"] is False
    assert failed["error"]["type"] == "RuntimeError"
    assert failed["elapsed_seconds"] >= 0
    assert_accounting(result, model)


def test_truncated_writer_response_is_not_published_as_complete():
    class TruncatedModel(FixedModel):
        def __call__(self, *args, **kwargs):
            result = super().__call__(*args, **kwargs)
            result.done_reason = "length"
            return result

    model = TruncatedModel()
    result = core.run_pair(task_case(), model)
    for row in result["rows"].values():
        assert row["status"] == "evaluation_failed"
        assert row["answer"] == core.SAFE_STOP
        assert row["revision_count"] == 0
        assert row["check_history"] == []
        assert row["writer_runs"][0]["truncated"] is True
    assert result["model_calls"][0]["truncated"] is True
    assert result["model_calls"][0]["completed"] is False
    assert "truncated" in result["model_calls"][0]["error"]["message"]
    assert_accounting(result, model)


def test_repair_writer_timeout_does_not_recheck_the_old_draft_or_claim_success():
    class RepairTimeout(FixedModel):
        def __call__(self, *args, **kwargs):
            if request_stage(kwargs) == "Writer" and any(
                    "上一版文章" in message["content"] for message in kwargs["messages"]):
                self.calls.append(copy.deepcopy(kwargs))
                self.writer_calls += 1
                raise TimeoutError("offline repair timeout")
            return super().__call__(*args, **kwargs)

    model = RepairTimeout()
    result = core.run_pair(task_case(fault={"mode": "append", "text": BAD_CLAIM}), model)
    b = result["rows"]["B"]
    assert b["status"] == "evaluation_failed" and b["answer"] == core.SAFE_STOP
    assert b["revision_count"] == 1 and model.writer_calls == 3
    assert len(b["check_history"]) == 1
    assert b["final_checker_completed"] is False
    assert b["draft"] == ""
    assert b["writer_runs"][-1]["error"]["type"] == "TimeoutError"
    assert not any(stage["stage"] == "Verifier" and stage["phase"] == "B" and stage["pass"] == 1
                   for stage in result["stages"])
    assert_accounting(result, model)


@pytest.mark.parametrize("stage", ["Writer", "Verifier"])
@pytest.mark.parametrize("missing", ["done", "done_reason", "both"])
def test_missing_finish_metadata_cannot_be_marked_completed(stage, missing):
    class IncompleteResponse(FixedModel):
        def __call__(self, *args, **kwargs):
            result = super().__call__(*args, **kwargs)
            current_stage = request_stage(kwargs)
            if current_stage == stage:
                for field in ("done", "done_reason"):
                    if missing in (field, "both"):
                        delattr(result, field)
            return result

    model = IncompleteResponse()
    result = core.run_pair(task_case(fault={"mode": "append", "text": BAD_CLAIM}), model)
    failed = result["model_calls"][-1]
    assert failed["stage"] == stage
    assert failed["completed"] is False and failed["error"]
    for row in result["rows"].values():
        assert row["status"] != "completed"
        assert row["answer"] == core.SAFE_STOP
        assert row["final_checker_completed"] is False
        assert row["revision_count"] == 0
    assert_accounting(result, model)


def test_checker_six_claim_limit_records_unchecked_candidates_and_blocks_pass():
    claims = [f"测试医学建议{index}需要充分证据" for index in range(7)]
    draft = "。".join(claims) + "。[1]"
    model = FixedModel([draft])
    result = core.run_pair(task_case(), model)
    check = result["rows"]["A"]["check_history"][0]
    assert check["completed"] is False
    assert len(check["candidate_claims"]) == 7
    assert len(check["claims"]) == 6
    assert check["unchecked_candidate_claims"] == [claims[-1]]
    assert check["checker_details"]["complete"] is False
    assert result["rows"]["A"]["status"] == "evaluation_failed"
    assert result["rows"]["A"]["final_checker_completed"] is False
    assert result["rows"]["B"]["status"] == "evaluation_failed"
    assert result["rows"]["B"]["stop_reason"] == "checker_incomplete"
    assert result["rows"]["B"]["revision_count"] == 0
    assert model.writer_calls == 2
    assert result["rows"]["A"]["quality_status"] == "pending_independent_human_review"
    assert_accounting(result, model)


def test_interrupted_verifier_distinguishes_failed_and_not_yet_checked_claims():
    draft = (
        "建议所有成年人关注足够睡眠时长。建议所有成年人关注良好睡眠质量。"
        "建议所有成年人保留健康生活方式。建议所有成年人限制睡前活动。[1]"
    )
    model = FixedModel([draft], fail_stage="Verifier")
    result = core.run_pair(task_case(), model)
    check = result["rows"]["A"]["check_history"][0]
    assert check["completed"] is False
    assert check["failed_candidate_claims"] == ["建议所有成年人关注足够睡眠时长"]
    assert check["unchecked_candidate_claims"] == ["建议所有成年人关注良好睡眠质量", "建议所有成年人保留健康生活方式", "建议所有成年人限制睡前活动"]
    assert len(check["claims"]) == 1
    assert result["rows"]["A"]["status"] == "evaluation_failed"
    assert_accounting(result, model)


def test_insufficient_evidence_task_can_safely_explain_the_missing_evidence():
    draft = (
        "当前资料介绍了成年人睡眠时长、年龄范围与睡眠质量之间的表达条件。[1]"
        "所提供材料并未涉及拉伸干预，也没有入睡时间变化的数据，无法据此确认所询问的效果。"
        "此处仅说明证据范围，不补充任何外部结论，也不作个人诊断或治疗判断。"
    )
    case = next(item for item in core.load_cases()["cases"] if item["category"] == "insufficient_evidence")
    model = FixedModel([draft])
    result = core.run_pair(case, model)
    for row in result["rows"].values():
        assert row["status"] == "completed"
        assert row["revision_count"] == 0
        assert row["final_errors"] == []
        assert "无法据此确认" in row["answer"]
        assert row["quality_status"] == "pending_independent_human_review"
    assert_accounting(result, model)


def test_duplicate_task_ids_are_rejected_before_any_model_invocation(tmp_path):
    suite = core.load_cases()
    suite["cases"][1]["id"] = suite["cases"][0]["id"]
    target = tmp_path / "bad-suite.json"
    target.write_text(json.dumps(suite, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="four unique"):
        core.load_cases(target)


def test_runner_labels_injected_invocations_as_test_double_and_leaves_all_quality_scores_blank(tmp_path):
    from evaluations.reflection_ab import run

    def fake_invoke(case, strategy, event_path, request_timeout_seconds, timeout_seconds):
        return core.run_case(case, strategy, FixedModel())

    report = run.run_suite(core.load_cases()["cases"], invoke=fake_invoke, event_dir=tmp_path / "events")
    assert report["execution_mode"] == "test_double"
    assert report["case_count"] == len(report["pairs"]) == 4
    assert report["branch_count"] == 8
    assert all(pair["execution_mode"] == "test_double" for pair in report["pairs"])
    review = run.build_human_review(report)
    assert review["row_count"] == len(review["rows"]) == 8
    assert review["human_review_completed"] is False
    for item in review["rows"]:
        assert item["medical_quality"] is None
        assert item["task_fulfillment"] is None
        assert item["evidence_faithfulness"] is None
        assert item["reviewer"] is None and item["reviewed_on"] is None
        assert item["review_status"] == "pending_independent_human_review"
    assert "`test_double`" in run.render_report(report)
    assert "尚未独立人工审核" in run.render_human_review(review)


def test_timeout_journal_preserves_attempt_lower_bound_and_unknown_final_results(tmp_path):
    from evaluations.reflection_ab import run

    def timed_out_invoke(case, strategy, event_path, request_timeout_seconds, timeout_seconds):
        call = {"id": f"{case['id']}/{strategy}/1", "phase": strategy, "stage": "Writer",
                "pass": 0, "model": core.MODEL, "think": False, "options": core.OPTIONS,
                "completed": False, "elapsed_seconds": None}
        event_path.write_text(json.dumps({"event": "model_call_started", "call": call})
                              + '\n{"event":', encoding="utf-8")
        raise subprocess.TimeoutExpired(["offline-timeout-fixture"], timeout_seconds)

    report = run.run_suite(core.load_cases()["cases"], invoke=timed_out_invoke,
                           event_dir=tmp_path / "events")
    assert report["execution_mode"] == "test_double"
    assert report["summary"]["physical_model_call_count_unknown_pairs"] == 4
    assert report["summary"]["observed_started_model_call_count_lower_bound"] == 8
    for pair in report["pairs"]:
        assert pair["physical_model_call_count"] is None
        assert pair["observed_started_model_call_count"] == 2
        assert sum(call.get("completed") is True for call in pair["model_calls"]) == 0
        assert pair["unknown_model_call_count"] is True
        assert sum(call.get("completed") is not True for call in pair["model_calls"]) == 2
        for row in pair["rows"].values():
            assert row["status"] == "unknown"
            assert row["model_call_count"] is None
            assert row["observed_started_model_call_count"] == 1
            assert row["revision_count"] is None
            assert row["final_errors"] is None
            assert row["no_evidence_safe_stop"] is None
            assert row["answer"] == core.SAFE_STOP
    # Recovery must still yield an unscored review queue, including unknown cases.
    assert run.build_human_review(report)["row_count"] == 8


def test_validate_only_does_not_start_preflight_or_worker(monkeypatch, capsys):
    from evaluations.reflection_ab import run

    def forbidden(*args, **kwargs):
        raise AssertionError("validate-only attempted model preparation or invocation")

    monkeypatch.setattr(run, "preflight_model", forbidden)
    monkeypatch.setattr(run, "run_suite", forbidden)
    assert run.main(["--validate-only"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["model_calls"] == payload["network_calls"] == 0
    assert payload["case_count"] == 4


def test_runner_rejects_fake_payload_that_claims_real_model_mode(tmp_path):
    from evaluations.reflection_ab import run

    def mislabeled_invoke(case, strategy, event_path, request_timeout_seconds, timeout_seconds):
        return core.run_case(case, strategy, FixedModel(), execution_mode="real_model")

    report = run.run_suite(core.load_cases()["cases"], invoke=mislabeled_invoke,
                           event_dir=tmp_path / "events")
    assert report["execution_mode"] == "test_double"
    assert len(report["rows"]) == 8
    for row in report["rows"]:
        assert row["model_call_count"] is None
        assert "test double" in row["error"]["message"]
        assert row["status"] == "unknown"
