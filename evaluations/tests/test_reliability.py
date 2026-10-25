"""Offline regressions for the observer, bounded execution and truthful reports."""

import copy
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
from ollama import ChatResponse

from evaluations.reliability import run


@pytest.fixture
def cases():
    return run.load_cases()["cases"]


@pytest.fixture
def production_nodes(monkeypatch):
    """Controlled node boundaries; app Writer/Checker behavior has separate regression tests."""
    from app import agent, claim_verifier
    captured = {"writer_inputs": [], "checker_inputs": []}

    def retriever(state):
        return {"evidence": [{"source": "fixture", "facts": ["证据事实"], "id": 1}], "status": "retrieved"}

    def writer(state):
        captured["writer_inputs"].append(copy.deepcopy(state))
        phase = "repair" if state["revision_count"] else "initial"
        metadata = {"phase": phase, "truncated": False, "error": None}
        try:
            response = agent.chat(model=run.MODEL, think=False,
                                  messages=[{"role": "user", "content": "fixture writer"}],
                                  options={"temperature": 0, "seed": 42, "num_predict": 512, "num_ctx": 4096})
            metadata.update({"truncated": response.done_reason == "length",
                             "eval_count": response.eval_count, "done_reason": response.done_reason,
                             "done": response.done})
            failed = metadata["truncated"]
            draft = response.message.content
        except Exception as exc:
            failed, draft = True, ""
            metadata["error"] = {"type": type(exc).__name__, "message": str(exc)}
        return {"status": "draft_generation_failed" if failed else "drafted", "draft": draft,
                "generation_failed": failed, "writer_runs": state.get("writer_runs", []) + [metadata]}

    def checker(state):
        captured["checker_inputs"].append(copy.deepcopy(state))
        injected = "治愈失眠" in state["draft"]
        if injected:
            claim_verifier.chat(model=run.MODEL, think=False,
                messages=[{"role": "user", "content": "fixture verifier"}], options={"temperature": 0})
        return {"status": "needs_revision" if injected else "completed",
                "answer": "" if injected else state["draft"],
                "issues": ["无依据治疗宣称"] if injected else [],
                "checker_details": {"complete": True, "claims": [{"method": "claim_verifier" if injected else "exact_fact"}],
                                    "segments": [], "unchecked_claims": [], "issue_codes": []}}

    def reflection(state):
        return {"revision_count": state["revision_count"] + 1, "status": "revising"}

    for name, function in (("retriever_node", retriever), ("writer_node", writer),
                           ("evidence_checker_node", checker), ("reflection_node", reflection)):
        monkeypatch.setattr(agent, name, function)
    return captured


def responses(*items):
    pending = list(items or ("证据事实。[1]",))
    def chat(**kwargs):
        value = pending.pop(0)
        if isinstance(value, Exception):
            raise value
        if isinstance(value, tuple):
            content, reason = value
        else:
            content, reason = value, "stop"
        return ChatResponse(model=run.MODEL, message={"role": "assistant", "content": content},
                            done=True, done_reason=reason, eval_count=15, prompt_eval_count=42)
    return chat


def test_frozen_three_tasks_match_original_and_order(cases):
    source = json.loads(run.CASES_PATH.read_text())
    assert [case["id"] for case in cases] == list(run.CASE_IDS)
    assert all(case == next(item for item in source["cases"] if item["id"] == case["id"]) for case in cases)


def test_validate_only_does_not_probe_or_invoke(monkeypatch, capsys):
    def reject(*args, **kwargs):
        raise AssertionError("Validation must not connect to a model")
    monkeypatch.setattr(run, "preflight_model", reject)
    monkeypatch.setattr(run, "invoke_worker", reject)
    assert run.main(["--validate-only"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["network_calls"] == output["model_calls"] == 0
    assert output["case_ids"] == list(run.CASE_IDS)


@pytest.mark.parametrize("task,request_timeout", [(241, 45), (180, 61), (float("nan"), 45), (180, 0), (True, 45)])
def test_rejects_unbounded_or_invalid_timeouts(task, request_timeout):
    with pytest.raises(ValueError):
        run.validate_timeouts(task, request_timeout)


def test_observer_preserves_writer_parameters_and_exact_fact_no_model_call(cases, production_nodes):
    result = run.run_case(cases[0], responses())
    assert result["status"] == "completed"
    assert result["execution_mode"] == "test_double"
    assert result["model_call_count"] == 1
    assert result["model_calls"][0]["options"]["num_predict"] == 512
    assert result["model_calls"][0]["eval_count"] == 15
    assert result["model_calls"][0]["prompt_eval_count"] == 42
    assert result["model_calls"][0]["completed"] is True
    assert result["checker_details"]["claims"][0]["method"] == "exact_fact"
    assert result["final_errors"] == [] and result["final_checker_completed"]
    assert result["stage_seconds"]["Reflection"] == 0


def test_completed_preserves_full_production_answer_even_when_different_from_draft(cases, production_nodes, monkeypatch):
    from app import agent
    answer = "生产完整答复前缀\n已核查的正文及原始尾注。"
    monkeypatch.setattr(agent, "evidence_checker_node", lambda state: {
        "status": "completed", "answer": answer, "issues": [], "checker_details": {"complete": True}})
    result = run.run_case(cases[0], responses("内部草稿"))
    assert result["status"] == "completed" and result["draft"] == "内部草稿"
    assert result["answer"] == result["production_answer"] == result["raw_final_state"]["answer"] == answer


def test_failed_check_uses_production_stop_answer(cases, production_nodes, monkeypatch):
    from app import agent
    stopped = []
    monkeypatch.setattr(agent, "evidence_checker_node", lambda state: {
        "status": "needs_revision", "issues": ["无依据事实"], "checker_details": {"complete": True}})
    def stop(state):
        stopped.append(state["revision_count"])
        return {"status": "evaluation_failed", "answer": "生产停止答复完整原文，不被观察器改写。"}
    monkeypatch.setattr(agent, "stop_node", stop)
    result = run.run_case(cases[0], responses("初稿", "修正稿"))
    assert result["status"] == "evaluation_failed" and stopped == [1]
    assert result["answer"] == result["production_answer"] == "生产停止答复完整原文，不被观察器改写。"
    assert result["failure_kind"] == "checker_issues_after_max_revision"


@pytest.mark.parametrize("done,reason", [(None, "stop"), (True, None), (False, "stop")])
def test_missing_or_incomplete_writer_finish_metadata_stops(cases, production_nodes, done, reason):
    def chat(**kwargs):
        return ChatResponse(model=run.MODEL, message={"role": "assistant", "content": "正文"},
                            done=done, done_reason=reason)
    result = run.run_case(cases[0], chat)
    assert result["status"] == "evaluation_failed" and result["revision_count"] == 0
    assert result["failure_kind"] == "writer_finish_metadata_invalid"
    assert not production_nodes["checker_inputs"]


@pytest.mark.parametrize("done,reason", [(None, "stop"), (True, None), (False, "stop")])
def test_missing_or_incomplete_verifier_finish_metadata_cannot_complete(cases, production_nodes, monkeypatch, done, reason):
    from app import agent, claim_verifier
    def checker(state):
        claim_verifier.chat(model=run.MODEL, think=False, options={"temperature": 0}, messages=[])
        return {"status": "completed", "answer": "保留生产核查节点返回原文", "issues": [],
                "checker_details": {"complete": True}}
    monkeypatch.setattr(agent, "evidence_checker_node", checker)
    def chat(**kwargs):
        if kwargs["messages"]:
            return responses("正文")(**kwargs)
        return ChatResponse(model=run.MODEL, message={"role": "assistant", "content": "{}"},
                            done=done, done_reason=reason)
    result = run.run_case(cases[0], chat)
    assert result["status"] == "error" and result["final_errors"] is None
    assert result["answer"] == result["production_answer"] == "保留生产核查节点返回原文"
    assert not result["final_checker_completed"]


def test_fake_cannot_label_itself_real_model(cases, production_nodes):
    with pytest.raises(ValueError, match="bound local Ollama"):
        run.run_case(cases[0], responses(), execution_mode="real_model")


def test_case_mutation_rejected_before_generation(cases, production_nodes):
    changed = copy.deepcopy(cases[0])
    changed["task"] += "改变"
    with pytest.raises(ValueError, match="unchanged"):
        run.run_case(changed, responses())


def test_injection_only_once_repair_rechecked_and_environment_restored(cases, production_nodes, monkeypatch):
    monkeypatch.setenv("INJECT_BAD_FACT", "1")
    monkeypatch.setenv("INJECT_FALSE_CLAIM", "1")
    observations = []
    def chat(**kwargs):
        observations.append((os.getenv("INJECT_BAD_FACT"), os.getenv("INJECT_FALSE_CLAIM")))
        return responses("证据事实。[1]")(**kwargs)
    result = run.run_case(cases[2], chat)
    assert result["status"] == "completed" and result["revision_count"] == 1
    assert result["model_call_count"] == 3
    assert len(production_nodes["checker_inputs"]) == 2
    assert result["initial_draft"].count(cases[2]["fault"]["text"]) == 1
    assert cases[2]["fault"]["text"] not in result["draft"]
    assert observations == [(None, None)] * 3
    assert os.getenv("INJECT_BAD_FACT") == os.getenv("INJECT_FALSE_CLAIM") == "1"
    assert [call["stage"] for call in result["model_calls"]] == ["Writer", "Verifier", "Writer"]


def test_initial_truncation_never_reaches_checker(cases, production_nodes):
    from app import agent
    result = run.run_case(cases[0], responses(("残缺正文", "length")))
    assert result["status"] == "evaluation_failed"
    assert result["answer"] == result["production_answer"] == agent.SAFE_STOP
    assert result["failure_kind"] == "draft_generation_failed"
    assert result["revision_count"] == 0 and result["model_call_count"] == 1
    assert not production_nodes["checker_inputs"]
    assert result["final_errors"] is None
    assert result["model_calls"][0]["truncated"] is True
    assert result["writer_metadata"][0]["truncated"] is True


def test_repair_timeout_keeps_initial_diagnostics_but_final_errors_unknown(cases, production_nodes):
    result = run.run_case(cases[2], responses("证据事实。[1]", "核查响应", TimeoutError("bounded request")))
    assert result["status"] == "evaluation_failed" and result["failure_kind"] == "draft_generation_failed"
    assert result["revision_count"] == 1 and result["model_call_count"] == 3
    assert result["detected_errors"] == ["无依据治疗宣称"]
    assert result["final_errors"] is None and not result["final_checker_completed"]
    assert len(production_nodes["checker_inputs"]) == 1
    assert result["model_calls"][-1]["response_received"] is False
    assert result["writer_metadata"][-1]["error"]["type"] == "TimeoutError"


def test_failed_recheck_does_not_retry(cases, production_nodes, monkeypatch):
    from app import agent
    calls = []
    def checker(state):
        calls.append(state["revision_count"])
        return {"status": "needs_revision", "issues": ["仍缺证据"],
                "checker_details": {"complete": True, "claims": [], "unchecked_claims": []}}
    monkeypatch.setattr(agent, "evidence_checker_node", checker)
    result = run.run_case(cases[0], responses("首稿", "修正稿"))
    assert result["status"] == "evaluation_failed" and result["answer"] == agent.SAFE_STOP
    assert calls == [0, 1] and result["revision_count"] == 1
    assert result["final_errors"] == ["仍缺证据"]


def test_incomplete_checker_cannot_return_completed(cases, production_nodes, monkeypatch):
    from app import agent
    monkeypatch.setattr(agent, "evidence_checker_node", lambda state: {
        "status": "completed", "issues": [], "checker_details": {"complete": False}})
    result = run.run_case(cases[0], responses())
    assert result["status"] == "error" and result["answer"] == run.SAFE_STOP
    assert result["final_errors"] is None and not result["final_checker_completed"]


def test_parseable_but_truncated_verifier_response_cannot_complete(cases, production_nodes, monkeypatch):
    from app import agent, claim_verifier
    def checker(state):
        # Simulate a verifier that accidentally accepts parseable JSON at the token limit.
        claim_verifier.chat(model=run.MODEL, think=False, options={"temperature": 0}, messages=[])
        return {"status": "completed", "issues": [], "checker_details": {"complete": True}}
    monkeypatch.setattr(agent, "evidence_checker_node", checker)
    result = run.run_case(cases[0], responses("正文", ('{"verdict":"supported"}', "length")))
    assert result["status"] == "error" and result["answer"] == run.SAFE_STOP
    assert result["final_errors"] is None and not result["final_checker_completed"]
    assert result["check_history"][0]["model_finish_valid"] is False


def test_no_evidence_uses_no_model(cases, production_nodes, monkeypatch):
    from app import agent
    monkeypatch.setattr(agent, "retriever_node", lambda state: {
        "status": "no_evidence", "evidence": [], "answer": "资料不足，停止。"})
    result = run.run_case(cases[0], responses())
    assert result["status"] == "no_evidence" and result["no_evidence_safe_stop"] is True
    assert result["model_call_count"] == 0 and result["answer"] == "资料不足，停止。"


def test_worker_timeout_recovers_only_observed_lower_bound(cases, tmp_path):
    event_path = tmp_path / "events.jsonl"
    event_path.write_text(json.dumps({"event": "model_call_started", "call": {
        "id": "one", "stage": "Writer", "started": True, "completed": False}}) + "\n" + '{"event":')
    result = run.recover_failed_case(cases[0], event_path, 180, subprocess.TimeoutExpired("worker", 180), "real_model")
    assert result["status"] == "unknown" and result["final_errors"] is None
    assert result["model_call_count"] is None and result["observed_started_model_call_count"] == 1
    assert result["completed_model_call_count"] == 0


def test_worker_timeout_preserves_completed_initial_check_history(cases, production_nodes, tmp_path):
    events = []
    run.run_case(cases[2], responses("证据事实。[1]", "核查响应", TimeoutError("repair timeout")), event_sink=events.append)
    # A killed worker has complete initial observations, but no repair node completion.
    first_repair_start = next(index for index, event in enumerate(events)
        if event["event"] == "stage_started" and event["stage"] == {
            "stage": "Writer", "pass": 1, "completed": False, "elapsed_seconds": None, "error": None})
    journal = tmp_path / "partial.jsonl"
    journal.write_text("\n".join(json.dumps(event, ensure_ascii=False) for event in events[:first_repair_start + 1]))
    result = run.recover_failed_case(cases[2], journal, 180, subprocess.TimeoutExpired("worker", 180), "real_model")
    assert result["raw_initial_draft"] == "证据事实。[1]"
    assert result["initial_draft"].endswith(cases[2]["fault"]["text"])
    assert len(result["check_history"]) == 1
    assert result["detected_errors"] == ["无依据治疗宣称"]
    assert result["final_errors"] is None


def test_custom_invoker_is_explicitly_test_double_and_fixed_order(cases, production_nodes, tmp_path):
    seen = []
    def invoke(case, path, request_timeout, task_timeout):
        seen.append(case["id"])
        return run.run_case(case, responses("证据事实。[1]", "核查响应", "证据事实。[1]"))
    report = run.run_suite(cases, invoke=invoke, event_dir=tmp_path / "events")
    assert seen == list(run.CASE_IDS) and report["execution_mode"] == "test_double"
    assert report["summary"]["engineering_passed_count"] == 3
    with pytest.raises(ValueError, match="fixed order"):
        run.run_suite(list(reversed(cases)), invoke=invoke, event_dir=tmp_path / "other")


def test_comparison_does_not_treat_old_initial_errors_as_final(cases, production_nodes):
    report = {"cases": [run.run_case(case, responses("证据事实。[1]", "核查响应", "证据事实。[1]")) for case in cases]}
    compared = run.comparison(report)
    assert all(row["before_B"]["final_errors"] is None for row in compared["rows"])
    assert compared["causal_effect_estimate"] is None
    assert compared["comparison_design"] == "historical_B_vs_new_ON_not_new_AB"
    assert "未知" in run.render_comparison(compared)
    review = run.build_human_review({**report, "execution_mode": "test_double"})
    assert review["row_count"] == 3 and not review["human_review_completed"]
    assert all(row["medical_quality"] is row["reviewer"] is row["false_positive_confirmed"] is None for row in review["rows"])


def test_new_json_artifacts_reject_overwrite(tmp_path):
    target = tmp_path / "report.json"
    run.write_json(target, {"kept": True})
    with pytest.raises(FileExistsError):
        run.write_json(target, {"overwritten": True})
    assert json.loads(target.read_text()) == {"kept": True}


def test_source_snapshot_covers_current_production_and_knowledge():
    sources = run.source_snapshot()
    assert "app/agent.py" in sources and "app/claim_verifier.py" in sources
    assert "app/writer.py" in sources and "app/evidence_checker.py" in sources
    assert "data/health_knowledge.json" in sources
    assert "evaluations/reliability/run.py" in sources
    assert "evaluations/tests/test_reliability.py" in sources


def test_model_freeze_uses_current_writer_options_and_effective_template(monkeypatch, tmp_path):
    import ollama
    from app.writer import OPTIONS, REQUEST_TIMEOUT_SECONDS, PROTOCOL
    template = "{{ .Prompt }} /no_think"
    class MetadataClient:
        def __init__(self, host, timeout):
            assert host == run.HOST and timeout == 45
        def list(self):
            item = SimpleNamespace(model=run.MODEL, model_dump=lambda mode: {"model": run.MODEL})
            return SimpleNamespace(models=[item])
        def show(self, model):
            assert model == run.MODEL
            return {"template": template}
    manifest = tmp_path / "manifest"
    manifest.write_text('{"schemaVersion": 2, "layers": []}')
    monkeypatch.setattr(ollama, "Client", MetadataClient)
    monkeypatch.setattr(run, "MODEL_MANIFEST", manifest)
    model = run.preflight_model(45)
    assert model["writer_options"] == OPTIONS
    assert model["writer_protocol"] == PROTOCOL
    assert model["production_writer_request_timeout_seconds"] == REQUEST_TIMEOUT_SECONDS
    assert model["template"] == template and model["template_sha256"]
    assert model["manifest_sha256"] == run.sha(manifest)


@pytest.mark.parametrize("output", [("未完成的正文", "length"), TimeoutError("offline simulated read timeout")])
def test_real_production_graph_writer_failure_stops_before_checker_or_reflection(monkeypatch, cases, output):
    from app import agent
    monkeypatch.setattr(agent, "chat", responses(output))
    def reject_check(*args, **kwargs):
        raise AssertionError("Failed Writer must never reach Evidence Checker")
    monkeypatch.setattr(agent, "check_evidence", reject_check)
    state = {"task": cases[0]["task"], "evidence": [], "draft": "", "issues": [],
             "revision_count": 0, "answer": "", "status": "running", "writer_runs": [],
             "generation_failed": False, "checker_details": {}}
    result = agent.graph.invoke(state, config={"recursion_limit": 20})
    assert result["status"] == "evaluation_failed"
    assert result["revision_count"] == 0 and result["generation_failed"] is True
    assert len(result["writer_runs"]) == 1 and result["writer_runs"][0]["error"]
    assert result["checker_details"] == {}
    assert result["answer"] == agent.SAFE_STOP and result["draft"] == ""


def test_real_production_graph_verifier_error_stops_without_repair(monkeypatch, cases):
    from app import agent
    document = next(doc for doc in agent.knowledge_base if "睡眠" in doc["topic"])
    article = "".join(fact + "[1]。" for fact in document["facts"]) + "睡前喝牛奶能够治愈失眠。[1]"
    encoded_article = json.dumps({"article": article}, ensure_ascii=False)
    monkeypatch.setattr(agent, "chat", responses(encoded_article, encoded_article))
    checks = []
    def broken_verifier(claim, documents):
        checks.append(claim)
        raise TimeoutError("offline simulated verifier timeout")
    monkeypatch.setattr(agent, "verify_claim", broken_verifier)
    state = {"task": cases[0]["task"], "evidence": [], "draft": "", "issues": [],
             "revision_count": 0, "answer": "", "status": "running", "writer_runs": [],
             "generation_failed": False, "checker_details": {}}
    result = agent.graph.invoke(state, config={"recursion_limit": 20})
    assert result["status"] == "evaluation_failed" and result["revision_count"] == 0
    assert len(result["writer_runs"]) == 1 and len(checks) == 1
    assert result["checker_details"]["complete"] is False
    assert result["answer"] == agent.SAFE_STOP
