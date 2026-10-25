"""Three fixed Reflection ON checks, using production nodes without changing the API."""

import argparse
import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from unittest.mock import patch
import zipfile

from ..public_fixtures import public_fixture_path


REPO = Path(__file__).resolve().parents[2]
CASES_PATH = REPO / "evaluations/reflection_ab/cases.json"
BASELINE_PATH = public_fixture_path("legacy_reflection_comparison")
MODEL_MANIFEST = Path.home() / ".ollama/models/manifests/registry.ollama.ai/library/qwen3/4b"
CASE_IDS = ("R-NORMAL", "R-INSUFFICIENT", "R-INJECT")
STAGES = ("Retriever", "Writer", "Verifier", "Reflection")
HOST = "http://127.0.0.1:11434"
MODEL = "qwen3:4b"
SAFE_STOP = "本次内容未完成可靠性核查，暂不提供文章，请人工复核。"
QUALITY_WARNING = "检查器通过仅表示工程流程通过，不能等同医学正确；内容质量待独立人工审核。"
DESIGN = "post_repair_reflection_on_sequential"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_value(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    return value


def load_cases():
    original = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    by_id = {case["id"]: case for case in original["cases"]}
    require(len(by_id) == len(original["cases"]), "Duplicate case identifiers")
    selected = [copy.deepcopy(by_id[case_id]) for case_id in CASE_IDS]
    for case in selected:
        require(isinstance(case["task"], str) and bool(case["task"]), "Missing frozen task")
        require(case["fault"]["mode"] in ("none", "append"), "Unknown fault mode")
        require(isinstance(case["fault"]["text"], str), "Invalid fault text")
    return {"suite_id": "healthagent-reliability-three-v1", "source_suite_id": original["suite_id"],
            "source_cases_path": str(CASES_PATH), "source_cases_sha256": sha(CASES_PATH),
            "cases": selected}


def validate_timeouts(timeout_seconds, request_timeout_seconds):
    for value, maximum, label in ((timeout_seconds, 240, "Task"),
                                  (request_timeout_seconds, 60, "Request")):
        require(type(value) in (int, float) and math.isfinite(value) and 0 < value <= maximum,
                f"{label} timeout must be positive, finite and at most {maximum} seconds")


@contextmanager
def clean_injection_environment():
    names = ("INJECT_BAD_FACT", "INJECT_FALSE_CLAIM")
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ.pop(name, None)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _real_client_method(chat_fn):
    from ollama import Client
    client = getattr(chat_fn, "__self__", None)
    return (isinstance(client, Client) and getattr(chat_fn, "__name__", None) == "chat"
            and str(client._client.base_url).rstrip("/") == HOST)


def run_case(case, chat_fn, *, execution_mode="test_double", event_sink=None):
    """Observe current production nodes; never substitute prompts, options or checking rules."""
    from app import agent, claim_verifier
    require(execution_mode in ("test_double", "real_model"), "Unknown execution mode")
    require(execution_mode != "real_model" or _real_client_method(chat_fn),
            "Only the bound local Ollama Client.chat can identify a real-model execution")
    frozen = {item["id"]: item for item in load_cases()["cases"]}
    require(case.get("id") in frozen and digest(case) == digest(frozen[case["id"]]),
            "Only unchanged frozen tasks and faults are accepted")
    emit = event_sink or (lambda event: None)
    started = time.perf_counter()
    state = {"task": case["task"], "evidence": [], "draft": "", "issues": [],
             "revision_count": 0, "answer": "", "status": "pending"}
    calls, stages, check_history = [], [], []
    active_stage = {"name": None, "pass": 0}
    initial_draft = raw_initial_draft = None
    final_checker_completed = False
    final_errors = None
    detected_errors = []
    error = None
    failure_kind = None

    def observed_chat(*args, **kwargs):
        require(not args, "Production calls must use named parameters for observation")
        call = {"id": f"{case['id']}-{len(calls) + 1:03d}", "stage": active_stage["name"],
                "pass": active_stage["pass"], "started": True, "completed": False,
                "response_received": False, "request": json_value(copy.deepcopy(kwargs)),
                "model": kwargs.get("model"), "think": kwargs.get("think"),
                "options": json_value(copy.deepcopy(kwargs.get("options", {}))),
                "eval_count": None, "prompt_eval_count": None, "done_reason": None,
                "truncated": None, "error": None, "elapsed_seconds": None}
        require(call["model"] == MODEL and call["think"] is False,
                "Unexpected model or thinking configuration")
        calls.append(call)
        emit({"event": "model_call_started", "call": copy.deepcopy(call)})
        call_start = time.perf_counter()
        try:
            response = chat_fn(**kwargs)
            payload = json_value(response)
            require(isinstance(payload, dict), "Model response must be an object")
            call.update({"response_received": True, "response": payload, "completed": True,
                         "eval_count": payload.get("eval_count"),
                         "prompt_eval_count": payload.get("prompt_eval_count"),
                         "done_reason": payload.get("done_reason"),
                         "done": payload.get("done"),
                         "truncated": payload.get("done_reason") == "length"})
            return response
        except Exception as exc:
            call["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            call["elapsed_seconds"] = time.perf_counter() - call_start
            emit({"event": "model_call_finished", "call": copy.deepcopy(call)})

    def node(name, function, pass_number):
        active_stage.update({"name": name, "pass": pass_number})
        item = {"stage": name, "pass": pass_number, "completed": False,
                "elapsed_seconds": None, "error": None}
        stages.append(item)
        emit({"event": "stage_started", "stage": copy.deepcopy(item)})
        node_start = time.perf_counter()
        try:
            update = function(copy.deepcopy(state))
            require(isinstance(update, dict), "Production node did not return state fields")
            state.update(update)
            item["completed"] = True
            return update
        except Exception as exc:
            item["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            item["elapsed_seconds"] = time.perf_counter() - node_start
            emit({"event": "stage_finished", "stage": copy.deepcopy(item),
                  "state": json_value(copy.deepcopy(state))})

    def writer_failed():
        runs = state.get("writer_runs", [])
        return (state.get("generation_failed") is True
                or state.get("status") == "draft_generation_failed"
                or not runs
                or runs[-1].get("done") is not True
                or runs[-1].get("done_reason") != "stop"
                or bool(runs[-1].get("truncated") or runs[-1].get("error")))

    def check(pass_number):
        nonlocal final_checker_completed, final_errors
        node("Verifier", agent.evidence_checker_node, pass_number)
        details = copy.deepcopy(state.get("checker_details", {}))
        issues = copy.deepcopy(state.get("issues", []))
        observed_verifier_calls = [call for call in calls
                                   if call["stage"] == "Verifier" and call["pass"] == pass_number]
        invalid_finish = any(call.get("truncated") is True or call.get("done") is not True
                             or call.get("done_reason") != "stop"
                             or call.get("error") or call.get("response_received") is not True
                             for call in observed_verifier_calls)
        check_history.append({"pass": pass_number, "draft": state.get("draft"),
                              "issues": issues, "checker_details": details,
                              "model_finish_valid": not invalid_finish})
        for issue in issues:
            if issue not in detected_errors:
                detected_errors.append(issue)
        final_checker_completed = details.get("complete") is True and not invalid_finish
        final_errors = issues if final_checker_completed else None
        if state.get("status") == "completed":
            require(final_checker_completed and not issues, "Incomplete/failed checker cannot complete")

    try:
        with clean_injection_environment(), patch.object(agent, "chat", observed_chat), \
                patch.object(claim_verifier, "chat", observed_chat):
            node("Retriever", agent.retriever_node, 0)
            if state.get("evidence"):
                node("Writer", agent.writer_node, 0)
                raw_initial_draft = state.get("draft")
                if not writer_failed():
                    if case["fault"]["mode"] == "append":
                        state["draft"] += "\n" + case["fault"]["text"]
                    initial_draft = state["draft"]
                    check(0)
                    if state.get("status") == "needs_revision" and final_checker_completed:
                        final_checker_completed, final_errors = False, None
                        node("Reflection", agent.reflection_node, 1)
                        require(state.get("revision_count") == 1, "Exactly one repair is permitted")
                        node("Writer", agent.writer_node, 1)
                        if not writer_failed():
                            check(1)
                else:
                    initial_draft = state.get("draft")
                if writer_failed():
                    failure_kind = ("draft_generation_failed" if state.get("generation_failed")
                                    else "writer_finish_metadata_invalid")
                    final_checker_completed, final_errors = False, None
                    node("SafetyStop", agent.stop_node, state["revision_count"])
                elif state.get("status") != "completed":
                    failure_kind = ("checker_incomplete" if not final_checker_completed
                                    else "checker_issues_after_max_revision")
                    node("SafetyStop", agent.stop_node, state["revision_count"])
            else:
                require(state.get("status") == "no_evidence", "Retriever returned inconsistent state")
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        failure_kind = "observer_or_node_exception"
        final_checker_completed, final_errors = False, None
    production_answer = state.get("answer")
    observed_status = "error" if error else state.get("status")
    answer = production_answer if production_answer or not error else SAFE_STOP
    elapsed = time.perf_counter() - started
    result = {"case_id": case["id"], "case": copy.deepcopy(case), "design": DESIGN,
              "execution_mode": execution_mode, "strategy": "Reflection ON (max 1)",
              "status": observed_status, "answer": answer,
              "production_answer": production_answer, "raw_final_state": copy.deepcopy(state),
              "failure_kind": failure_kind,
              "draft": state.get("draft"), "raw_initial_draft": raw_initial_draft,
              "initial_draft": initial_draft, "evidence": state.get("evidence"),
              "detected_errors": detected_errors, "final_errors": final_errors,
              "final_checker_completed": final_checker_completed,
              "revision_count": state.get("revision_count"),
              "writer_metadata": copy.deepcopy(state.get("writer_runs", [])),
              "checker_details": copy.deepcopy(state.get("checker_details", {})),
              "check_history": check_history, "model_calls": calls,
              "model_call_count": len(calls), "observed_started_model_call_count": len(calls),
              "completed_model_call_count": sum(call["completed"] for call in calls),
              "stage_seconds": {name: sum(item["elapsed_seconds"] for item in stages
                                           if item["stage"] == name) for name in STAGES},
              "stages": stages, "total_seconds": elapsed, "error": error,
              "no_evidence_safe_stop": (not calls and state.get("status") == "no_evidence")
                  if not state.get("evidence") else None,
              "false_positive_count": None, "medical_quality": None,
              "quality_status": "pending_independent_human_review"}
    emit({"event": "case_finished", "result": copy.deepcopy(result)})
    return result


def invoke_worker(case, event_path, request_timeout_seconds, timeout_seconds):
    completed = subprocess.run([sys.executable, "-m", "evaluations.reliability.worker"], cwd=REPO,
        input=json.dumps({"case": case, "event_path": str(Path(event_path).resolve()),
                          "request_timeout_seconds": request_timeout_seconds}, ensure_ascii=False),
        capture_output=True, text=True, timeout=timeout_seconds, check=True)
    return json.loads(completed.stdout)


def read_events(path):
    events = []
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
                if isinstance(item, dict):
                    events.append(item)
            except json.JSONDecodeError:
                pass  # A terminated process can leave one unfinished final line.
    return events


def recover_failed_case(case, event_path, elapsed, exc, execution_mode):
    calls, stages, state = {}, {}, {}
    check_history, detected_errors = [], []
    raw_initial_draft = initial_draft = None
    for event in read_events(event_path):
        if event.get("event") in ("model_call_started", "model_call_finished"):
            call = event["call"]
            calls[call["id"]] = call
        if event.get("event") in ("stage_started", "stage_finished"):
            stage = event["stage"]
            stages[(stage["stage"], stage["pass"])] = stage
            state = event.get("state", state)
            if event["event"] == "stage_finished" and stage.get("completed"):
                if stage["stage"] == "Writer" and stage["pass"] == 0:
                    raw_initial_draft = state.get("draft")
                    initial_draft = raw_initial_draft
                    if (state.get("generation_failed") is not True
                            and case["fault"]["mode"] == "append" and isinstance(initial_draft, str)):
                        initial_draft += "\n" + case["fault"]["text"]
                if stage["stage"] == "Verifier":
                    issues = copy.deepcopy(state.get("issues", []))
                    check_history.append({"pass": stage["pass"], "draft": state.get("draft"),
                        "issues": issues, "checker_details": copy.deepcopy(state.get("checker_details", {})),
                        "observation_note": "Recovered completed node from journal; full task result unavailable"})
                    for issue in issues:
                        if issue not in detected_errors:
                            detected_errors.append(issue)
    call_list = list(calls.values())
    return {"case_id": case["id"], "case": copy.deepcopy(case), "design": DESIGN,
            "execution_mode": execution_mode, "strategy": "Reflection ON (max 1)",
            "status": "unknown", "answer": state.get("answer") or SAFE_STOP,
            "production_answer": state.get("answer"), "raw_final_state": copy.deepcopy(state),
            "failure_kind": "worker_result_unavailable", "draft": state.get("draft"),
            "raw_initial_draft": raw_initial_draft, "initial_draft": initial_draft, "evidence": state.get("evidence"),
            "detected_errors": detected_errors, "final_errors": None,
            "final_checker_completed": False, "revision_count": state.get("revision_count"),
            "writer_metadata": state.get("writer_runs", []),
            "checker_details": state.get("checker_details", {}), "check_history": check_history,
            "model_calls": call_list, "model_call_count": None,
            "observed_started_model_call_count": len(call_list),
            "completed_model_call_count": sum(call.get("completed") is True for call in call_list),
            "stage_seconds": {name: (None if any(item.get("elapsed_seconds") is None for item in stages.values()
                                                 if item["stage"] == name)
                                      else sum(item["elapsed_seconds"] for item in stages.values()
                                               if item["stage"] == name)) for name in STAGES},
            "stages": list(stages.values()), "total_seconds": elapsed,
            "error": {"type": type(exc).__name__, "message": str(exc),
                      "stderr": (getattr(exc, "stderr", "") or "")[-4000:]},
            "observation_note": "Worker未返回完整结果；调用数仅有事件日志观测下界，末轮错误未知。",
            "no_evidence_safe_stop": None, "false_positive_count": None, "medical_quality": None,
            "quality_status": "pending_independent_human_review"}


def validate_result(result, case, execution_mode):
    require(result.get("case_id") == case["id"] and result.get("case") == case,
            "Worker returned a different task or fault")
    require(result.get("execution_mode") == execution_mode and result.get("design") == DESIGN,
            "Execution identity differs; test doubles cannot masquerade as real model runs")
    require(type(result.get("revision_count")) is int and 0 <= result["revision_count"] <= 1,
            "Invalid repair count")
    require(result.get("model_call_count") == len(result.get("model_calls", [])), "Call count differs")
    require(all(call.get("model") == MODEL and call.get("think") is False
                for call in result["model_calls"]), "Unexpected model configuration")
    if result.get("status") == "completed":
        require(result.get("final_checker_completed") is True and result.get("final_errors") == [],
                "Completed result requires a full successful final checker")
        require(not any(run.get("truncated") or run.get("error") for run in result.get("writer_metadata", [])),
                "Writer failure cannot produce completed")
        require(bool(result.get("writer_metadata")) and all(
                    item.get("done") is True and item.get("done_reason") == "stop"
                    for item in result["writer_metadata"]), "Incomplete Writer metadata cannot produce completed")
        final_pass = result["revision_count"]
        require(not any(call.get("truncated") is True or call.get("done") is not True
                        or call.get("done_reason") != "stop"
                        or call.get("error") or call.get("response_received") is not True
                        for call in result["model_calls"]
                        if call.get("stage") == "Verifier" and call.get("pass") == final_pass),
                "An incomplete final Verifier response cannot produce completed")


def run_suite(cases, timeout_seconds=180, request_timeout_seconds=45, *, invoke=None,
              event_dir=None, on_result=None):
    validate_timeouts(timeout_seconds, request_timeout_seconds)
    require(cases == load_cases()["cases"], "Only the three fixed tasks in fixed order may run")
    selected_invoke = invoke or invoke_worker
    mode = "real_model" if selected_invoke is invoke_worker else "test_double"
    require(event_dir is not None, "An append-only event directory is required")
    Path(event_dir).mkdir(parents=True, exist_ok=False)
    results = []
    for case in cases:
        event_path = Path(event_dir) / (case["id"] + ".jsonl")
        started = time.perf_counter()
        try:
            result = selected_invoke(case, event_path, request_timeout_seconds, timeout_seconds)
            validate_result(result, case, mode)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
            result = recover_failed_case(case, event_path, time.perf_counter() - started, exc, mode)
        results.append(result)
        if on_result:
            on_result(result)
    return {"schema_version": "1.0", "design": DESIGN, "execution_mode": mode,
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "cases": results,
            "summary": summarize(results), "quality_warning": QUALITY_WARNING}


def summarize(results):
    calls = [call for row in results for call in row["model_calls"]]
    passed = sum(row["status"] == "completed" and row["final_checker_completed"] for row in results)
    return {"task_count": len(results), "engineering_passed_count": passed,
            "engineering_pass_rate": passed / len(results) if results else None,
            "no_evidence_safe_stop_count": sum(row.get("no_evidence_safe_stop") is True for row in results),
            "model_call_count_known": sum(row["model_call_count"] or 0 for row in results),
            "model_call_count_unknown_tasks": sum(row["model_call_count"] is None for row in results),
            "observed_started_model_call_count_lower_bound": len(calls),
            "completed_model_call_count": sum(call.get("completed") is True for call in calls),
            "truncation_count_observed": sum(call.get("truncated") is True for call in calls),
            "timeout_count_observed": sum("timeout" in json.dumps(call.get("error") or {}).lower() for call in calls)
                + sum("TimeoutExpired" == (row.get("error") or {}).get("type") for row in results),
            "stage_seconds_known": {name: sum(row["stage_seconds"][name] or 0 for row in results)
                                    for name in STAGES},
            "total_seconds": sum(row["total_seconds"] for row in results),
            "remaining_error_count_unknown_tasks": sum(row["final_errors"] is None for row in results),
            "false_positive_count": None, "medical_correctness": None,
            "independent_human_reviews_completed": 0}


def comparison(report, baseline=None):
    baseline = baseline or json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    old = {pair["case_id"]: pair for pair in baseline["pairs"]}
    rows = []
    for current in report["cases"]:
        pair = old[current["case_id"]]
        require(pair["case"]["task"] == current["case"]["task"]
                and pair["case"]["fault"] == current["case"]["fault"], "Baseline task/fault differs")
        prior = copy.deepcopy(pair["rows"]["B"])
        # Historical B stored initial issues after a failed repair. They are not a completed final check.
        prior["final_errors"] = prior.get("final_errors") if prior.get("final_checker_completed") is True else None
        rows.append({"case_id": current["case_id"], "same_task_and_fault": True,
                     "before_B": prior, "after_ON": current})
    return {"baseline_path": str(BASELINE_PATH), "baseline_sha256": sha(BASELINE_PATH),
            "comparison_design": "historical_B_vs_new_ON_not_new_AB",
            "causal_effect_estimate": None,
            "limitations": ["采样时间不同，不能将差异归因于单一因素。",
                "Writer提示、长度上限及核查规则已改变，旧B为共享初稿分叉，新ON为独立顺序运行。",
                "旧B末轮未完成时仍存在错误数为未知，不沿用初检问题数，也不能记为0。",
                QUALITY_WARNING],
            "before_parameters": baseline.get("freeze", {}).get("model"),
            "after_parameters": report.get("freeze", {}).get("model"), "rows": rows}


def build_human_review(report):
    return {"human_review_completed": False, "row_count": len(report["cases"]),
            "execution_mode": report["execution_mode"], "instructions": QUALITY_WARNING,
            "rows": [{"case_id": row["case_id"], "task": row["case"]["task"],
                "status": row["status"], "answer": row["answer"], "draft": row["draft"],
                "initial_draft": row["initial_draft"], "evidence": row["evidence"],
                "check_history": row["check_history"], "false_positive_candidates": row["detected_errors"],
                "false_positive_candidate_note": "仅为检查诊断，是否误拦截须逐句人工判定；尚无真实误拦截计数。",
                "medical_quality": None, "task_fulfillment": None, "evidence_faithfulness": None,
                "false_positive_confirmed": None, "human_review_notes": None,
                "reviewer": None, "reviewed_on": None, "review_status": "pending_independent_human_review"}
                for row in report["cases"]]}


def git_snapshot():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()
    return {"head": git("rev-parse", "HEAD"),
            "status_porcelain": git("status", "--porcelain=v1", "--untracked-files=all"),
            "automatic_commit": False}


def source_snapshot():
    paths = set((REPO / "app").rglob("*.py"))
    paths.update((REPO / "evaluations/reliability").glob("*.py"))
    paths.add(REPO / "evaluations/reliability/README.md")
    paths.update((REPO / "data").rglob("*"))
    paths.update((CASES_PATH, BASELINE_PATH, REPO / "evaluations/tests/test_reliability.py"))
    return {str(path.relative_to(REPO)): sha(path) for path in sorted(paths) if path.is_file()}


def preflight_model(request_timeout_seconds):
    from ollama import Client
    from app.writer import OPTIONS as WRITER_OPTIONS, REQUEST_TIMEOUT_SECONDS, PROTOCOL
    client = Client(host=HOST, timeout=request_timeout_seconds)
    listed = client.list()
    models = [json_value(item) for item in listed.models if getattr(item, "model", None) == MODEL]
    require(models and MODEL_MANIFEST.is_file(), "The fixed local model or manifest is unavailable")
    shown = json_value(client.show(MODEL))
    template = shown.get("template", "")
    require(isinstance(template, str) and template, "The effective model template is unavailable")
    return {"name": MODEL, "host": HOST, "think": False,
            "writer_options": copy.deepcopy(WRITER_OPTIONS),
            "writer_protocol": PROTOCOL,
            "production_writer_request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
            "request_parameters_note": "每次真实请求的原始options保存在model_calls中；不覆盖生产node参数。",
            "sdk_version": version("ollama"), "manifest_path": str(MODEL_MANIFEST),
            "manifest_sha256": sha(MODEL_MANIFEST), "manifest": json.loads(MODEL_MANIFEST.read_text()),
            "template_sha256": hashlib.sha256(template.encode()).hexdigest(),
            "template": template, "show_metadata": shown, "list_metadata": models,
            "metadata_only_probe": True}


def reverify_freeze(freeze):
    changed = [relative for category in ("source_sha256", "historical_result_sha256")
               for relative, expected in freeze[category].items()
               if not (REPO / relative).is_file() or sha(REPO / relative) != expected]
    if not MODEL_MANIFEST.is_file() or sha(MODEL_MANIFEST) != freeze["model"]["manifest_sha256"]:
        changed.append("local_model_manifest")
    return {"passed": not changed, "changed_paths": changed}


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as target:
        json.dump(value, target, ensure_ascii=False, indent=2)
        target.write("\n")


def display(value):
    return "未知" if value is None else (f"{value:.3f}" if isinstance(value, float)
        else str(value).replace("|", "\\|").replace("\n", " "))


def render_report(report):
    summary = report["summary"]
    lines = ["# HealthAgent 工程可靠性修复后小规模验证", "",
        f"执行模式：`{report['execution_mode']}`。仅3条独立Reflection ON流程，各最多修正一次；这不是新的A/B实验。", "",
        QUALITY_WARNING, "",
        f"工程通过：{summary['engineering_passed_count']}/{summary['task_count']}；"
        f"已知调用数 {summary['model_call_count_known']}，调用数未知任务 {summary['model_call_count_unknown_tasks']}；"
        f"观测截断 {summary['truncation_count_observed']}，观测超时 {summary['timeout_count_observed']}。", "",
        "|任务|状态|修正|调用|Writer秒|Verifier秒|Reflection秒|Retriever秒|总秒|末轮检出问题|", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["cases"]:
        values = [row["case_id"], row["status"], row["revision_count"], row["model_call_count"],
                  *(row["stage_seconds"][stage] for stage in ("Writer", "Verifier", "Reflection", "Retriever")),
                  row["total_seconds"], None if row["final_errors"] is None else len(row["final_errors"])]
        lines.append("|" + "|".join(display(value) for value in values) + "|")
    lines += ["", "Reflection仅路由与计数；重写耗时计入Writer，重查耗时计入Verifier。"
        "调用completed表示收到了响应，不代表生成合格；长度截断仍安全停止。", "",
        "真实误拦截次数尚未裁定；human_review.json中的诊断候选待人工逐句核对。", ""]
    for row in report["cases"]:
        lines += [f"## {row['case_id']} / {row['status']}", "", row["case"]["task"], "",
                  "最终回答：", "", row["answer"], "", "累计检测：" + ("；".join(row["detected_errors"]) or "未记录"),
                  "末轮检测：" + ("未知，末轮未完整核查" if row["final_errors"] is None else "；".join(row["final_errors"]) or "未检出"), "",
                  "Writer记录：`" + json.dumps(row["writer_metadata"], ensure_ascii=False).replace("`", "'") + "`", ""]
    if report.get("comparison"):
        lines += render_comparison(report["comparison"]).splitlines()
    return "\n".join(lines) + "\n"


def render_comparison(compared):
    lines = ["## 历史B与修复后ON对照", "",
        f"历史公开投影：`{compared['baseline_path']}`；投影SHA256 `{compared['baseline_sha256']}`；原报告指纹见夹具来源。", "",
        *compared["limitations"], "",
        "|任务|历史B状态|修复后ON状态|历史/新Writer秒|历史/新Verifier秒|历史/新总秒|历史/新调用|历史/新末轮问题|",
        "|---|---|---|---|---|---|---|---|"]
    for row in compared["rows"]:
        before, after = row["before_B"], row["after_ON"]
        pair = lambda key: display(before.get(key)) + " / " + display(after.get(key))
        stages = lambda name: display(before["stage_seconds"][name]) + " / " + display(after["stage_seconds"][name])
        errors = lambda item: None if item["final_errors"] is None else len(item["final_errors"])
        lines.append("|" + "|".join([row["case_id"], before["status"], after["status"], stages("Writer"), stages("Verifier"),
            pair("total_seconds"), pair("model_call_count"), display(errors(before)) + " / " + display(errors(after))]) + "|")
    lines += ["", "旧参数：`" + json.dumps(compared["before_parameters"], ensure_ascii=False) + "`", "",
              "新参数及模板哈希：见freeze.json及每次model_calls原始请求。", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true", help="Validate exactly three tasks offline")
    mode.add_argument("--run-small", action="store_true", help="Run exactly three fixed ON tasks, without retries")
    parser.add_argument("--timeout-seconds", type=float, default=180)
    parser.add_argument("--request-timeout-seconds", type=float, default=45)
    parser.add_argument("--output-dir", type=Path, default=REPO / "evaluations/results")
    args = parser.parse_args(argv)
    try:
        validate_timeouts(args.timeout_seconds, args.request_timeout_seconds)
        suite = load_cases()
    except (ValueError, KeyError, OSError, TypeError) as exc:
        parser.error(str(exc))
    if args.validate_only:
        print(json.dumps({"validation": "passed", "case_ids": list(CASE_IDS), "design": DESIGN,
                          "max_revisions": 1, "source_cases_sha256": suite["source_cases_sha256"],
                          "model_calls": 0, "network_calls": 0}, ensure_ascii=False, indent=2))
        return 0
    sources = source_snapshot()
    history = {str(path.relative_to(REPO)): sha(path)
               for path in sorted((REPO / "evaluations/results").rglob("*")) if path.is_file()}
    model = preflight_model(args.request_timeout_seconds)
    output = args.output_dir.resolve() / ("reliability-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                                         + "-" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    freeze = {"schema_version": "1.0", "suite": suite, "execution_mode": "real_model", "design": DESIGN,
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": sources,
              "historical_result_sha256": history, "model": model, "git_before": git_snapshot(),
              "python_version": sys.version, "timeout_seconds_per_task": args.timeout_seconds,
              "request_timeout_seconds": args.request_timeout_seconds, "retries": 0,
              "max_revisions": 1, "case_order": list(CASE_IDS)}
    write_json(output / "freeze.json", freeze)
    with zipfile.ZipFile(output / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for relative, expected in sources.items():
            require(sha(REPO / relative) == expected, "Source changed before inference: " + relative)
            archive.write(REPO / relative, relative)
    with (output / "progress.jsonl").open("x", encoding="utf-8") as progress:
        def record(result):
            progress.write(json.dumps(result, ensure_ascii=False) + "\n")
            progress.flush()
            os.fsync(progress.fileno())
            print(f"{result['case_id']}: {result['status']}, calls={result['model_call_count']}", flush=True)
        print(f"REAL MODEL: exactly 3 sequential ON tasks, no retries; output={output}", flush=True)
        report = run_suite(suite["cases"], args.timeout_seconds, args.request_timeout_seconds,
                           event_dir=output / "events", on_result=record)
    report.update({"freeze": freeze, "freeze_reverification": reverify_freeze(freeze), "git_after": git_snapshot()})
    report["comparison"] = comparison(report)
    write_json(output / "report.json", report)
    with (output / "report.md").open("x", encoding="utf-8") as target:
        target.write(render_report(report))
    write_json(output / "human_review.json", build_human_review(report))
    print(f"Report: {output / 'report.md'}", flush=True)
    return 1 if (not report["freeze_reverification"]["passed"] or any(
        row["status"] in ("error", "unknown", "draft_generation_failed") for row in report["cases"])) else 0


if __name__ == "__main__":
    raise SystemExit(main())
