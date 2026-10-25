"""Independent production-node Reflection OFF/ON runs with unchanged requests.

Each strategy executes its own end-to-end graph. The evaluation graph reuses the
production nodes and every production routing callable; OFF changes only the
Checker's ``reflect`` destination to ``stop``. No app source or prompt is edited.
"""

import copy
from contextlib import ExitStack, redirect_stdout
import hashlib
import io
import json
import marshal
import os
from pathlib import Path
import time
import types
from unittest.mock import patch

from langgraph.graph import END, START, StateGraph

from app import agent, claim_verifier, writer
from app.evidence_checker import MAX_MEDICAL_CLAIMS, classify_segments
from app.knowledge import load_knowledge_base
from app.task_parameters import check_task_parameter_fidelity


CASES_PATH = Path(__file__).with_name("cases.json")
MODEL = writer.MODEL
OPTIONS = dict(writer.OPTIONS)  # Compatibility name; never applied to Verifier.
STAGES = ("Retriever", "Writer", "Verifier", "Reflection")
MAX_CALLS_OFF = 1 + MAX_MEDICAL_CLAIMS
MAX_CALLS_ON = 2 * (1 + MAX_MEDICAL_CLAIMS)
MAX_CALLS_PER_PAIR = MAX_CALLS_OFF + MAX_CALLS_ON
SAFE_STOP = writer.SAFE_STOP
DESIGN = "independent_end_to_end_paired"


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _plain(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, types.SimpleNamespace):
        return {key: _plain(item) for key, item in vars(value).items()}
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def production_config():
    """Read current production settings without a network or model call.

    Verifier settings are captured from its actual callable, rather than copied
    from the Writer or duplicated as evaluation defaults.
    """
    captured = {}

    def capture(**kwargs):
        captured.update(copy.deepcopy(kwargs))
        return types.SimpleNamespace(
            message=types.SimpleNamespace(content=json.dumps({
                "verdict": "not_supported", "reason": "configuration capture only", "evidence_id": 0})),
            done=True, done_reason="stop")

    with patch.object(claim_verifier, "chat", capture):
        claim_verifier.verify_claim("configuration capture only", [{"facts": ["configuration capture only"]}])
    verifier = {key: copy.deepcopy(captured[key]) for key in ("model", "think", "options", "format")}
    verifier.update(request_timeout_seconds=writer.REQUEST_TIMEOUT_SECONDS,
                    system_prompt_sha256=digest(captured["messages"][0]["content"]),
                    configuration_source="actual_production_verify_claim_request_offline_capture")
    return {
        "writer": {"model": writer.MODEL, "think": False, "options": dict(writer.OPTIONS),
                   "format": copy.deepcopy(writer.OUTPUT_SCHEMA), "protocol": writer.PROTOCOL,
                   "request_timeout_seconds": writer.REQUEST_TIMEOUT_SECONDS,
                   "system_prompt_sha256": digest(writer.SYSTEM_PROMPT)},
        "verifier": verifier,
        "knowledge_base_sha256": digest(load_knowledge_base()),
        "max_reflection_revisions": agent.MAX_REVISIONS,
        "max_medical_claims": MAX_MEDICAL_CLAIMS,
    }


def _loaded_proof(function):
    path = Path(function.__code__.co_filename).resolve()
    compiled = compile(path.read_text(encoding="utf-8"), str(path), "exec", dont_inherit=True)
    code = next(item for item in compiled.co_consts
                if isinstance(item, types.CodeType) and item.co_name == function.__name__)
    if code != function.__code__:
        raise ValueError(f"Loaded function differs from current source: {function.__name__}")
    return {"function": function.__name__, "path": str(path), "source_sha256": _sha(path),
            "loaded_code_sha256": hashlib.sha256(marshal.dumps(function.__code__)).hexdigest(),
            "compiled_current_source_equals_loaded": True}


def load_cases(path=None):
    result = json.loads(Path(path or CASES_PATH).read_text(encoding="utf-8"))
    cases = result.get("cases", [])
    if (result.get("schema_version") != "1.0" or len(cases) != 4
            or len({case["id"] for case in cases}) != 4):
        raise ValueError("The bounded Reflection suite must contain exactly four unique tasks")
    categories = {"normal_sleep", "no_evidence", "unsupported_claim_injection", "insufficient_evidence"}
    if {case.get("category") for case in cases} != categories:
        raise ValueError("The four required task categories must be present")
    for case in cases:
        if not isinstance(case.get("task"), str) or not case["task"].strip():
            raise ValueError("Task must be nonempty text")
        fault = case.get("fault", {})
        if fault.get("mode") not in ("none", "append", "prepend") or not isinstance(fault.get("text"), str):
            raise ValueError("Invalid deterministic fault fixture")
        if (fault["mode"] == "none") != (fault["text"] == ""):
            raise ValueError("Only injected cases may have fault text")
        retrieved = any(word in case["task"] for doc in load_knowledge_base() for word in doc["keywords"])
        if retrieved == (case["category"] == "no_evidence"):
            raise ValueError("No-evidence task must retrieve nothing; other tasks must retrieve local facts")
    return result


def claim_candidates(draft):
    return [item["text"] for item in classify_segments(draft) if item["kind"] == "medical_claim"]


def apply_fault(draft, fault):
    if fault["mode"] == "none":
        return draft
    return (draft + "\n" + fault["text"] if fault["mode"] == "append"
            else fault["text"] + "\n" + draft)


def _initial_state(task):
    return {"task": task, "evidence": [], "draft": "", "issues": [],
            "revision_count": 0, "answer": "", "status": "running",
            "writer_runs": [], "generation_failed": False, "checker_details": {}}


def _terminal_error(state):
    if state.get("generation_failed"):
        runs = state.get("writer_runs", [])
        return copy.deepcopy(runs[-1].get("error") if runs else None) or {
            "type": "ValueError", "message": "Writer failed without a complete draft"}
    return next((copy.deepcopy(item["error"])
                 for item in state.get("checker_details", {}).get("claims", []) if item.get("error")), None)


def _gate_outcomes(state, checks, error):
    details = state.get("checker_details", {})
    latest = checks[-1] if checks else None
    current = bool(latest and latest["pass"] == state["revision_count"])
    complete = bool(current and details.get("complete") is True)
    medical = (complete and not details.get("unchecked_claims")
               and all(not item.get("error") and not item.get("quantity_issues")
                       and (item.get("verification") or {}).get("verdict") == "supported"
                       for item in details.get("claims", []))) if current else None
    parameter = details.get("task_parameter_fidelity", {}).get("passed") if current else None
    content = details.get("content_assessment", {}).get("status") == "ok" if current else None
    checker = bool(current and complete and not state["issues"] and parameter is True
                   and latest["status"] == "completed" and not state.get("generation_failed") and error is None)
    if state["status"] == "completed":
        publication = bool(checker and state["answer"] == latest["answer"])
    elif state["status"] == "no_evidence":
        publication = not state["evidence"] and not state["draft"]
    elif state["status"] == "insufficient_content":
        publication = bool(state["answer"] and state["answer"] != state.get("draft"))
    elif state["status"] == "evaluation_failed":
        publication = state["answer"] == SAFE_STOP
    else:
        publication = False
    return {"medical_evidence_passed": medical, "task_parameter_fidelity_passed": parameter,
            "content_requirements_passed": content, "checker_passed": checker if current else None,
            "publication_integrity_passed": publication, "medical_review_passed": None}


def run_case(case, strategy, chat_fn, execution_mode="test_double", event_sink=None):
    """Run one independent production-node task; only OFF suppresses rewriting."""
    if strategy not in ("A", "B") or execution_mode not in ("test_double", "real_model"):
        raise ValueError("Invalid strategy or execution mode")
    if agent.MAX_REVISIONS != 1:
        raise ValueError("Production must retain its one-revision bound")
    config = production_config()
    if agent.MODEL != writer.MODEL or claim_verifier.MODEL != writer.MODEL:
        raise ValueError("Writer and Verifier must use the same production model")
    original_nodes = {"Retriever": agent.retriever_node, "Writer": agent.writer_node,
                      "Verifier": agent.evidence_checker_node, "Reflection": agent.reflection_node,
                      "Stop": agent.stop_node}
    # Reuse the actual compiled production Writer condition, including failures.
    writer_route = agent.builder.branches["writer"]["condition"].path.func
    retriever_route = agent.builder.branches["retriever"]["route_retriever"].path.func
    checker_route = agent.builder.branches["checker"]["route_checker"].path.func
    if retriever_route is not agent.route_retriever or checker_route is not agent.route_checker:
        raise ValueError("Production routing identity changed")
    proofs = [_loaded_proof(function) for function in [*original_nodes.values(), retriever_route,
               writer_route, checker_route, writer.generate_article, claim_verifier.verify_claim,
               agent.check_evidence, check_task_parameter_fidelity]]
    root = Path(agent.__file__).resolve().parents[1]
    sources = sorted((root / "app").glob("*.py")) + [root / "data/health_knowledge.json", CASES_PATH]
    source_hashes = {str(path.relative_to(root)): _sha(path) for path in sources}
    started = time.perf_counter()
    calls, stages, checks, faults = [], [], [], []
    current = {"phase": strategy, "stage": None, "pass": 0}
    last_state = _initial_state(case["task"])
    raw_initial_draft, initial_draft = "", ""
    logs = io.StringIO()
    thrown_error = None

    def emit(event):
        if event_sink:
            event_sink({"case_id": case["id"], "strategy": strategy, **copy.deepcopy(event)})

    def observed_chat(*args, **kwargs):
        budget = MAX_CALLS_OFF if strategy == "A" else MAX_CALLS_ON
        if len(calls) >= budget:
            raise ValueError("Bounded per-task model-call budget exceeded")
        stage = current["stage"]
        expected = config["writer" if stage == "Writer" else "verifier"]
        if (stage not in ("Writer", "Verifier") or kwargs.get("model") != expected["model"]
                or kwargs.get("think") != expected["think"]
                or kwargs.get("options") != expected["options"]
                or kwargs.get("format") != expected["format"]):
            raise ValueError("Production request differs from captured production configuration")
        call = {"id": f"{case['id']}/{strategy}/{len(calls) + 1}", **current,
                "model": kwargs.get("model"), "think": kwargs.get("think"),
                "options": copy.deepcopy(kwargs.get("options")),
                "request": copy.deepcopy(kwargs), "request_sha256": digest(kwargs),
                "response": None, "response_content": None, "completed": False,
                "response_received": False, "error": None, "elapsed_seconds": None, "truncated": False}
        calls.append(call)
        emit({"event": "model_call_started", "call": call})
        begin = time.perf_counter()
        try:
            # No option, message, schema or response rewriting.
            response = chat_fn(*args, **kwargs)
            call.update(response_received=True, response=_plain(response), response_content=response.message.content or "")
            for attr in ("done", "done_reason", "eval_count", "prompt_eval_count", "total_duration"):
                call[attr] = getattr(response, attr, None)
            call["truncated"] = call.get("done_reason") == "length"
            call["completed"] = call.get("done") is True and call.get("done_reason") == "stop"
            if not call["completed"]:
                call["error"] = {"type": "ValueError", "message": "Model output was truncated" if call["truncated"]
                                 else "Model response lacks a complete stop"}
            return response  # Production nodes enforce their own finish/JSON guards.
        except Exception as exc:
            call["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            call["elapsed_seconds"] = time.perf_counter() - begin
            emit({"event": "model_call_finished", "call": call})

    def traced(name, function):
        def invoke(state):
            nonlocal last_state, raw_initial_draft, initial_draft
            current.update(stage=name, **{"pass": state["revision_count"]})
            if state["revision_count"] > (0 if strategy == "A" else 1):
                raise ValueError("Strategy revision limit exceeded")
            entry = {**current, "input": copy.deepcopy(state), "output": None,
                     "completed": False, "error": None, "elapsed_seconds": None}
            stages.append(entry)
            emit({"event": "stage_started", "stage": entry})
            begin = time.perf_counter()
            try:
                update = function(state)
                if name == "Writer" and state["revision_count"] == 0 and not update.get("generation_failed"):
                    raw_initial_draft = update["draft"]
                    update = {**update, "draft": apply_fault(update["draft"], case["fault"])}
                    initial_draft = update["draft"]
                    fault = {"pass": 0, "fault": copy.deepcopy(case["fault"]),
                             "raw_draft": raw_initial_draft, "injected_draft": initial_draft,
                             "applied": case["fault"]["mode"] != "none"}
                    faults.append(fault)
                    emit({"event": "initial_fault_fixture", "fault": fault})
                entry["output"] = copy.deepcopy(update)
                entry["completed"] = True
                last_state = {**copy.deepcopy(state), **copy.deepcopy(update)}
                if name == "Verifier":
                    details = update.get("checker_details", {})
                    claims = copy.deepcopy(details.get("claims", []))
                    checks.append({"phase": strategy, "pass": state["revision_count"], "draft": state["draft"],
                                   "status": update["status"], "answer": update.get("answer", ""),
                                   "candidate_claims": [s["text"] for s in details.get("segments", []) if s["kind"] == "medical_claim"],
                                   "claims": claims, "checker_details": copy.deepcopy(details),
                                   "unchecked_candidate_claims": copy.deepcopy(details.get("unchecked_claims", [])),
                                   "failed_candidate_claims": [item["claim"] for item in claims if item.get("error")],
                                   "issues": list(update.get("issues", [])), "completed": details.get("complete") is True})
                return update
            except Exception as exc:
                entry["error"] = {"type": type(exc).__name__, "message": str(exc)}
                last_state = copy.deepcopy(state)
                raise
            finally:
                entry["elapsed_seconds"] = time.perf_counter() - begin
                emit({"event": "stage_finished", "stage": entry})
        return invoke

    def policy_checker_route(state):
        destination = checker_route(state)
        return "stop" if strategy == "A" and destination == "reflect" else destination

    graph = StateGraph(agent.HealthState)
    names = {"Retriever": "retriever", "Writer": "writer", "Verifier": "checker",
             "Reflection": "reflection", "Stop": "stop"}
    for name, function in original_nodes.items():
        graph.add_node(names[name], traced(name, function))
    graph.add_edge(START, "retriever")
    graph.add_conditional_edges("retriever", retriever_route, {"write": "writer", "finish": END})
    graph.add_conditional_edges("writer", writer_route, {"stop": "stop", "check": "checker"})
    graph.add_conditional_edges("checker", policy_checker_route,
                                {"finish": END, "reflect": "reflection", "stop": "stop"})
    graph.add_edge("reflection", "writer")
    graph.add_edge("stop", END)
    compiled_graph = graph.compile()
    emit({"event": "task_started", "production_config": config, "max_revisions": 0 if strategy == "A" else 1})
    with ExitStack() as stack:
        stack.enter_context(patch.object(agent, "knowledge_base", load_knowledge_base()))
        stack.enter_context(patch.object(agent, "chat", observed_chat))
        stack.enter_context(patch.object(claim_verifier, "chat", observed_chat))
        stack.enter_context(patch.dict(os.environ, {"INJECT_BAD_FACT": "0", "INJECT_FALSE_CLAIM": "0"}))
        stack.enter_context(redirect_stdout(logs))
        try:
            last_state = compiled_graph.invoke(_initial_state(case["task"]), config={"recursion_limit": 20})
        except Exception as exc:
            thrown_error = {"type": type(exc).__name__, "message": str(exc)}
            # An evaluation infrastructure failure is recorded separately while
            # the public-style result remains a production-safe terminal state.
            last_state.update(traced("Stop", agent.stop_node)(last_state))
    total_seconds = time.perf_counter() - started
    error = thrown_error or _terminal_error(last_state)
    outcomes = _gate_outcomes(last_state, checks, error)
    if last_state["status"] == "completed":
        stop_reason = "checker_passed"
    elif last_state["status"] in ("no_evidence", "insufficient_content"):
        stop_reason = last_state["status"]
    elif last_state.get("generation_failed"):
        stop_reason = "writer_generation_failed"
    elif error:
        stop_reason = "infrastructure_error"
    elif last_state.get("checker_details", {}).get("complete") is False:
        stop_reason = "checker_incomplete"
    else:
        stop_reason = "reflection_disabled" if strategy == "A" else "revision_limit_1"
    detected = list(dict.fromkeys(issue for check in checks for issue in check["issues"]))
    latest = checks[-1] if checks else None
    current_check = bool(latest and latest["pass"] == last_state["revision_count"])
    stage_seconds = {stage: sum(item["elapsed_seconds"] for item in stages if item["stage"] == stage) for stage in STAGES}
    row = {"case_id": case["id"], "case": copy.deepcopy(case), "strategy": strategy,
           "execution_mode": execution_mode, "design": DESIGN, "end_to_end_invocation_count": 1,
           "entrypoint": "evaluation_graph_reusing_production_nodes_and_routes",
           "public_production_entrypoint_called": False,
           "status": last_state["status"], "answer": last_state["answer"], "draft": last_state["draft"],
           "initial_draft": initial_draft, "raw_initial_draft": raw_initial_draft,
           "evidence": copy.deepcopy(last_state["evidence"]),
           "input_evidence_sha256": digest(last_state["evidence"]), "initial_draft_sha256": digest(initial_draft),
           "detected_errors": detected, "final_errors": list(last_state["issues"]),
           "final_checker_completed": bool(current_check and latest["completed"] and error is None),
           "final_task_parameter_fidelity": (copy.deepcopy(last_state.get("checker_details", {}).get("task_parameter_fidelity"))
                                             if current_check else None),
           "gate_outcomes": outcomes, "publication_integrity_passed": outcomes["publication_integrity_passed"],
           "revision_count": last_state["revision_count"], "reflection_runs": sum(s["stage"] == "Reflection" for s in stages),
           "reflection_repair_succeeded": bool(strategy == "B" and last_state["revision_count"] == 1
                                                and last_state["status"] == "completed" and outcomes["publication_integrity_passed"]),
           "stop_reason": stop_reason, "model_calls": calls, "model_call_count": len(calls),
           "model_call_ids": [call["id"] for call in calls], "shared_model_call_count": 0,
           "exclusive_model_call_count": len(calls), "writer_model_call_count": sum(c["stage"] == "Writer" for c in calls),
           "verifier_model_call_count": sum(c["stage"] == "Verifier" for c in calls),
           "stage_seconds": stage_seconds, "stop_seconds": sum(s["elapsed_seconds"] for s in stages if s["stage"] == "Stop"),
           "reflection_cycle_seconds": sum(s["elapsed_seconds"] for s in stages if s["pass"] == 1 and s["stage"] in ("Writer", "Verifier", "Reflection")),
           "total_seconds": total_seconds, "shared_prefix_seconds": 0, "exclusive_tail_seconds": total_seconds,
           "no_evidence_safe_stop": (last_state["status"] == "no_evidence" and not calls and not last_state["draft"])
                                    if not last_state["evidence"] else None,
           "check_history": checks, "stages": stages, "error": error,
           "writer_runs": copy.deepcopy(last_state.get("writer_runs", [])), "fault_applications": faults,
           "production_config": config, "runtime_source_proof": proofs, "runtime_source_sha256": source_hashes,
           "production_route_identity_verified": True, "outer_retries": 0, "node_log": logs.getvalue(),
           "quality_status": "pending_independent_human_review", "medical_correctness": None,
           "task_fulfillment": None, "human_review_completed": False}
    emit({"event": "task_finished", "row": row})
    return row


def run_pair(case, chat_fn, execution_mode="test_double", event_sink=None):
    """Compatibility wrapper: two independent end-to-end runs, no shared calls."""
    begin = time.perf_counter()
    rows = {strategy: run_case(case, strategy, chat_fn, execution_mode, event_sink) for strategy in ("A", "B")}
    calls = [call for row in rows.values() for call in row["model_calls"]]
    stages = [stage for row in rows.values() for stage in row["stages"]]
    result = {"case_id": case["id"], "case": copy.deepcopy(case), "execution_mode": execution_mode,
              "design": DESIGN, "rows": rows, "model_calls": calls, "stages": stages,
              "physical_model_call_count": len(calls), "completed_model_call_count": sum(c["completed"] for c in calls),
              "shared_model_call_ids": [], "raw_initial_draft": rows["A"]["raw_initial_draft"],
              "raw_initial_draft_sha256": digest(rows["A"]["raw_initial_draft"]), "fault": copy.deepcopy(case["fault"]),
              "shared_initial_check_sha256": None, "physical_wall_seconds": time.perf_counter() - begin,
              "initial_drafts_equal": rows["A"]["initial_draft"] == rows["B"]["initial_draft"],
              "quality_status": "pending_independent_human_review"}
    if event_sink:
        event_sink({"case_id": case["id"], "event": "pair_finished", "physical_model_call_count": len(calls)})
    return result
