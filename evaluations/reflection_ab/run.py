"""Four fixed tasks, eight independent production-aligned OFF/ON runs."""

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile

from app import writer
from app.evidence_checker import MAX_MEDICAL_CLAIMS
from .core import CASES_PATH, MODEL, SAFE_STOP, STAGES, digest, load_cases, production_config


REPO = Path(__file__).resolve().parents[2]
EVALUATIONS = REPO / "evaluations"
HOST = "http://127.0.0.1:11434"
MODEL_MANIFEST = Path.home() / ".ollama/models/manifests/registry.ollama.ai/library/qwen3/4b"
DESIGN = "independent_end_to_end_paired"
QUALITY_WARNING = ("待独立人工审核；completed只表示当前程序发布门禁通过，不等于医学正确或完整满足任务。"
                   f"每轮最多核查{MAX_MEDICAL_CLAIMS}个医学单元，任务参数保真另行检查。")
SAFE_STATUSES = {"evaluation_failed", "insufficient_content", "no_evidence"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_timeouts(timeout_seconds, request_timeout_seconds):
    require(isinstance(timeout_seconds, (float, int)) and math.isfinite(timeout_seconds)
            and 0 < timeout_seconds <= 300, "Task timeout must be positive, finite and at most 300 seconds")
    require(request_timeout_seconds == writer.REQUEST_TIMEOUT_SECONDS == 45,
            "Request timeout must match production: exactly 45 seconds; no evaluation override")


def execution_plan(cases):
    return [{"index": index * 2 + offset + 1, "case_id": case["id"], "strategy": strategy}
            for index, case in enumerate(cases)
            for offset, strategy in enumerate(("A", "B") if index % 2 == 0 else ("B", "A"))]


def invoke_worker(case, strategy, event_path, request_timeout_seconds, timeout_seconds):
    request = {"case": case, "strategy": strategy, "event_path": str(Path(event_path).resolve()),
               "request_timeout_seconds": request_timeout_seconds}
    completed = subprocess.run([sys.executable, "-m", "evaluations.reflection_ab.worker"], cwd=REPO,
                               input=json.dumps(request, ensure_ascii=False), capture_output=True, text=True,
                               timeout=timeout_seconds, check=True)
    return json.loads(completed.stdout)


def read_events(path):
    result = []
    if Path(path).is_file():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                result.append(event)
    return result


def recover_failed_run(case, strategy, event_path, elapsed, exc, execution_mode):
    calls, stages = {}, {}
    events = read_events(event_path)
    for event in reversed(events):
        if event.get("event") == "task_finished" and isinstance(event.get("row"), dict):
            try:
                recovered = validate_run(copy.deepcopy(event["row"]), case, strategy, execution_mode)
            except (ValueError, KeyError, TypeError):
                continue
            recovered["recovered_from_completed_journal"] = True
            recovered["parent_transport_error"] = {"type": type(exc).__name__, "message": str(exc)}
            return recovered
    for event in events:
        if event.get("event") in ("model_call_started", "model_call_finished"):
            call = event.get("call", {})
            if isinstance(call.get("id"), str):
                calls[call["id"]] = copy.deepcopy(call)
        if event.get("event") in ("stage_started", "stage_finished"):
            stage = event.get("stage", {})
            stages[(stage.get("stage"), stage.get("pass"))] = copy.deepcopy(stage)
    error = {"type": type(exc).__name__, "message": str(exc)}
    if isinstance(exc, subprocess.TimeoutExpired):
        error["message"] = "Independent task worker exceeded the predeclared timeout; not retried"
    if isinstance(exc, subprocess.CalledProcessError):
        error["stderr"] = (exc.stderr or "")[-4000:]
    call_list, stage_list = list(calls.values()), list(stages.values())
    times = {}
    for name in STAGES:
        selected = [stage for stage in stage_list if stage.get("stage") == name]
        times[name] = (None if any(stage.get("elapsed_seconds") is None for stage in selected)
                       else sum(stage["elapsed_seconds"] for stage in selected))
    return {"case_id": case["id"], "case": copy.deepcopy(case), "strategy": strategy,
            "execution_mode": execution_mode, "design": DESIGN, "status": "unknown", "answer": SAFE_STOP,
            "draft": None, "initial_draft": None, "raw_initial_draft": None, "evidence": None,
            "input_evidence_sha256": None, "initial_draft_sha256": None, "detected_errors": [],
            "final_errors": None, "revision_count": None, "stop_reason": "worker_result_unavailable",
            "model_call_count": None, "observed_started_model_call_count": len(call_list),
            "model_calls": call_list, "model_call_ids": list(calls), "stage_seconds": times, "stages": stage_list,
            "total_seconds": None, "parent_process_wall_seconds": elapsed, "check_history": [], "error": error,
            "final_checker_completed": False, "final_task_parameter_fidelity": None,
            "publication_integrity_passed": None, "no_evidence_safe_stop": None, "writer_runs": [],
            "fault": copy.deepcopy(case["fault"]), "quality_status": "pending_independent_human_review",
            "observation_note": "Journal calls are a lower bound; final worker/publication result remains unknown."}


def final_details(row):
    history = row.get("check_history") or []
    return (history[-1].get("checker_details", {}) if history
            and history[-1].get("pass") == row.get("revision_count") else {})


def medical_gate_passed(row):
    if "medical_evidence_passed" in row.get("gate_outcomes", {}):
        return row["gate_outcomes"]["medical_evidence_passed"]
    details = final_details(row)
    if not details:
        return None
    codes = details.get("issue_codes", [])
    medical_codes = {"unsupported_claim", "verifier_error", "claim_coverage_limit", "no_input_facts"}
    return (details.get("complete") is True
            and not any(code in medical_codes or str(code).startswith("quantity_") or code == "malformed_quantity" for code in codes)
            and all(not item.get("error") and not item.get("quantity_issues")
                    and item.get("verification", {}).get("verdict") == "supported" for item in details.get("claims", [])))


def parameter_gate_passed(row):
    if "task_parameter_fidelity_passed" in row.get("gate_outcomes", {}):
        return row["gate_outcomes"]["task_parameter_fidelity_passed"]
    value = row.get("final_task_parameter_fidelity") or final_details(row).get("task_parameter_fidelity")
    return value.get("passed") if isinstance(value, dict) else None


def publication_integrity(row):
    declared = row.get("gate_outcomes", {}).get("publication_integrity_passed")
    if row.get("status") != "completed":
        if row.get("status") == "unknown":
            return None
        if row.get("status") == "evaluation_failed":
            return bool(row.get("answer") == SAFE_STOP and declared is not False)
        if row.get("status") == "no_evidence":
            return bool(row.get("no_evidence_safe_stop") is True and declared is not False)
        return declared if declared is not None else bool(row.get("answer") and row.get("answer") != row.get("draft"))
    details = final_details(row)
    return bool(details.get("complete") and not details.get("issue_codes") and not details.get("unchecked_claims")
                and not row.get("final_errors") and medical_gate_passed(row) and parameter_gate_passed(row) is True
                and declared is not False)


def validate_run(result, case, strategy, execution_mode):
    require(isinstance(result, dict) and result.get("case_id") == case["id"] and result.get("strategy") == strategy,
            "Worker returned the wrong task/strategy")
    require(result.get("execution_mode") == execution_mode, "A test double must never masquerade as a real model")
    require(result.get("design") == DESIGN, "Expected independent end-to-end design")
    calls = result.get("model_calls")
    require(isinstance(calls, list), "Missing observed calls")
    maximum = (1 if strategy == "A" else 2) * (1 + MAX_MEDICAL_CLAIMS)
    require(type(result.get("model_call_count")) is int and result["model_call_count"] == len(calls) <= maximum,
            "Independent task call count exceeds observed bounded calls")
    ids = [call.get("id") for call in calls]
    require(all(isinstance(value, str) for value in ids) and len(set(ids)) == len(ids), "Call IDs must be unique")
    config = production_config()
    for call in calls:
        request = call.get("request", {})
        require(request.get("model") == MODEL and request.get("think") is False, "Model/think differs from production")
        key = "writer" if call.get("stage") == "Writer" else "verifier" if call.get("stage") == "Verifier" else None
        require(key is not None, "Unknown model stage")
        require(request.get("options") == config[key]["options"] and request.get("format") == config[key]["format"],
                f"{key} config differs from production")
    require(result.get("revision_count") in ((0,) if strategy == "A" else (0, 1)), "Reflection policy exceeded")
    require(result.get("status") in {"completed", "evaluation_failed", "insufficient_content", "no_evidence"},
            "Worker did not preserve a production terminal state")
    result["observed_started_model_call_count"] = len(calls)
    result["publication_integrity_passed"] = publication_integrity(result)
    result["medical_gate_passed"] = medical_gate_passed(result)
    result["task_parameter_gate_passed"] = parameter_gate_passed(result)
    return result


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def summarize(pairs):
    strategies = {}
    for strategy in ("A", "B"):
        rows = [pair["rows"][strategy] for pair in pairs]
        completed = [row for row in rows if row["status"] == "completed" and publication_integrity(row)]
        evidence_rows = [row for row in rows if row.get("case", {}).get("category") != "no_evidence"]
        known_times = [row["total_seconds"] for row in rows if row.get("total_seconds") is not None]
        safe = [row for row in rows if row["status"] in SAFE_STATUSES and publication_integrity(row) is True]
        no_evidence = [row for row in rows if row.get("case", {}).get("category") == "no_evidence"
                       and row.get("no_evidence_safe_stop") is True]
        gates = {}
        for label, fn in (("medical", medical_gate_passed), ("task_parameter", parameter_gate_passed)):
            values = [fn(row) for row in rows]
            gates[label] = {"passed": sum(value is True for value in values), "failed": sum(value is False for value in values),
                            "not_run_or_unknown": sum(value is None for value in values)}
        count = len(rows)
        strategies[strategy] = {
            "task_count": count, "checker_passed_count": len(completed), "completed_program_gate_count": len(completed),
            "completed_rate_all_tasks": len(completed) / count if count else None,
            "completed_rate_evidence_tasks": len(completed) / len(evidence_rows) if evidence_rows else None,
            "program_defined_task_success_count": len(completed) + len(no_evidence),
            "program_defined_task_success_rate": (len(completed) + len(no_evidence)) / count if count else None,
            "independent_human_task_success_count": None, "independent_medical_pass_count": None,
            "no_evidence_safe_stop_count": len(no_evidence), "safe_stop_count": len(safe),
            "safe_stop_rate": len(safe) / count if count else None,
            "unsafe_publication_detected_count": sum(publication_integrity(row) is False for row in rows),
            "publication_outcome_unknown_tasks": sum(row["status"] == "unknown" for row in rows),
            "failed_or_unknown_count": sum(row["status"] not in ("completed", "no_evidence") for row in rows),
            "revision_count": sum(row.get("revision_count") or 0 for row in rows),
            "reflection_attempt_tasks": sum((row.get("revision_count") or 0) > 0 for row in rows),
            "reflection_repair_success_count": sum(row in completed and row.get("revision_count") == 1 for row in rows),
            "revision_count_unknown_tasks": sum(row.get("revision_count") is None for row in rows),
            "attributed_model_call_count_known": sum(row.get("model_call_count") or 0 for row in rows),
            "model_call_count_unknown_tasks": sum(row.get("model_call_count") is None for row in rows),
            "writer_model_call_count_known": sum(call.get("stage") == "Writer" for row in rows for call in row.get("model_calls", [])),
            "verifier_model_call_count_known": sum(call.get("stage") == "Verifier" for row in rows for call in row.get("model_calls", [])),
            "attributed_total_seconds_known": sum(known_times), "time_unknown_tasks": count - len(known_times),
            "worker_wall_seconds_known": sum(row.get("parent_process_wall_seconds") or 0 for row in rows),
            "stage_seconds_known": {name: sum(row.get("stage_seconds", {}).get(name) or 0 for row in rows) for name in STAGES},
            "latency_sample_count": len(known_times), "latency_p50_seconds": percentile(known_times, .5),
            "latency_p95_seconds": percentile(known_times, .95), "gate_counts": gates,
            "detected_issue_count": sum(len(row.get("detected_errors") or []) for row in rows),
            "remaining_detected_issue_count_known": sum(len(row.get("final_errors") or []) for row in rows),
            "remaining_issue_count_unknown_tasks": sum(row.get("final_errors") is None for row in rows),
            "failure_reasons": [{"case_id": row["case_id"], "status": row["status"], "stop_reason": row.get("stop_reason"),
                                 "issues": row.get("final_errors"), "error": row.get("error")}
                                for row in rows if row["status"] not in ("completed", "no_evidence")],
        }
    rows = [row for pair in pairs for row in pair["rows"].values()]
    calls = {call["id"]: call for row in rows for call in row.get("model_calls", [])}
    a, b = strategies["A"], strategies["B"]
    extra_calls = None if a["model_call_count_unknown_tasks"] or b["model_call_count_unknown_tasks"] else b["attributed_model_call_count_known"] - a["attributed_model_call_count_known"]
    extra_seconds = None if a["time_unknown_tasks"] or b["time_unknown_tasks"] else b["attributed_total_seconds_known"] - a["attributed_total_seconds_known"]
    return {"strategies": strategies,
            "physical_model_call_count_known": sum(row.get("model_call_count") or 0 for row in rows),
            "physical_model_call_count_unknown_runs": sum(row.get("model_call_count") is None for row in rows),
            "physical_model_call_count_unknown_pairs": sum(any(row.get("model_call_count") is None for row in pair["rows"].values()) for pair in pairs),
            "observed_started_model_call_count_lower_bound": len(calls),
            "observed_completed_model_call_count": sum(call.get("completed") is True for call in calls.values()),
            "observed_response_received_count": sum(call.get("response_received") is True for call in calls.values()),
            "physical_wall_seconds": sum(row.get("parent_process_wall_seconds") or 0 for row in rows),
            "shared_calls_not_additive_across_branches": False,
            "ON_minus_OFF_completed_count": b["completed_program_gate_count"] - a["completed_program_gate_count"],
            "ON_extra_model_calls": extra_calls, "ON_extra_total_seconds": extra_seconds,
            "ON_extra_model_calls_percent": 100 * extra_calls / a["attributed_model_call_count_known"] if extra_calls is not None and a["attributed_model_call_count_known"] else None,
            "ON_extra_total_seconds_percent": 100 * extra_seconds / a["attributed_total_seconds_known"] if extra_seconds is not None and a["attributed_total_seconds_known"] else None,
            "percentile_method": "linear interpolation at (n-1)*p; n=4 per group including no-evidence; descriptive only"}


def run_suite(cases, timeout_seconds=300, request_timeout_seconds=45, invoke=invoke_worker, on_result=None, event_dir=None):
    validate_timeouts(timeout_seconds, request_timeout_seconds)
    suite = load_cases()
    require(digest(cases) == digest(suite["cases"]), "Only the four fixed unchanged cases may be run")
    mode = "real_model" if invoke is invoke_worker else "test_double"
    plan, by_id, rows = execution_plan(cases), {case["id"]: case for case in cases}, []
    started = datetime.now(timezone.utc).isoformat()
    with tempfile.TemporaryDirectory(prefix="final-reflection-ab-") as temporary:
        journals = Path(event_dir or temporary)
        journals.mkdir(parents=True, exist_ok=True)
        for entry in plan:
            case, strategy = by_id[entry["case_id"]], entry["strategy"]
            path = journals / f"{case['id']}.{strategy}.jsonl"
            require(not path.exists(), "Refusing to overwrite an independent run journal")
            begin = time.perf_counter()
            try:
                row = validate_run(invoke(case, strategy, path, request_timeout_seconds, timeout_seconds), case, strategy, mode)
            except Exception as exc:
                row = recover_failed_run(case, strategy, path, time.perf_counter() - begin, exc, mode)
            row.update(parent_process_wall_seconds=time.perf_counter() - begin, event_log=str(path), execution_index=entry["index"])
            rows.append(row)
            if on_result:
                on_result(row)
    pairs = []
    for case in cases:
        paired = {row["strategy"]: row for row in rows if row["case_id"] == case["id"]}
        calls = [call for row in paired.values() for call in row.get("model_calls", [])]
        pairs.append({"case_id": case["id"], "case": case, "fault": case["fault"], "rows": paired, "model_calls": calls,
                      "execution_mode": mode, "design": DESIGN,
                      "observed_started_model_call_count": len(calls),
                      "unknown_model_call_count": any(row.get("model_call_count") is None for row in paired.values()),
                      "physical_model_call_count": len(calls) if all(row.get("model_call_count") is not None for row in paired.values()) else None,
                      "physical_wall_seconds": sum(row["parent_process_wall_seconds"] for row in paired.values()),
                      "initial_drafts_equal": (paired["A"].get("initial_draft_sha256") == paired["B"].get("initial_draft_sha256"))
                      if paired["A"].get("initial_draft_sha256") and paired["B"].get("initial_draft_sha256") else None})
    return {"schema_version": "2.0", "execution_mode": mode, "design": DESIGN, "suite_id": suite["suite_id"],
            "status": "small_sample_descriptive", "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "case_count": 4, "branch_count": 8, "end_to_end_run_count": 8, "execution_plan": plan,
            "production_configuration": production_config(), "timeout_seconds_per_run": timeout_seconds,
            "request_timeout_seconds": request_timeout_seconds, "physical_model_call_upper_bound": 9 * (1 + MAX_MEDICAL_CLAIMS),
            "rows": rows, "pairs": pairs, "summary": summarize(pairs), "quality_status": "pending_independent_human_review",
            "quality_warning": QUALITY_WARNING,
            "interpretation": "Four paired tasks; OFF and ON execute independently once. No prefix/call sharing; no retries."}


def source_snapshot():
    paths = set((REPO / "app").glob("*.py"))
    paths.update(Path(__file__).parent.glob("*.py"))
    paths.update((EVALUATIONS / "tests").glob("test*reflection*.py"))
    paths.update((EVALUATIONS / "tests").glob("test_final_ab*.py"))
    paths.update((EVALUATIONS / "frozen").glob("*.json"))
    paths.update((Path(__file__).parent / "README.md", CASES_PATH))
    for name in ("data/health_knowledge.json", "evaluations/review_checklist_ai_suggestions.csv",
                 "evaluations/annotation_guidelines.md", "evaluations/claims.v1.json", "evaluations/claims.holdout.v2.json", "evaluations/review_checklist.csv"):
        paths.add(REPO / name)
    return {str(path.relative_to(REPO)): sha(path) for path in sorted(paths)}


def git_snapshot():
    def git(*arguments):
        return subprocess.check_output(["git", *arguments], cwd=REPO, text=True, timeout=10).rstrip()
    return {"head": git("rev-parse", "HEAD"), "status_porcelain": git("status", "--porcelain=v1", "--untracked-files=all"), "automatic_commit": False}


def preflight_model(request_timeout_seconds):
    from ollama import Client
    metadata = Client(host=HOST, timeout=request_timeout_seconds)
    models = [item.model_dump(mode="json") for item in metadata.list().models if getattr(item, "model", None) == MODEL]
    require(models, "Local qwen3:4b unavailable; inference not started")
    require(MODEL_MANIFEST.is_file(), "Local manifest unavailable")
    return {"name": MODEL, "host": HOST, "sdk_version": version("ollama"), "manifest_path": str(MODEL_MANIFEST),
            "manifest_sha256": sha(MODEL_MANIFEST), "list_metadata": models,
            "show_metadata": metadata.show(MODEL).model_dump(mode="json"), "metadata_only_probe": True}


def reverify_freeze(freeze):
    changed = [path for group in ("source_sha256", "historical_result_sha256", "label_sha256")
               for path, expected in freeze.get(group, {}).items()
               if not (REPO / path).is_file() or sha(REPO / path) != expected]
    if not MODEL_MANIFEST.is_file() or sha(MODEL_MANIFEST) != freeze["model"]["manifest_sha256"]:
        changed.append("local_model_manifest")
    return {"passed": not changed, "changed_paths": changed}


def display(value):
    if value is None:
        return "未执行/未知"
    if isinstance(value, bool):
        return "通过" if value else "未通过"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def comparison_records(report):
    return [{"case_id": row["case_id"], "strategy": row["strategy"], "status": row["status"],
             "program_checks_passed": row["status"] == "completed" and publication_integrity(row) is True,
             "medical_gate_passed": medical_gate_passed(row), "task_parameter_gate_passed": parameter_gate_passed(row),
             "reflection_count": row.get("revision_count"), "model_call_count": row.get("model_call_count"),
             "Writer_seconds": row.get("stage_seconds", {}).get("Writer"), "Verifier_seconds": row.get("stage_seconds", {}).get("Verifier"),
             "total_seconds": row.get("total_seconds"), "stop_reason": row.get("stop_reason"), "final_errors": row.get("final_errors"),
             "answer": row.get("answer"), "independent_medical_review_passed": None}
            for pair in report["pairs"] for row in (pair["rows"]["A"], pair["rows"]["B"])]


def render_comparison(report):
    lines = ["|任务|策略|状态|医学证据门禁|任务参数保真|修正|调用|Writer秒|Verifier秒|总秒|", "|---|---|---|---|---|---:|---:|---:|---:|---:|"]
    keys = ("case_id", "strategy", "status", "medical_gate_passed", "task_parameter_gate_passed", "reflection_count", "model_call_count", "Writer_seconds", "Verifier_seconds", "total_seconds")
    lines.extend("|" + "|".join(display(row[key]) for key in keys) + "|" for row in comparison_records(report))
    return "\n".join(lines) + "\n"


def render_report(report):
    summary = report["summary"]
    a, b = summary["strategies"]["A"], summary["strategies"]["B"]
    delta = b["completed_program_gate_count"] - a["completed_program_gate_count"]
    conclusion = "ON增加程序完成任务数" if delta > 0 else "ON未增加程序完成任务数" if delta == 0 else "ON程序完成任务数更少"
    lines = [f"本样本中{conclusion}：OFF {a['completed_program_gate_count']}/4，ON {b['completed_program_gate_count']}/4。"
             f"额外模型调用 {display(summary['ON_extra_model_calls'])} 次（{display(summary['ON_extra_model_calls_percent'])}%），"
             f"已知任务耗时增加 {display(summary['ON_extra_total_seconds'])} 秒（{display(summary['ON_extra_total_seconds_percent'])}%）。", "",
             f"模式 `{report['execution_mode']}`；设计 `{DESIGN}`；四场景、8次独立端到端运行。"
             "A保留全部检查并关闭重写，B最多一次重写；评测图复用同一生产节点与真实路由，未直接调用公共run_health_agent函数。"
             "任务、知识库、Prompt、JSON协议、参数和发布门禁相同。", "",
             "|指标|OFF|ON|", "|---|---:|---:|"]
    metrics = (("completed门禁通过数（分母4）", "completed_program_gate_count"), ("completed率（所有任务）", "completed_rate_all_tasks"),
               ("completed率（有证据任务，分母3）", "completed_rate_evidence_tasks"),
               ("程序定义任务成功率（含R-NONE正确安全停）", "program_defined_task_success_rate"),
               ("安全停止数", "safe_stop_count"), ("安全停止率", "safe_stop_rate"),
               ("可观测门禁错误发布数", "unsafe_publication_detected_count"), ("发布结果未知任务数", "publication_outcome_unknown_tasks"),
               ("Reflection尝试任务数", "reflection_attempt_tasks"), ("Reflection修正后通过数", "reflection_repair_success_count"),
               ("模型调用（已知）", "attributed_model_call_count_known"), ("Writer调用", "writer_model_call_count_known"),
               ("Verifier调用", "verifier_model_call_count_known"), ("任务耗时合计秒", "attributed_total_seconds_known"),
               ("父进程实测运行墙钟秒（含子进程开销）", "worker_wall_seconds_known"),
               ("任务耗时P50秒", "latency_p50_seconds"), ("任务耗时P95秒", "latency_p95_seconds"))
    lines.extend(f"|{label}|{display(a[key])}|{display(b[key])}|" for label, key in metrics)
    lines.extend(f"|{stage}耗时合计秒|{display(a['stage_seconds_known'][stage])}|{display(b['stage_seconds_known'][stage])}|" for stage in STAGES)
    for label, key in (("医学证据门禁", "medical"), ("任务参数门禁", "task_parameter")):
        values = [" / ".join(str(metrics["gate_counts"][key][state]) for state in ("passed", "failed", "not_run_or_unknown"))
                  for metrics in (a, b)]
        lines.append(f"|{label}：通过 / 失败 / 未执行或未知|{values[0]}|{values[1]}|")
    lines += ["", "completed仅为程序发布门禁通过。程序定义任务成功另把R-NONE正确无证据停止计入；独立任务质量和医学审核均未完成，成功率未知。", "", render_comparison(report)]
    for pair in report["pairs"]:
        lines += ["", f"**{pair['case_id']}**：{pair['case']['task']}", "",
                  f"两次独立首稿是否逐字一致：{display(pair.get('initial_drafts_equal'))}。"]
        for strategy in ("A", "B"):
            row = pair["rows"][strategy]
            lines += ["", f"**{strategy} / {row['status']}**", "", row.get("answer") or "未取得完整回答", "",
                      f"停止原因：{row.get('stop_reason')}；累计问题：" + ("；".join(row.get("detected_errors") or []) or "未检出/未执行"),
                      "末轮问题：" + ("未知" if row.get("final_errors") is None else "；".join(row["final_errors"]) or "未检出")]
            if row.get("error"):
                lines.append(json.dumps(row["error"], ensure_ascii=False))
    lines += ["", "每组仅4项（含1项零模型调用），P50/P95按(n-1)*p线性插值，只作小样本描述。无显著性结论，不能推断总体成功率或医学正确率。",
              "顺序预先交替AB/BA；独立生成不保证逐字一致，缓存、系统负载及顺序仍可能影响耗时。",
              "错误发布数只表示当前门禁的可观测错误，不证明不存在未检出的医学/语义错误。",
              "超时/worker失败保留journal观测下界，未知调用或结果不伪记零；无重跑。Reflection重写耗时计入Writer，复核计入Verifier，不能重复相加。",
              QUALITY_WARNING, "", "完整请求、响应、每轮草稿和门禁见report.json；配置/版本、人工审核空表及剩余交付见同目录文件。旧实验未覆盖。", ""]
    return "\n".join(lines)


def build_human_review(report):
    records = []
    for pair in report["pairs"]:
        for strategy in ("A", "B"):
            row = pair["rows"][strategy]
            records.append({"case_id": pair["case_id"], "strategy": strategy, "task": pair["case"]["task"], "status": row["status"],
                            "answer": row.get("answer"), "input_evidence": row.get("evidence"), "raw_initial_draft": row.get("raw_initial_draft"),
                            "injected_initial_draft": row.get("initial_draft"), "final_candidate_draft": row.get("draft"), "fault": pair["fault"],
                            "check_history": row.get("check_history", []), "detected_errors": row.get("detected_errors"),
                            "final_detected_errors": row.get("final_errors"), "medical_quality": None, "task_fulfillment": None,
                            "evidence_faithfulness": None, "human_review_notes": None, "reviewer": None, "reviewed_on": None,
                            "review_status": "pending_independent_human_review", "coverage_warning": QUALITY_WARNING})
    return {"schema_version": "2.0", "execution_mode": report["execution_mode"], "human_review_completed": False, "row_count": len(records), "rows": records}


def render_human_review(review):
    lines = ["以下记录尚未独立人工审核，医学质量、任务满足度和证据忠实度均未评分。", "",
             "|任务|策略|状态|医学质量|任务满足度|证据忠实度|", "|---|---|---|---|---|---|"]
    lines.extend(f"|{row['case_id']}|{row['strategy']}|{row['status']}|待审核|待审核|待审核|" for row in review["rows"])
    return "\n".join(lines) + "\n"


def render_configuration(report):
    freeze = report.get("freeze", {})
    return ("两个策略共用生产配置，只有是否允许自动重写不同。\n\n"
            f"设计：{DESIGN}。Git HEAD：{freeze.get('git_before', {}).get('head', '测试替身未冻结')}。\n\n"
            "```json\n" + json.dumps(report["production_configuration"], ensure_ascii=False, indent=2) + "\n```\n\n"
            "预先冻结顺序：\n\n```json\n" + json.dumps(report["execution_plan"], ensure_ascii=False, indent=2) + "\n```\n\n"
            "源码、知识库、Prompt、JSON schema、模型manifest与历史哈希见freeze.json；实际加载证明见report.json。"
            "工作区已有未提交内容，Git HEAD不能单独唯一标识实验，另保存完整源码快照。\n")


def render_checklist(report):
    return ("本轮已交付8次独立运行、原始记录、逐任务对照、配置版本快照和人工审核空表。\n\n剩余交付：\n\n"
            "- 人工逐条评估8条输出的医学准确性、证据忠实度与任务满足度，填写human_review.json；当前均null。\n"
            "- 对32条AI建议做独立人工裁定，尤其U006争议，不能冒充人工标签。\n"
            "- 如实保留Writer数字失真、Reflection未修复、误拦与门禁语义范围限制；本轮不再调Prompt。\n"
            "- 将冻结版本和实验结论整理进项目说明；Git提交由用户另行决定。\n"
            "- 更大样本、多种子、独立医学审核后的评价需另行定义方案和预算，本轮不自动执行。\n")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_outputs(output, report):
    review = build_human_review(report)
    write_json(output / "report.json", report)
    (output / "report.md").write_text(render_report(report), encoding="utf-8")
    (output / "task_comparison.md").write_text(render_comparison(report), encoding="utf-8")
    records = comparison_records(report)
    with (output / "task_comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        csv_writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        csv_writer.writeheader()
        for record in records:
            csv_writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value for key, value in record.items()})
    write_json(output / "human_review.json", review)
    (output / "human_review.md").write_text(render_human_review(review), encoding="utf-8")
    (output / "configuration_and_version.md").write_text(render_configuration(report), encoding="utf-8")
    (output / "remaining_delivery_checklist.md").write_text(render_checklist(report), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--run-small", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=300)
    parser.add_argument("--request-timeout-seconds", type=float, default=writer.REQUEST_TIMEOUT_SECONDS)
    parser.add_argument("--output-dir", type=Path, default=EVALUATIONS / "results")
    parser.add_argument("--pytest-junit", type=Path)
    args = parser.parse_args(argv)
    try:
        validate_timeouts(args.timeout_seconds, args.request_timeout_seconds)
        suite, config = load_cases(), production_config()
        require(config["writer"]["options"] == {"temperature": 0, "seed": 42, "num_predict": 512, "num_ctx": 4096}, "Writer differs from authorized baseline")
        require(config["writer"]["model"] == "qwen3:4b" and config["writer"]["protocol"] == "json_article_v1", "Unexpected production baseline")
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    if args.validate_only:
        print(json.dumps({"validation": "passed", "design": DESIGN, "case_count": 4, "end_to_end_run_count": 8,
                          "execution_plan": execution_plan(suite["cases"]), "production_configuration": config,
                          "physical_model_call_upper_bound": 9 * (1 + MAX_MEDICAL_CLAIMS), "model_calls": 0, "network_calls": 0}, ensure_ascii=False, indent=2))
        return 0
    pytest_evidence = None
    if args.pytest_junit:
        import xml.etree.ElementTree as ET
        junit = args.pytest_junit.resolve()
        tests = list(ET.parse(junit).getroot().iter("testsuite"))
        require(tests and all(int(s.attrib.get("failures", 0)) == int(s.attrib.get("errors", 0)) == 0 for s in tests), "Offline pytest evidence contains failure/error")
        pytest_evidence = {"path": str(junit), "sha256": sha(junit), "tests": sum(int(s.attrib["tests"]) for s in tests)}
    sources = source_snapshot()
    history = {str(path.relative_to(REPO)): sha(path) for path in sorted((EVALUATIONS / "results").rglob("*")) if path.is_file()}
    labels = {str(path.relative_to(REPO)): sha(path) for path in sorted(EVALUATIONS.glob("*.csv"))}
    model = preflight_model(args.request_timeout_seconds)
    output = args.output_dir.resolve() / ("reflection-final-ab-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    freeze = {"schema_version": "2.0", "suite_id": suite["suite_id"], "execution_mode": "real_model", "design": DESIGN,
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "execution_plan": execution_plan(suite["cases"]),
              "source_sha256": sources, "historical_result_sha256": history, "label_sha256": labels,
              "production_configuration": config, "model": model, "git_before": git_snapshot(), "python_version": sys.version,
              "offline_pytest_evidence": pytest_evidence, "request_timeout_seconds": args.request_timeout_seconds,
              "timeout_seconds_per_run": args.timeout_seconds, "max_end_to_end_runs": 8, "outer_retries": 0,
              "physical_model_call_upper_bound": 9 * (1 + MAX_MEDICAL_CLAIMS)}
    write_json(output / "freeze.json", freeze)
    with zipfile.ZipFile(output / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for path, expected in sources.items():
            require(sha(REPO / path) == expected, "Source changed before inference: " + path)
            archive.write(REPO / path, path)
    with (output / "progress.jsonl").open("x", encoding="utf-8") as stream:
        def record(row):
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"{row['execution_index']}/8 {row['case_id']} {row['strategy']}: {row['status']}; calls={row.get('model_call_count')}; reflection={row.get('revision_count')}; seconds={row.get('total_seconds')}", flush=True)
        print(f"REAL: exactly 8 independent end-to-end tasks, no retries; output={output}", flush=True)
        report = run_suite(suite["cases"], args.timeout_seconds, args.request_timeout_seconds, on_result=record, event_dir=output / "events")
    report.update(freeze=freeze, freeze_reverification=reverify_freeze(freeze), git_after=git_snapshot())
    write_outputs(output, report)
    print("Report: " + str(output / "report.md"), flush=True)
    print(json.dumps(report["summary"], ensure_ascii=False), flush=True)
    return 1 if not report["freeze_reverification"]["passed"] or any(row["status"] == "unknown" for row in report["rows"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
