"""Offline checks for fixed independent runs, recovery and honest metrics."""

import copy
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from app import writer
from app.knowledge import load_knowledge_base
from evaluations.reflection_ab import core, run


GOOD = "".join(next(doc for doc in load_knowledge_base() if doc["id"] == "CDC001")["facts"]) + "[1]"
LIMIT = "资料未提及睡前拉伸20分钟的效果，因此无法确认睡前拉伸20分钟能否让所有人的入睡时间减半。[1]"


class Fixture:
    def __init__(self, case):
        self.case = case
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        if kwargs["format"] == writer.OUTPUT_SCHEMA:
            body = LIMIT if self.case["id"] == "R-INSUFFICIENT" else GOOD
            content = json.dumps({"article": body}, ensure_ascii=False)
        else:
            bad = "牛奶" in kwargs["messages"][-1]["content"]
            content = json.dumps({"verdict": "not_supported" if bad else "supported",
                                  "reason": "offline fixture", "evidence_id": 0 if bad else 1})
        return SimpleNamespace(message=SimpleNamespace(content=content), done=True,
                               done_reason="stop", eval_count=1, prompt_eval_count=1)


def invoke_fixture(case, strategy, event_path, request_timeout_seconds, timeout_seconds):
    assert request_timeout_seconds == 45
    with Path(event_path).open("x") as stream:
        def emit(event):
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        return core.run_case(case, strategy, Fixture(case), event_sink=emit)


def suite(tmp_path, invoke=invoke_fixture):
    return run.run_suite(core.load_cases()["cases"], invoke=invoke, event_dir=tmp_path / "events")


def test_eight_independent_runs_use_a_fixed_alternating_plan_and_additive_costs(tmp_path):
    report = suite(tmp_path)
    assert report["design"] == "independent_end_to_end_paired"
    assert report["end_to_end_run_count"] == len(report["rows"]) == 8
    assert [row["execution_index"] for row in report["rows"]] == list(range(1, 9))
    assert [row["strategy"] for row in report["rows"]] == ["A", "B", "B", "A", "A", "B", "B", "A"]
    assert len({(row["case_id"], row["strategy"]) for row in report["rows"]}) == 8
    assert len(list((tmp_path / "events").glob("*.jsonl"))) == 8
    assert report["summary"]["physical_model_call_count_known"] == sum(row["model_call_count"] for row in report["rows"])
    assert report["summary"]["shared_calls_not_additive_across_branches"] is False
    assert report["summary"]["strategies"]["A"]["completed_program_gate_count"] == 2
    assert report["summary"]["strategies"]["B"]["completed_program_gate_count"] == 3
    assert report["summary"]["strategies"]["A"]["program_defined_task_success_rate"] == .75
    assert report["summary"]["strategies"]["B"]["program_defined_task_success_rate"] == 1
    assert report["summary"]["strategies"]["B"]["reflection_repair_success_count"] == 1


def test_not_run_gates_and_human_quality_are_not_claimed_as_failures_or_successes(tmp_path):
    report = suite(tmp_path)
    for strategy in ("A", "B"):
        metrics = report["summary"]["strategies"][strategy]
        assert metrics["gate_counts"]["medical"]["not_run_or_unknown"] == 1
        assert metrics["gate_counts"]["task_parameter"]["not_run_or_unknown"] == 1
        assert metrics["independent_human_task_success_count"] is None
        assert metrics["independent_medical_pass_count"] is None
    review = run.build_human_review(report)
    assert review["row_count"] == 8 and review["human_review_completed"] is False
    assert all(row["medical_quality"] is row["task_fulfillment"] is None for row in review["rows"])


def test_request_settings_are_captured_from_production_without_verifier_override(tmp_path):
    report = suite(tmp_path)
    config = report["production_configuration"]
    assert config["writer"]["options"] == writer.OPTIONS
    assert config["verifier"]["options"] == {"temperature": 0}
    for row in report["rows"]:
        for call in row["model_calls"]:
            key = "writer" if call["stage"] == "Writer" else "verifier"
            assert call["request"]["options"] == config[key]["options"]
    with pytest.raises(ValueError, match="45"):
        run.run_suite(core.load_cases()["cases"], request_timeout_seconds=90, invoke=invoke_fixture)


def test_one_timeout_preserves_unknown_lower_bound_and_continues_exactly_seven_other_runs(tmp_path):
    invoked = []

    def fail_first(case, strategy, path, request_timeout_seconds, timeout_seconds):
        invoked.append((case["id"], strategy))
        if len(invoked) == 1:
            path.write_text(json.dumps({"event": "model_call_started", "call": {
                "id": case["id"] + "/" + strategy + "/1", "stage": "Writer", "completed": False,
                "elapsed_seconds": None}}) + '\n{"event":')
            raise subprocess.TimeoutExpired(["offline fixture"], timeout_seconds)
        return invoke_fixture(case, strategy, path, request_timeout_seconds, timeout_seconds)

    report = suite(tmp_path, fail_first)
    assert len(invoked) == len(set(invoked)) == 8
    first = report["rows"][0]
    assert first["status"] == "unknown" and first["model_call_count"] is None
    assert first["observed_started_model_call_count"] == 1
    assert first["publication_integrity_passed"] is None
    assert report["summary"]["physical_model_call_count_unknown_runs"] == 1
    assert report["summary"]["strategies"]["A"]["publication_outcome_unknown_tasks"] == 1
    assert all(row["status"] != "unknown" for row in report["rows"][1:])


def test_completed_journal_is_recovered_without_repeating_inference(tmp_path):
    calls = []

    def lose_stdout(case, strategy, path, request_timeout_seconds, timeout_seconds):
        calls.append((case["id"], strategy))
        invoke_fixture(case, strategy, path, request_timeout_seconds, timeout_seconds)
        raise RuntimeError("stdout lost after task_finished")

    report = suite(tmp_path, lose_stdout)
    assert len(calls) == 8
    assert all(row["status"] != "unknown" and row["recovered_from_completed_journal"] for row in report["rows"])
    assert all(row["parent_transport_error"]["type"] == "RuntimeError" for row in report["rows"])


def test_inconsistent_completed_is_counted_as_observable_bad_publication_not_hidden(tmp_path):
    def unsafe_fixture(case, strategy, path, request_timeout_seconds, timeout_seconds):
        row = invoke_fixture(case, strategy, path, request_timeout_seconds, timeout_seconds)
        if case["id"] == "R-NORMAL" and strategy == "A":
            row["final_errors"] = ["synthetic failed release gate"]
        return row

    report = suite(tmp_path, unsafe_fixture)
    assert report["rows"][0]["status"] == "completed"
    assert report["rows"][0]["publication_integrity_passed"] is False
    assert report["summary"]["strategies"]["A"]["unsafe_publication_detected_count"] == 1


def test_small_sample_percentiles_and_all_delivery_files_are_explicit(tmp_path):
    assert run.percentile([0, 10, 20, 30], .5) == 15
    assert run.percentile([0, 10, 20, 30], .95) == pytest.approx(28.5)
    assert run.percentile([], .5) is None
    report = suite(tmp_path)
    output = tmp_path / "delivery"
    output.mkdir()
    run.write_outputs(output, report)
    assert {path.name for path in output.iterdir()} == {
        "report.json", "report.md", "task_comparison.csv", "task_comparison.md", "human_review.json",
        "human_review.md", "configuration_and_version.md", "remaining_delivery_checklist.md"}
    text = (output / "report.md").read_text()
    assert "独立任务质量和医学审核均未完成" in text
    assert "只作小样本描述" in text
