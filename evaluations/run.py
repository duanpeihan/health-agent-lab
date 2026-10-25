"""Opt-in serial offline benchmark. Offline means local evidence, not a mock LLM."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

from .dataset import DEFAULT_DATASET, LABELS, ROOT, load_dataset, resolve_documents
from .metrics import summarize


def select_cases(dataset, mode):
    if mode not in ("smoke", "full"):
        raise ValueError("mode must be smoke or full")
    return dataset["cases"][:3] if mode == "smoke" else dataset["cases"]


def invoke_verifier(claim, documents, timeout_seconds):
    completed = subprocess.run(
        [sys.executable, "-m", "evaluations.worker"], cwd=ROOT,
        input=json.dumps({"claim": claim, "documents": documents}, ensure_ascii=False),
        capture_output=True, text=True, timeout=timeout_seconds, check=True,
    )
    return json.loads(completed.stdout)


def _source_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_state():
    try:
        def git(*args):
            return subprocess.check_output(["git", *args], cwd=ROOT, text=True, timeout=5).strip()
        return {"head": git("rev-parse", "HEAD"), "status": git("status", "--short")}
    except (OSError, subprocess.SubprocessError):
        return {"head": None, "status": "unavailable"}


def run_evaluation(dataset, mode, timeout_seconds, invoke=None, on_result=None):
    if timeout_seconds <= 0 or not float(timeout_seconds) < float("inf"):
        raise ValueError("timeout_seconds must be finite and positive")
    cases = select_cases(dataset, mode)
    invocation = invoke_verifier if invoke is None else invoke
    execution_mode = "real_model" if invoke is None else "test_double"
    started = datetime.now(timezone.utc).isoformat()
    run_start = time.perf_counter()
    results, observed_models, runtimes = [], set(), []
    for case in cases:
        documents = resolve_documents(case, dataset["_knowledge"])
        model_call_expected = any(doc["facts"] for doc in documents)
        case_start = time.perf_counter()
        verification, error, predicted = {}, None, "error"
        observed_calls, call_details = None, []
        try:
            response = invocation(case["claim"], documents, timeout_seconds)
            if not isinstance(response, dict) or not isinstance(response.get("verification"), dict):
                raise ValueError("Worker must return a verification object")
            if not isinstance(response.get("model"), str) or not response["model"]:
                raise ValueError("Worker must identify its model")
            if execution_mode == "real_model":
                candidate_count = response.get("model_call_count")
                candidate_details = response.get("model_calls")
                if (type(candidate_count) is not int or candidate_count != int(model_call_expected)
                        or not isinstance(candidate_details, list) or len(candidate_details) != candidate_count
                        or any(not isinstance(call, dict) or call.get("model") != response["model"]
                               or call.get("completed") is not True for call in candidate_details)):
                    raise ValueError("Worker model-call observation does not match selected evidence")
                observed_calls, call_details = candidate_count, candidate_details
                runtimes.append({"ollama_python_version": response.get("ollama_python_version"),
                                 "model_digest": response.get("model_digest")})
            observed_models.add(response["model"])
            verification = response["verification"]
            if verification.get("verdict") not in LABELS:
                raise ValueError("Verifier returned an invalid verdict")
            if not all(isinstance(verification.get(key), str) for key in ("reason", "evidence_quote")):
                raise ValueError("Verifier must return string reason/evidence_quote")
            predicted = verification["verdict"]
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
            if isinstance(exc, subprocess.TimeoutExpired):
                error["message"] = f"Verifier worker exceeded {timeout_seconds:g} seconds"
            elif isinstance(exc, subprocess.CalledProcessError):
                error["message"] = f"Verifier worker exited {exc.returncode}"
                error["stderr"] = (exc.stderr or "")[-4000:]
        row = {
            "id": case["id"], "claim": case["claim"], "expected_label": case["expected_label"],
            "predicted_label": predicted, "correct": predicted == case["expected_label"],
            "reason": verification.get("reason", ""), "evidence_quote": verification.get("evidence_quote", ""),
            "error": error, "elapsed_seconds": time.perf_counter() - case_start,
            "label_status": case["label_status"], "rationale": case["rationale"],
            "evidence_origin": case["evidence_origin"], "documents": documents,
            "expected_evidence_quotes": case["expected_evidence_quotes"],
            "model_call_expected": model_call_expected,
            "observed_model_call_count": observed_calls, "model_calls": call_details,
        }
        row["evidence_quote_matches_expected"] = (
            row["evidence_quote"] in case["expected_evidence_quotes"]
            if case["expected_evidence_quotes"] else row["evidence_quote"] == ""
        ) if error is None else False
        results.append(row)
        if on_result:
            on_result(row)
    metrics = summarize(results)
    metrics["by_evidence_origin"] = {
        origin: summarize([row for row in results if row["evidence_origin"] == origin])
        for origin in sorted({row["evidence_origin"] for row in results})
    }
    metrics["observed_model_cases"] = summarize([
        row for row in results if row["observed_model_call_count"] and row["error"] is None
    ])
    return {
        "schema_version": "1.0", "dataset_id": dataset["dataset_id"],
        "dataset_path": dataset["_path"], "dataset_sha256": dataset["_sha256"],
        "knowledge_sha256": dataset["knowledge_sha256"],
        "verifier_sha256": _source_hash(ROOT / "app" / "claim_verifier.py"),
        "harness_sha256": {name: _source_hash(Path(__file__).with_name(name))
                           for name in ("run.py", "worker.py", "dataset.py", "metrics.py")},
        "python_version": sys.version, "git": _git_state(),
        "mode": mode, "execution_mode": execution_mode, "provisional": True,
        "metric_interpretation": "Agreement with unreviewed seed labels; not established medical accuracy",
        "model": sorted(observed_models), "started_at_utc": started,
        "model_runtime_observations": runtimes,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - run_start, "timeout_seconds_per_case": timeout_seconds,
        "case_count": len(results),
        "model_requests_expected": sum(row["model_call_expected"] for row in results),
        "observed_model_call_count": sum(row["observed_model_call_count"] or 0 for row in results),
        "unknown_model_call_count_cases": sum(row["observed_model_call_count"] is None for row in results),
        "no_evidence_shortcut_count": sum(not row["model_call_expected"] for row in results),
        "metrics": metrics, "results": results,
        "errors": [row for row in results if not row["correct"] or not row["evidence_quote_matches_expected"]],
    }


def render_report(report):
    metrics = report["metrics"]
    lines = ["# Claim Verifier 离线评测", "",
             f"模式：`{report['execution_mode']}` / `{report['mode']}`；标签：**待人工复核**。", "",
             "以下是对初始标签的一致率，不能宣称为已验证的医疗准确率。", "",
             f"模型：{', '.join(report['model']) or '未取得模型响应'}；样本：{report['case_count']}。",
             f"Accuracy：{metrics['accuracy']:.4f}；Macro F1：{metrics['macro_f1']:.4f}；"
             f"调用错误/超时：{metrics['error_count']}（计入分母）。",
             f"耗时：{report['wall_seconds']:.2f}s；每条 mean/p50/p95/max："
             + "/".join(f"{metrics['latency_seconds'][key]:.2f}s" for key in ("mean", "p50", "p95", "max")),
             f"预期模型请求：{report['model_requests_expected']}；观测到的调用：{report['observed_model_call_count']}；"
             f"调用数未知的样本：{report['unknown_model_call_count_cases']}；无证据样本：{report['no_evidence_shortcut_count']}。", "",
             "|类别|Precision|Recall|F1|Support|", "|---|---:|---:|---:|---:|"]
    for label, item in metrics["per_class"].items():
        lines.append(f"|{label}|{item['precision']:.4f}|{item['recall']:.4f}|{item['f1']:.4f}|{item['support']}|")
    lines.extend(["", "混淆矩阵：行=人工初标，列=预测。", "",
                  "|真实\\预测|" + "|".join((*LABELS, "error")) + "|",
                  "|---|" + "---:|" * (len(LABELS) + 1)])
    for label, values in metrics["confusion_matrix"].items():
        lines.append("|" + label + "|" + "|".join(str(values[key]) for key in (*LABELS, "error")) + "|")
    lines.extend(["", "Support=0 的类别未被本次覆盖；零分母记 0，smoke 的 Macro F1 不用于比较模型。",
                  "JSON 含逐条理由、证据原文、引用检查、耗时、数据与代码摘要及错误明细。", "", "## 错误与引用差异", ""])
    for row in report["errors"]:
        lines.extend([f"- {row['id']}: {row['expected_label']} → {row['predicted_label']}；"
                      f"{row['elapsed_seconds']:.2f}s；引用匹配={row['evidence_quote_matches_expected']}",
                      f"  - 主张：{row['claim']}",
                      "  - 详情：" + json.dumps(row["error"] or row["reason"], ensure_ascii=False)])
    if not report["errors"]:
        lines.append("本次未发现标签或引用差异。")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true", help="Validate labels and pinned evidence without model calls")
    mode.add_argument("--smoke", action="store_true", help="Exactly the first 3 cases, real production verifier")
    mode.add_argument("--full", action="store_true", help="All cases, real production verifier; may be slow")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("results"))
    args = parser.parse_args(argv)
    try:
        dataset = load_dataset(args.dataset)
        if args.timeout_seconds <= 0 or not args.timeout_seconds < float("inf"):
            raise ValueError("timeout-seconds must be finite and positive")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    if args.validate_only:
        print(json.dumps({"validation": "passed", "cases": len(dataset["cases"]),
                          "labels": dict(Counter(case["expected_label"] for case in dataset["cases"])),
                          "label_status": dataset["label_status"], "model_calls": 0}, ensure_ascii=False, indent=2))
        return 0
    selected_mode = "smoke" if args.smoke else "full"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output = args.output_dir.resolve() / (selected_mode + "-" + run_id)
    output.mkdir(parents=True, exist_ok=False)
    # Incremental rows survive interruption; a missing report.json means incomplete.
    with (output / "progress.jsonl").open("x", encoding="utf-8") as progress:
        def record(row):
            progress.write(json.dumps(row, ensure_ascii=False) + "\n")
            progress.flush()
            print(f"{row['id']}: {row['predicted_label']} ({row['elapsed_seconds']:.2f}s)", flush=True)
        print(f"REAL MODEL / {selected_mode}: labels pending human review; output={output}", flush=True)
        report = run_evaluation(dataset, selected_mode, args.timeout_seconds, on_result=record)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text(render_report(report), encoding="utf-8")
    print(f"Report: {output / 'report.md'}", flush=True)
    # Disagreements are benchmark findings; infrastructure failures use exit 1.
    return 1 if report["metrics"]["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
