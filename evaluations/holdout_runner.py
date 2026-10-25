"""Run both frozen strategies once on each reviewed holdout item."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

from app import claim_verifier
from . import v2_strategy
from .compare_v2 import citation_audit, citation_summary
from .dataset import ROOT, LABELS, load_dataset, resolve_documents
from .metrics import summarize


DEFAULT_DATASET = Path(__file__).with_name("claims.holdout.v2.json")
DEFAULT_FREEZE = Path(__file__).with_name("frozen") / "holdout-v2.freeze.json"
MODEL_MANIFEST = Path.home() / ".ollama/models/manifests/registry.ollama.ai/library/qwen3/4b"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_freeze(freeze, dataset_path=DEFAULT_DATASET):
    for relative, digest in freeze["source_sha256"].items():
        if sha(ROOT / relative) != digest:
            raise ValueError(f"frozen source changed: {relative}")
    if sha(Path(__file__).with_name("frozen") / "holdout-label-review.json") != freeze["agent_semantic_review_sha256"]:
        raise ValueError("frozen semantic review changed")
    if Path(dataset_path).resolve() != DEFAULT_DATASET.resolve():
        raise ValueError("holdout runner requires the frozen dataset path")
    if sha(MODEL_MANIFEST) != freeze["model"]["manifest_sha256"]:
        raise ValueError("local qwen3:4b manifest changed")
    if claim_verifier.MODEL != v2_strategy.MODEL or claim_verifier.MODEL != freeze["model"]["name"]:
        raise ValueError("V1/V2 model names differ from freeze")
    v1_fingerprint = sha(ROOT / "app/claim_verifier.py")
    v2_fingerprint = hashlib.sha256(v2_strategy.SYSTEM_PROMPT_V2.encode()).hexdigest()
    if (v1_fingerprint != freeze["prompt_fingerprints"]["v1_source_sha256"]
            or v2_fingerprint != freeze["prompt_fingerprints"]["v2_system_prompt_sha256"]):
        raise ValueError("frozen prompt changed")
    if freeze["model"]["options"] != {"temperature": 0} or freeze["model"]["think"] is not False:
        raise ValueError("frozen model options invalid")
    if version("ollama") != freeze["model"]["sdk_version"]:
        raise ValueError("ollama SDK version changed")
    dataset = load_dataset(dataset_path)
    if len(dataset["cases"]) != 8 or {label: sum(c["expected_label"] == label for c in dataset["cases"])
                                      for label in LABELS} != freeze["labels"]:
        raise ValueError("frozen holdout size or labels changed")
    return dataset


def invoke_worker(strategy, claim, documents, timeout_seconds):
    request = {"strategy": strategy, "claim": claim, "documents": documents}
    completed = subprocess.run([sys.executable, "-m", "evaluations.holdout_worker"], cwd=ROOT,
                               input=json.dumps(request, ensure_ascii=False), capture_output=True,
                               text=True, timeout=timeout_seconds, check=True)
    return json.loads(completed.stdout)


def run_paired(dataset, freeze, timeout_seconds, invoke=invoke_worker, on_result=None):
    if not 0 < timeout_seconds < float("inf"):
        raise ValueError("timeout must be finite and positive")
    mode = "real_model" if invoke is invoke_worker else "test_double"
    rows = {strategy: [] for strategy in ("v1", "v2")}
    started = datetime.now(timezone.utc).isoformat()
    for index, case in enumerate(dataset["cases"]):
        documents = resolve_documents(case, dataset["_knowledge"])
        evidence_hash = hashlib.sha256(json.dumps(documents, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        order = ("v1", "v2") if index % 2 == 0 else ("v2", "v1")
        for strategy in order:
            start = time.perf_counter()
            verification, model_calls, count, error = {}, [], None, None
            try:
                response = invoke(strategy, case["claim"], documents, timeout_seconds)
                if response.get("strategy") != strategy or response.get("model") != freeze["model"]["name"]:
                    raise ValueError("worker returned the wrong strategy or model")
                count, model_calls = response["model_call_count"], response["model_calls"]
                if (type(count) is not int or count != 1 or not isinstance(model_calls, list)
                        or len(model_calls) != 1 or model_calls[0].get("completed") is not True):
                    raise ValueError("worker model-call observation invalid")
                if mode == "real_model" and (model_calls[0].get("temperature") != 0
                                             or model_calls[0].get("think") is not False
                                             or model_calls[0].get("model") != freeze["model"]["name"]):
                    raise ValueError("worker model options differ from freeze")
                verification = response["verification"]
                if verification.get("verdict") not in LABELS or not isinstance(verification.get("reason"), str):
                    raise ValueError("invalid verifier response")
                citation = citation_audit(verification["verdict"], verification, documents, strategy)
            except Exception as exc:
                error = {"type": type(exc).__name__, "message": str(exc)}
                if isinstance(exc, subprocess.TimeoutExpired):
                    error["message"] = f"worker exceeded {timeout_seconds:g} seconds"
                elif isinstance(exc, subprocess.CalledProcessError):
                    error["stderr"] = (exc.stderr or "")[-4000:]
                citation = {"cited": False, "valid": None, "required_citation_satisfied": False}
            row = {"id": case["id"], "claim": case["claim"], "expected_label": case["expected_label"],
                   "predicted_label": verification.get("verdict", "error") if error is None else "error",
                   "reason": verification.get("reason", ""), "verification": verification,
                   "evidence_origin": case["evidence_origin"], "documents": documents,
                   "input_evidence_sha256": evidence_hash, "citation": citation,
                   "model_call_count": count, "model_calls": model_calls,
                   "elapsed_seconds": time.perf_counter() - start, "error": error}
            rows[strategy].append(row)
            if on_result:
                on_result(strategy, row)
    for old, new in zip(rows["v1"], rows["v2"]):
        if old["id"] != new["id"] or old["claim"] != new["claim"] or old["input_evidence_sha256"] != new["input_evidence_sha256"]:
            raise ValueError("paired inputs differed")
    return {"schema_version": "1.0", "execution_mode": mode, "status": "exploratory",
            "label_status": "pending_independent_human_review", "started_at_utc": started,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "freeze_id": freeze["freeze_id"], "frozen_source_sha256": freeze["source_sha256"],
            "model": freeze["model"], "case_count": len(dataset["cases"]),
            "timeout_seconds_per_invocation": timeout_seconds,
            "call_order": "alternating V1/V2 per case",
            "strategies": {strategy: {"rows": items, "metrics": summarize(items),
                                      "citations": citation_summary(items),
                                      "observed_model_call_count": sum(row["model_call_count"] or 0 for row in items),
                                      "unknown_model_call_count_cases": sum(row["model_call_count"] is None for row in items)}
                           for strategy, items in rows.items()}}


def render_report(report):
    v1, v2 = (report["strategies"][strategy] for strategy in ("v1", "v2"))
    lines = ["# 8 条独立保留案例：V1 / V2 配对真实评测", "",
             "**探索性初步评估。** 标签仅经 Agent 按给定证据语义复核，尚待独立人工复核；未作医学专家验证。",
             "两种策略在同一程序中使用相同主张与证据、qwen3:4b、temperature=0、think=False，逐条交替调用。", "",
             "|指标|V1|V2|", "|---|---:|---:|"]
    for name, key in (("Accuracy", "accuracy"), ("Macro-F1", "macro_f1")):
        lines.append(f"|{name}|{v1['metrics'][key]:.4f}|{v2['metrics'][key]:.4f}|")
    for key in ("p50", "p95"):
        lines.append(f"|{key} 秒|{v1['metrics']['latency_seconds'][key]:.2f}|{v2['metrics']['latency_seconds'][key]:.2f}|")
    lines.append(f"|观测模型调用次数|{v1['observed_model_call_count']}|{v2['observed_model_call_count']}|")
    lines.append(f"|引用有效率（有效/非空）|{v1['citations']['valid_cited_cases']}/{v1['citations']['cited_cases']}|"
                 f"{v2['citations']['valid_cited_cases']}/{v2['citations']['cited_cases']}|")
    lines.extend(["", "|类别|V1 Precision|V1 Recall|V1 F1|V2 Precision|V2 Recall|V2 F1|",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for label in LABELS:
        a,b=v1["metrics"]["per_class"][label],v2["metrics"]["per_class"][label]
        lines.append(f"|{label}|{a['precision']:.4f}|{a['recall']:.4f}|{a['f1']:.4f}|"
                     f"{b['precision']:.4f}|{b['recall']:.4f}|{b['f1']:.4f}|")
    for strategy, data in (("V1", v1), ("V2", v2)):
        lines.extend(["", f"## {strategy} 混淆矩阵", "", "|初标\\预测|" + "|".join((*LABELS, "error")) + "|",
                      "|---|" + "---:|" * (len(LABELS) + 1)])
        for label, values in data["metrics"]["confusion_matrix"].items():
            lines.append("|" + label + "|" + "|".join(str(values[key]) for key in (*LABELS, "error")) + "|")
    lines.extend(["", "## 逐条预测与错误", "", "|ID|初标|V1|V2|V1 引用|V2 引用|",
                  "|---|---|---|---|---|---|"])
    for a,b in zip(v1["rows"],v2["rows"]):
        lines.append(f"|{a['id']}|{a['expected_label']}|{a['predicted_label']}|{b['predicted_label']}|"
                     f"{'有效' if a['citation']['valid'] is True else '无引用' if not a['citation']['cited'] else '无效'}|"
                     f"{'有效' if b['citation']['valid'] is True else '无引用' if not b['citation']['cited'] else '无效'}|")
    for strategy, data in (("V1", v1), ("V2", v2)):
        errors=[row for row in data["rows"] if row["predicted_label"] != row["expected_label"] or row["citation"]["valid"] is False]
        lines.extend(["", f"### {strategy} 差异（{len(errors)} 条）", ""])
        for row in errors:
            lines.append(f"- **{row['id']}** {row['expected_label']} → {row['predicted_label']}；"
                         f"{row['elapsed_seconds']:.2f} 秒；理由：{row['reason'] or row['error']}")
        if not errors:
            lines.append("无分类或引用差异。")
    lines.extend(["", "引用有效率只核对非空引用是否来自实际输入原文，不证明推理正确；无引用不进入该分母。",
                  "P50/P95 含子进程启动，P95 为 nearest-rank；8 条样本过小，不足以估计真实表现。",
                  "完整原文证据、结构化引用、调用记录和逐条耗时见同目录 JSON。", ""])
    return "\n".join(lines)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",action="store_true",help="Explicitly authorize 16 real-model invocations")
    parser.add_argument("--timeout-seconds",type=float,default=60)
    parser.add_argument("--output-dir",type=Path,default=Path(__file__).with_name("results"))
    args=parser.parse_args(argv)
    if not args.run:
        parser.error("pass --run to execute both strategies on all 8 frozen cases")
    freeze=json.loads(DEFAULT_FREEZE.read_text())
    dataset=verify_freeze(freeze)
    output=args.output_dir/("holdout-paired-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+uuid.uuid4().hex[:8])
    output.mkdir(parents=True,exist_ok=False)
    with (output/"progress.jsonl").open("x",encoding="utf-8") as progress:
        def record(strategy,row):
            progress.write(json.dumps({"strategy":strategy,**row},ensure_ascii=False)+"\n")
            progress.flush()
            print(f"{row['id']} {strategy}: {row['predicted_label']} ({row['elapsed_seconds']:.2f}s)",flush=True)
        report=run_paired(dataset,freeze,args.timeout_seconds,on_result=record)
    verify_freeze(freeze)
    report["freeze_verified_postrun"] = True
    (output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (output/"report.md").write_text(render_report(report),encoding="utf-8")
    print(f"Report: {output/'report.md'}",flush=True)
    return 1 if any(part["metrics"]["error_count"] for part in report["strategies"].values()) else 0


if __name__=="__main__":
    raise SystemExit(main())
