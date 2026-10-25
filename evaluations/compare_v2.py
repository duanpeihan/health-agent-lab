"""Bounded V1 archive versus V2 real-model experiment on identical V1 cases."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

from .dataset import DEFAULT_DATASET, ROOT, LABELS, load_dataset, resolve_documents
from .metrics import summarize
from .public_fixtures import public_fixture_path


DEFAULT_BASELINE = public_fixture_path("v1_full")
DEFAULT_IDS = ("S001", "C002", "C006", "N001", "U001", "U004")


def citation_audit(verdict, verification, documents, strategy):
    facts = []
    for doc in documents:
        for index, fact in enumerate(doc["facts"], 1):
            facts.append({"id": len(facts) + 1, "document_id": doc.get("id"),
                          "fact_index": index, "quote": fact})
    # Compare against the actual facts passed to the model, independent of seed-label quotes.
    if strategy == "v1":
        quote = verification.get("evidence_quote", "")
        cited = bool(quote)
        valid = quote in [fact["quote"] for fact in facts] if cited else None
        required = verdict in ("supported", "contradicted")
    else:
        refs = verification.get("evidence_references", [])
        cited = bool(refs)
        valid = (isinstance(refs, list) and all(ref in facts for ref in refs)
                 and len({ref["id"] for ref in refs}) == len(refs)
                 and verification.get("evidence_quote", "") == (refs[0]["quote"] if refs else "")) if cited else None
        required = verdict in ("supported", "contradicted", "uncertain")
    return {"cited": cited, "valid": valid,
            "required_citation_satisfied": (cited and valid is True) if required else (valid is not False)}


def citation_summary(rows):
    cited = [row for row in rows if row["citation"]["cited"]]
    return {"cited_cases": len(cited), "valid_cited_cases": sum(row["citation"]["valid"] is True for row in cited),
            "citation_validity_rate": (sum(row["citation"]["valid"] is True for row in cited) / len(cited)) if cited else None,
            "required_citation_satisfied": sum(row["citation"]["required_citation_satisfied"] for row in rows),
            "required_citation_rate": (sum(row["citation"]["required_citation_satisfied"] for row in rows) / len(rows)) if rows else None}


def invoke_v2(claim, documents, timeout_seconds):
    completed = subprocess.run([sys.executable, "-m", "evaluations.worker_v2"], cwd=ROOT,
                               input=json.dumps({"claim": claim, "documents": documents}, ensure_ascii=False),
                               capture_output=True, text=True, timeout=timeout_seconds, check=True)
    return json.loads(completed.stdout)


def compare(dataset, baseline, ids, timeout_seconds, invoke=invoke_v2):
    if baseline.get("execution_mode") != "real_model" or baseline.get("case_count") != len(dataset["cases"]):
        raise ValueError("baseline must be a completed full real-model V1 report")
    if baseline.get("dataset_sha256") != dataset["_sha256"]:
        raise ValueError("baseline dataset SHA256 differs")
    if baseline.get("verifier_sha256") != hashlib.sha256((ROOT / "app" / "claim_verifier.py").read_bytes()).hexdigest():
        raise ValueError("V1 production verifier has changed")
    if not ids or len(ids) != len(set(ids)) or len(ids) > 8:
        raise ValueError("choose 1 to 8 unique IDs; larger runs need a separate decision")
    cases = {case["id"]: case for case in dataset["cases"]}
    stored = {row["id"]: row for row in baseline["results"]}
    if any(case_id not in cases or case_id not in stored for case_id in ids):
        raise ValueError("selected IDs must exist in the archived V1 report")
    v1, v2 = [], []
    for case_id in ids:
        case = cases[case_id]
        documents = resolve_documents(case, dataset["_knowledge"])
        old = stored[case_id]
        old_verification = {"verdict": old["predicted_label"], "evidence_quote": old["evidence_quote"]}
        v1.append({"id": case_id, "expected_label": case["expected_label"],
                   "predicted_label": old["predicted_label"], "elapsed_seconds": old["elapsed_seconds"],
                   "reason": old["reason"], "verification": old_verification,
                   "citation": citation_audit(old["predicted_label"], old_verification, documents, "v1"),
                   "model_call_count": old["observed_model_call_count"], "error": old["error"]})
        start = time.perf_counter()
        verification, call_count, error = {}, None, None
        try:
            response = invoke(case["claim"], documents, timeout_seconds)
            verification = response["verification"]
            call_count = response["model_call_count"]
            expected_calls = int(bool([fact for doc in documents for fact in doc["facts"]]))
            if type(call_count) is not int or call_count != expected_calls or len(response["model_calls"]) != call_count:
                raise ValueError("V2 model call observation mismatch")
            if any(call.get("completed") is not True for call in response["model_calls"]):
                raise ValueError("V2 model call incomplete")
            if verification["verdict"] not in LABELS:
                raise ValueError("V2 invalid verdict")
            citation = citation_audit(verification["verdict"], verification, documents, "v2")
            if citation["valid"] is False or not citation["required_citation_satisfied"]:
                raise ValueError("V2 citation contract failed")
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
            if isinstance(exc, subprocess.TimeoutExpired):
                error["message"] = f"V2 worker exceeded {timeout_seconds:g} seconds"
            elif isinstance(exc, subprocess.CalledProcessError):
                error["stderr"] = (exc.stderr or "")[-4000:]
            citation = {"cited": False, "valid": None, "required_citation_satisfied": False}
        v2.append({"id": case_id, "expected_label": case["expected_label"],
                   "predicted_label": verification.get("verdict", "error") if error is None else "error",
                   "elapsed_seconds": time.perf_counter() - start, "reason": verification.get("reason", ""),
                   "verification": verification, "citation": citation, "model_call_count": call_count,
                   "error": error})
        print(f"{case_id}: V1={v1[-1]['predicted_label']} V2={v2[-1]['predicted_label']}", flush=True)
    return {"status": "exploratory", "label_status": "pending_human_review",
            "note": "Same six V1 seed cases; V1 rows reused from historical full run; only V2 was called now. Runtime environments are not simultaneous.",
            "selected_ids": list(ids), "dataset_sha256": dataset["_sha256"],
            "baseline_started_at_utc": baseline["started_at_utc"],
            "v2_started_at_utc": datetime.now(timezone.utc).isoformat(),
            "v1": {"rows": v1, "metrics": summarize(v1), "citations": citation_summary(v1),
                   "observed_model_call_count": sum(row["model_call_count"] or 0 for row in v1)},
            "v2": {"rows": v2, "metrics": summarize(v2), "citations": citation_summary(v2),
                   "observed_model_call_count": sum(row["model_call_count"] or 0 for row in v2),
                   "unknown_model_call_count_cases": sum(row["model_call_count"] is None for row in v2)},
            "v2_source_sha256": hashlib.sha256(Path(__file__).with_name("v2_strategy.py").read_bytes()).hexdigest()}


def render(report):
    lines = ["# Claim Verifier V1 / V2 小样本对照", "", "**探索性**：原24条初标尚待独立人工复核。"
             "相同样本上复用V1历史真实结果，本次只调用V2；P50/P95不可视为严格同环境速度对比。", "",
             f"案例：{', '.join(report['selected_ids'])}", "",
             "|指标|V1 历史|V2 本次|", "|---|---:|---:|"]
    for key in ("accuracy", "macro_f1"):
        lines.append(f"|{key}|{report['v1']['metrics'][key]:.4f}|{report['v2']['metrics'][key]:.4f}|")
    for label in LABELS:
        lines.append(f"|{label} recall|{report['v1']['metrics']['per_class'][label]['recall']:.4f}|{report['v2']['metrics']['per_class'][label]['recall']:.4f}|")
    for key in ("p50", "p95"):
        lines.append(f"|{key} 秒|{report['v1']['metrics']['latency_seconds'][key]:.2f}|{report['v2']['metrics']['latency_seconds'][key]:.2f}|")
    lines.append(f"|观测模型调用次数|{report['v1']['observed_model_call_count']}|{report['v2']['observed_model_call_count']}|")
    lines.append(f"|引用有效率（有效/非空引用）|{report['v1']['citations']['valid_cited_cases']}/{report['v1']['citations']['cited_cases']}|"
                 f"{report['v2']['citations']['valid_cited_cases']}/{report['v2']['citations']['cited_cases']}|")
    lines.extend(["", "引用有效率仅检查编号/原文是否确实来自输入证据，不证明证据能支持结论。"
                  "未引用样本不进入该分母；分类和引用分开审计。", "", "|ID|初标|V1|V2|V2 数字检查|V2 量词检查|", "|---|---|---|---|---|---|"])
    for old, new in zip(report["v1"]["rows"], report["v2"]["rows"]):
        verification = new["verification"]
        lines.append(f"|{old['id']}|{old['expected_label']}|{old['predicted_label']}|{new['predicted_label']}|"
                     f"{verification.get('numeric_check', '')}|{verification.get('quantifier_check', '')}|")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--case-ids", nargs="+", default=list(DEFAULT_IDS))
    parser.add_argument("--timeout-seconds", type=float, default=180)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("results"))
    args = parser.parse_args(argv)
    if not 0 < args.timeout_seconds < float("inf"):
        parser.error("timeout must be finite and positive")
    dataset = load_dataset(DEFAULT_DATASET)
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    report = compare(dataset, baseline, args.case_ids, args.timeout_seconds)
    output = args.output_dir / ("compare-v2-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text(render(report), encoding="utf-8")
    print(f"Report: {output / 'report.md'}")
    return 1 if report["v2"]["metrics"]["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
