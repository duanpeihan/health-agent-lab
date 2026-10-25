"""Rescore saved Claim Verifier predictions; never invoke or import a model client."""

import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

from .dataset import LABELS, ROOT, load_dataset, resolve_documents
from .metrics import summarize
from .review_suggestions_check import DEFAULT_CSV, validate_suggestions


RESULTS = Path(__file__).with_name("results")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_labels(path=DEFAULT_CSV):
    validation = validate_suggestions(path)
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        labels = {row["案例ID"]: row for row in csv.DictReader(stream)}
    require(len(labels) == 32, "AI suggestions must contain all 32 cases")
    sources = {}
    datasets = []
    for filename in ("claims.v1.json", "claims.holdout.v2.json"):
        dataset = load_dataset(Path(__file__).with_name(filename))
        datasets.append(dataset)
        for case in dataset["cases"]:
            sources[case["id"]] = {
                "case": case, "documents": resolve_documents(case, dataset["_knowledge"]),
                "dataset_sha256": dataset["_sha256"],
            }
    require(set(labels) == set(sources), "AI suggestions and source case IDs differ")
    return labels, sources, datasets, validation


def archive_parts(report):
    """Keep every archive separate, including legacy failed V2 attempts."""
    if "results" in report and report.get("mode") in ("full", "smoke"):
        return report["mode"], {"v1": report["results"]}
    if "v1" in report and "v2" in report and "selected_ids" in report:
        return "legacy_v1_v2_comparison", {key: report[key]["rows"] for key in ("v1", "v2")}
    if "strategies" in report and "v1" in report["strategies"] and "v2" in report["strategies"]:
        return "heldout_paired", {key: report["strategies"][key]["rows"] for key in ("v1", "v2")}
    return None


def score_rows(rows, labels, sources, report):
    require(isinstance(rows, list) and bool(rows), "Archive rows must be a nonempty list")
    ids = [row.get("id") for row in rows]
    require(len(ids) == len(set(ids)), "Archive has duplicate case IDs")
    require(all(case_id in labels and case_id in sources for case_id in ids),
            "Archive has a case missing from label/source records")
    if "selected_ids" in report:
        require(ids == report["selected_ids"], "Archive rows omit/reorder selected cases")
    elif "case_count" in report:
        require(len(rows) == report["case_count"], "Archive case_count differs from rows")
    original, proposed, details = [], [], []
    for row in rows:
        case_id = row["id"]
        source = sources[case_id]
        case = source["case"]
        require(row.get("expected_label") == case["expected_label"] == labels[case_id]["原始标签"],
                f"{case_id}: archived original label differs from source")
        require(labels[case_id]["建议人工复核标签"] in LABELS, f"{case_id}: invalid AI label")
        if report.get("dataset_sha256"):
            require(report["dataset_sha256"] == source["dataset_sha256"],
                    f"{case_id}: archive dataset SHA256 differs")
        if "claim" in row:
            require(row["claim"] == case["claim"], f"{case_id}: archived claim differs")
        if "documents" in row:
            require(row["documents"] == source["documents"], f"{case_id}: archived evidence differs")
        if row.get("input_evidence_sha256"):
            encoded = json.dumps(source["documents"], ensure_ascii=False, sort_keys=True).encode()
            require(hashlib.sha256(encoded).hexdigest() == row["input_evidence_sha256"],
                    f"{case_id}: archived input evidence hash differs")
        prediction = "error" if row.get("error") else row.get("predicted_label")
        require(prediction in (*LABELS, "error"), f"{case_id}: invalid archived prediction")
        elapsed = row.get("elapsed_seconds")
        require(type(elapsed) in (int, float) and 0 <= elapsed < float("inf"),
                f"{case_id}: invalid archived latency")
        seed = {"expected_label": case["expected_label"], "predicted_label": prediction,
                "elapsed_seconds": elapsed}
        ai = dict(seed, expected_label=labels[case_id]["建议人工复核标签"])
        original.append(seed)
        proposed.append(ai)
        details.append({
            "id": case_id, "claim": case["claim"], "predicted_label": prediction,
            "original_label": seed["expected_label"], "ai_suggested_label": ai["expected_label"],
            "original_agreement": prediction == seed["expected_label"],
            "ai_suggested_agreement": prediction == ai["expected_label"],
            "label_changed": seed["expected_label"] != ai["expected_label"],
            "original_error": deepcopy(row.get("error")),
            "claim_check": "matched" if "claim" in row else "not_saved_in_legacy_archive",
            "evidence_check": "matched" if "documents" in row else "not_saved_in_legacy_archive",
        })
    original_metrics, ai_metrics = summarize(original), summarize(proposed)
    historical_latency = original_metrics.pop("latency_seconds")
    ai_metrics.pop("latency_seconds")
    return {
        "case_count": len(rows), "covered_ids": ids,
        "original_labels": original_metrics, "ai_suggested_labels": ai_metrics,
        "historical_latency_seconds": historical_latency,
        "latency_interpretation": "Copied historical per-case duration; not newly measured or comparable new runtime",
        "changed_label_rows": [row for row in details if row["label_changed"]],
        "changed_agreement_rows": [row for row in details if row["original_agreement"] != row["ai_suggested_agreement"]],
        "rows": details,
    }


def rescore_archives(paths, labels, sources):
    archives = []
    for path in paths:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        parts = archive_parts(report)
        if parts is None:
            continue
        kind, rows_by_strategy = parts
        if report.get("knowledge_sha256"):
            require(report["knowledge_sha256"] == digest(ROOT / "data/health_knowledge.json"),
                    f"Knowledge snapshot differs for {path}")
        for relative, expected in report.get("frozen_source_sha256", {}).items():
            require(digest(ROOT / relative) == expected, f"Frozen source differs: {relative}")
        archive = {
            "source_report_path": str(Path(path).resolve()), "source_report_sha256": digest(path),
            "archive_kind": kind,
            "source_execution_mode": report.get("execution_mode", "not_recorded_in_legacy_archive"),
            "source_started_at_utc": report.get("started_at_utc", report.get("v2_started_at_utc")),
            "strategy_scores": {strategy: score_rows(rows, labels, sources, report)
                                for strategy, rows in rows_by_strategy.items()},
            "source_note": report.get("note"),
        }
        if kind == "legacy_v1_v2_comparison":
            archive["reuse_note"] = "V1 rows reused from the historical full baseline; not independent new V1 observations"
        archives.append(archive)
    return archives


def build_report(paths, label_path=DEFAULT_CSV):
    labels, sources, datasets, validation = load_labels(label_path)
    archives = rescore_archives(paths, labels, sources)
    require(bool(archives), "No supported historical archives found")
    return {
        "schema_version": "1.0", "execution_mode": "offline_rescore", "new_model_calls": 0,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": "Agreement with two unreviewed label versions; not a new real-model experiment or medical accuracy",
        "human_review_completed": False,
        "label_source": {"path": str(Path(label_path).resolve()), "sha256": digest(label_path),
                         "validation": validation},
        "datasets": [{"id": data["dataset_id"], "path": data["_path"], "sha256": data["_sha256"]}
                     for data in datasets],
        "archives": archives,
        "pooling_policy": "No pooled score: overlapping/reused rows and failed attempts remain in their own archives",
        "coverage_note": "No 32-case V2 run exists; V2 coverage is only the IDs saved in each archive",
    }


def render_report(report):
    lines = ["# 已保存 V1/V2 预测的离线重评分", "",
             "执行模式：`offline_rescore`；本次新增模型调用：**0**。",
             "以下分别计算与原始标签、AI 建议标签的一致情况。两版均未经独立人工审核，不能称医学准确率。",
             "U006 的 AI 建议暂为 not_supported，原始 uncertain 和争议说明保持不变。", "",
             "每份历史报告独立重评分，不将重复 V1 基线或 V2 失败尝试混池，也不拼成不存在的 32 条 V2 实验。", "",
             "|历史报告|策略|条数|原标正确数/总数|AI建议正确数/总数|原标 Macro F1|AI建议 Macro F1|历史调用错误|",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for archive in report["archives"]:
        name = Path(archive["source_report_path"]).parent.name
        for strategy, score in archive["strategy_scores"].items():
            old, new = score["original_labels"], score["ai_suggested_labels"]
            lines.append(f"|{name}|{strategy}|{score['case_count']}|{old['correct']}/{old['total']} ({old['accuracy']:.2%})|"
                         f"{new['correct']}/{new['total']} ({new['accuracy']:.2%})|{old['macro_f1']:.4f}|{new['macro_f1']:.4f}|{old['error_count']}|")
    for archive in report["archives"]:
        name = Path(archive["source_report_path"]).parent.name
        lines.extend(["", f"## {name}", "", f"源报告 SHA256：`{archive['source_report_sha256']}`。",
                      f"源执行模式记录：`{archive['source_execution_mode']}`。"])
        if archive.get("reuse_note"):
            lines.append("V1 复用历史全量基线，不是一次新的独立 V1 实验。旧对照报告没有保存逐条主张和证据，相关字段核验标为不可用；已核对数据集哈希和原始标签。")
        for strategy, score in archive["strategy_scores"].items():
            lines.extend(["", f"### {strategy}：{score['case_count']} 条", "",
                          "覆盖 ID：" + ", ".join(score["covered_ids"]), "",
                          "|类别|原标 P|原标 R|原标 F1|原标 support|AI建议 P|AI建议 R|AI建议 F1|AI建议 support|",
                          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
            for label in LABELS:
                a = score["original_labels"]["per_class"][label]
                b = score["ai_suggested_labels"]["per_class"][label]
                lines.append(f"|{label}|{a['precision']:.4f}|{a['recall']:.4f}|{a['f1']:.4f}|{a['support']}|"
                             f"{b['precision']:.4f}|{b['recall']:.4f}|{b['f1']:.4f}|{b['support']}|")
            changed = score["changed_label_rows"]
            if changed:
                for row in changed:
                    lines.append(f"\n改标覆盖：{row['id']}，{row['original_label']} → {row['ai_suggested_label']}；"
                                 f"历史预测 {row['predicted_label']}；原标一致={row['original_agreement']}，AI建议一致={row['ai_suggested_agreement']}。")
            else:
                lines.append("\n本报告覆盖的案例没有标签变化。")
            for version_name in ("original_labels", "ai_suggested_labels"):
                title = "原始标签" if version_name == "original_labels" else "AI建议标签"
                lines.extend(["", f"{title}混淆矩阵（行=标签、列=预测）：", "",
                              "|标签\\预测|" + "|".join((*LABELS, "error")) + "|",
                              "|---|" + "---:|" * (len(LABELS) + 1)])
                for label, row in score[version_name]["confusion_matrix"].items():
                    lines.append("|" + label + "|" + "|".join(str(row[key]) for key in (*LABELS, "error")) + "|")
    lines.extend(["", "所有延时均沿用历史记录，只在 JSON 的 historical_latency_seconds 保留，不是本次新测速。",
                  "调用失败/超时仍计入分母；缺类的 smoke 的 Macro F1 不用于策略优劣结论。",
                  "完整逐条标签差异、错误、证据核验状态及源路径见同目录 report.json。", ""])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--labels", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    args = parser.parse_args(argv)
    report = build_report(sorted(args.results_dir.glob("*/report.json")), args.labels)
    output = args.output_dir / ("rescore-ai-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text(render_report(report), encoding="utf-8")
    print(f"OFFLINE RESCORE / new_model_calls=0 / archives={len(report['archives'])}: {output}")


if __name__ == "__main__":
    main()
