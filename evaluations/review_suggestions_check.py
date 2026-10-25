"""Check AI review suggestions offline; this is not semantic or human validation.

Install as evaluations/review_suggestions_check.py and run:
    python -m evaluations.review_suggestions_check [CSV_PATH]
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re

from . import review_checks
from .dataset import LABELS, load_dataset, resolve_documents


EVALUATIONS = Path(__file__).resolve().parent
REPO = EVALUATIONS.parent
MANIFEST = EVALUATIONS / "review_suggestions_manifest.json"
OUTPUT_RELATIVE = "evaluations/review_checklist_ai_suggestions.csv"
DEFAULT_CSV = REPO / OUTPUT_RELATIVE
DATE = "2026-10-24"
AI_STATUS = "AI生成建议；未独立人工审核"
EXTRA_FIELDS = (
    "建议人工复核标签",
    "建议人工复核理由",
    "证据关键片段（JSON）",
    "本轮争议类型",
    "需人工裁定项",
    "复核优先级",
    "建议来源与状态",
    "建议日期",
)
DISPUTE_TYPES = (
    "无明显标签争议",
    "标签边界",
    "规约与逻辑方向冲突",
    "历史解释忠实度",
)
PRIORITIES = (
    "常规（逐行确认）",
    "优先（标签边界）",
    "最高（标签裁定）",
    "优先（解释核对）",
)
REQUIRED_INPUTS = {
    "evaluations/review_checklist.csv",
    "evaluations/annotation_guidelines.md",
    "evaluations/claims.v1.json",
    "evaluations/claims.holdout.v2.json",
    "data/health_knowledge.json",
}
# Public clones retain the authoritative synthetic evidence/label inputs, while
# the original whole-repository manifest and private raw archives stay intact.
# Only strict_snapshot=True claims to verify that entire historical workspace.
PUBLIC_PROTECTED_INPUTS = REQUIRED_INPUTS | {
    "evaluations/explanation_fidelity_review.csv",
    "evaluations/frozen/holdout-label-review.json",
    "evaluations/frozen/holdout-v2.freeze.json",
}
# Legacy runtime subset retained for consumers of the original snapshot audit.
# The public evidence/label scope above now also reports unavailable private
# archives and unrelated engineering drift; full historical strictness remains
# available only through strict_snapshot=True. Original manifest hashes stay intact.
RUNTIME_SNAPSHOT_FILES = {"app/agent.py", "app/claim_verifier.py", "tests/test_refactor.py",
                          "evaluations/tests/test_holdout_runner.py",
                          "evaluations/tests/test_v2_experiment.py"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON {path}: {exc}") from exc


def repo_file(relative):
    require(isinstance(relative, str) and bool(relative), "Missing repository-relative file path")
    local = Path(relative)
    require(not local.is_absolute() and ".." not in local.parts,
            f"Path must stay relative to the repository: {relative}")
    target = (REPO / local).resolve()
    require(target.is_relative_to(REPO.resolve()), f"Path escapes repository: {relative}")
    require(target.is_file(), f"Missing file: {relative}")
    return target


def verify_hash(path, expected, description):
    require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected) is not None,
            f"Invalid SHA256 for {description}")
    try:
        observed = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError(f"Cannot read {description}: {exc}") from exc
    require(observed == expected,
            f"SHA256 mismatch for {description}: expected {expected}, observed {observed}")


def verify_manifest(strict_snapshot=True):
    manifest = read_json(MANIFEST)
    require(isinstance(manifest, dict), "Manifest must be an object")
    require(manifest.get("schema_version") == "1.0", "Unsupported manifest schema_version")
    require(manifest.get("generated_on") == DATE, "Manifest generated_on differs")
    require(manifest.get("time_zone") == "Asia/Shanghai", "Manifest time_zone differs")
    require(manifest.get("ai_suggestions_only") is True, "Manifest must identify AI-only suggestions")
    require(manifest.get("human_review_completed") is False,
            "Manifest must not claim completed human review")
    inputs = manifest.get("inputs")
    require(isinstance(inputs, list) and bool(inputs), "Manifest inputs must be a nonempty list")
    seen, runtime_changes, unavailable = set(), [], []
    for entry in inputs:
        require(isinstance(entry, dict), "Each manifest input must be an object")
        relative = entry.get("path")
        require(isinstance(relative, str) and bool(relative), "Missing repository-relative file path")
        local = Path(relative)
        require(not local.is_absolute() and ".." not in local.parts,
                f"Path must stay relative to the repository: {relative}")
        path = (REPO / local).resolve()
        require(path.is_relative_to(REPO.resolve()), f"Path escapes repository: {relative}")
        require(relative not in seen, f"Duplicate manifest input: {relative}")
        seen.add(relative)
        expected = entry.get("sha256")
        require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected) is not None,
                f"Invalid SHA256 for {relative}")
        if not path.is_file() and not strict_snapshot and relative not in PUBLIC_PROTECTED_INPUTS:
            unavailable.append({"path": relative, "historical_sha256": expected})
            continue
        require(path.is_file(), f"Missing file: {relative}")
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if (not strict_snapshot and relative not in PUBLIC_PROTECTED_INPUTS
                and observed != expected):
            runtime_changes.append({"path": relative, "historical_sha256": entry.get("sha256"),
                                    "current_sha256": observed})
        else:
            verify_hash(path, entry.get("sha256"), relative)
    require(REQUIRED_INPUTS.issubset(seen),
            f"Manifest omits protected inputs: {sorted(REQUIRED_INPUTS - seen)}")
    output = manifest.get("output")
    require(isinstance(output, dict), "Manifest output must be an object")
    require(output.get("path") == OUTPUT_RELATIVE, "Manifest canonical output path differs")
    # Always verify the canonical file, even when the CLI receives a different CSV.
    verify_hash(repo_file(OUTPUT_RELATIVE), output.get("sha256"), OUTPUT_RELATIVE)
    return {**manifest, "snapshot_validation": {
        "strict_snapshot": strict_snapshot,
        "validation_scope": "full_historical_workspace" if strict_snapshot else "public_evidence_and_label_inputs",
        "runtime_source_changes": runtime_changes,
        "unavailable_historical_files": unavailable,
        "all_historical_snapshot_files_match": not runtime_changes and not unavailable,
        "label_and_evidence_inputs_match": True,
    }}


def read_csv(path, fields):
    try:
        with Path(path).open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            require(tuple(reader.fieldnames or ()) == tuple(fields),
                    f"CSV columns or column order differ: {path}")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"Cannot read CSV {path}: {exc}") from exc
    for line, row in enumerate(rows, start=2):
        require(None not in row and all(isinstance(value, str) for value in row.values()),
                f"Malformed CSV field count at record {line}: {path}")
    return rows


def source_cases():
    cases = []
    for group, name in (("原始24条", "claims.v1.json"),
                        ("独立保留8条", "claims.holdout.v2.json")):
        data = load_dataset(EVALUATIONS / name)
        for case in data["cases"]:
            documents = resolve_documents(case, data["_knowledge"])
            facts = [fact for document in documents for fact in document["facts"]]
            evidence = "\n".join(
                f"[{document['id']}] {fact}"
                for document in documents for fact in document["facts"]
            ) or "（无输入证据）"
            cases.append((group, case, facts, evidence))
    require(len(cases) == 32 and len({case["id"] for _, case, _, _ in cases}) == 32,
            "Datasets must contain exactly 32 unique cases")
    return cases


def validate_suggestions(path=DEFAULT_CSV, *, strict_snapshot=False):
    manifest = verify_manifest(strict_snapshot=strict_snapshot)
    # This validates the original proposals and empty human fields, not new labels.
    review_checks.validate_checklist()
    originals = read_csv(EVALUATIONS / "review_checklist.csv", review_checks.FIELDS)
    rows = read_csv(path, review_checks.FIELDS + EXTRA_FIELDS)
    cases = source_cases()
    expected_ids = [case["id"] for _, case, _, _ in cases]
    require(len(rows) == 32 and len({row["案例ID"] for row in rows}) == 32,
            "Suggestions must contain exactly 32 unique case IDs")
    require([row["案例ID"] for row in rows] == expected_ids,
            "Suggestion case IDs/order differ from the two source datasets")
    require([row["案例ID"] for row in originals] == expected_ids,
            "Original checklist IDs/order differ from source datasets")
    changed = []
    labels = Counter()
    priority_ids = {priority: [] for priority in PRIORITIES}
    disputes = Counter()
    for row, original, (group, case, facts, evidence) in zip(rows, originals, cases):
        case_id = case["id"]
        prefix = f"{case_id}: "
        for field in review_checks.FIELDS:
            require(row[field] == original[field], prefix + f"Original column changed: {field}")
        expected_source = {
            "数据集": group,
            "待核查主张": case["claim"],
            "实际输入证据原文": evidence,
            "原始标签": case["expected_label"],
            "原始标注理由": case["rationale"],
        }
        for field, expected in expected_source.items():
            require(row[field] == expected, prefix + f"Source dataset/facts mismatch: {field}")
        label = row["建议人工复核标签"]
        require(label in LABELS, prefix + "Invalid suggested label")
        require(len(row["建议人工复核理由"].strip()) >= 30,
                prefix + "Suggested reason must contain at least 30 characters")
        try:
            quotes = json.loads(row["证据关键片段（JSON）"])
        except json.JSONDecodeError as exc:
            raise ValueError(prefix + "Evidence fragments are not valid JSON") from exc
        require(isinstance(quotes, list), prefix + "Evidence fragments must be a JSON array")
        if case_id == "N006":
            require(not facts and quotes == [], prefix + "Empty evidence must use []")
        else:
            require(bool(facts) and bool(quotes), prefix + "Evidence fragments must be nonempty")
        for quote in quotes:
            require(isinstance(quote, str) and bool(quote.strip()),
                    prefix + "Each evidence fragment must be nonempty text")
            require(any(quote in fact for fact in facts),
                    prefix + f"Fragment is not an exact substring of one actual fact: {quote!r}")
        dispute_type = row["本轮争议类型"]
        priority = row["复核优先级"]
        require(dispute_type in DISPUTE_TYPES, prefix + "Invalid dispute type")
        require(priority in PRIORITIES, prefix + "Invalid review priority")
        require(bool(row["需人工裁定项"].strip()), prefix + "Missing human adjudication item")
        require(row["建议来源与状态"] == AI_STATUS, prefix + "AI-only status differs")
        require(row["建议日期"] == DATE, prefix + "Suggestion date differs")
        labels[label] += 1
        disputes[dispute_type] += 1
        priority_ids[priority].append(case_id)
        if label != case["expected_label"]:
            changed.append({"id": case_id, "original_label": case["expected_label"],
                            "suggested_label": label})
    return {
        "cases": len(rows),
        "labels": {label: labels[label] for label in LABELS},
        "changed_label_count": len(changed),
        "changed_labels": changed,
        "priority_ids": priority_ids,
        "dispute_counts": {value: disputes[value] for value in DISPUTE_TYPES},
        "human_labels_filled": sum(bool(row["人工最终标签"].strip()) for row in rows),
        "human_comments_filled": sum(bool(row["人工复核意见"].strip()) for row in rows),
        "ai_suggestions_only": True,
        "human_review_completed": False,
        "snapshot_validation": manifest["snapshot_validation"],
        "validation_scope": "结构、来源、哈希和逐字片段检查；不证明语义或医学正确性",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", nargs="?", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--strict-snapshot", action="store_true",
                        help="Also require the original runtime and test source hashes")
    args = parser.parse_args(argv)
    print(json.dumps(validate_suggestions(args.csv_path, strict_snapshot=args.strict_snapshot),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
