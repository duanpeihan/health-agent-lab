"""Build an offline candidate queue; never assign an automatic fidelity score."""

import argparse
import csv
import json
from pathlib import Path
import re

from .public_fixtures import public_fixture_path


ROOT = Path(__file__).resolve().parent
DEFAULT_REPORT = public_fixture_path("holdout_paired")
DEFAULT_QUEUE = ROOT / "explanation_fidelity_review.csv"
FIELDS = (
    "案例ID", "策略", "待核查主张", "实际输入证据原文", "模型理由",
    "分类与初标一致", "引用状态", "候选新增内容（仅提示）",
    "人工解释忠实度", "人工问题片段", "人工证据定位", "人工审核意见",
)
EFFECT_TERMS = ("有益", "改善", "治愈", "治疗", "降低风险", "提高", "保证", "显著改善")
NUMBER_WITH_UNIT = re.compile(r"\d+(?:[.:]\d+)?(?:%|小时|分钟|天|岁|毫克)")


def candidate_additions(claim, facts, reason):
    """Lexical warning only; absence is not evidence of unfaithfulness."""
    source = claim + "\n" + "\n".join(facts)
    candidates = [f"效果词:{term}" for term in EFFECT_TERMS if term in reason and term not in source]
    candidates.extend(f"数字:{token}" for token in dict.fromkeys(NUMBER_WITH_UNIT.findall(reason))
                      if token not in source)
    return candidates


def build_queue(report):
    if report.get("execution_mode") != "real_model" or report.get("case_count") != 8:
        raise ValueError("expected the completed real-model eight-case paired report")
    rows = []
    for strategy in ("v1", "v2"):
        for row in report["strategies"][strategy]["rows"]:
            facts = [fact for doc in row["documents"] for fact in doc["facts"]]
            citation = row["citation"]
            citation_status = ("有效" if citation["valid"] is True else
                               "无效" if citation["valid"] is False else "无引用")
            rows.append(dict(zip(FIELDS, (
                row["id"], strategy, row["claim"],
                "\n".join(f"[{doc['id']}] {fact}" for doc in row["documents"] for fact in doc["facts"]),
                row["reason"], "是" if row["predicted_label"] == row["expected_label"] else "否",
                citation_status, "；".join(candidate_additions(row["claim"], facts, row["reason"])),
                "", "", "", "",
            ))))
    if len(rows) != 16:
        raise ValueError("expected exactly 16 archived model reasons")
    return rows


def validate_queue(path=DEFAULT_QUEUE, report_path=DEFAULT_REPORT):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError("fidelity review queue columns differ")
        actual = list(reader)
    expected = build_queue(json.loads(Path(report_path).read_text(encoding="utf-8")))
    if actual != expected:
        raise ValueError("fidelity review queue differs from archived report or human fields are filled")
    return {"rows": len(actual), "candidate_rows": sum(bool(row["候选新增内容（仅提示）"]) for row in actual),
            "human_scores_filled": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Create queue once; refuse to overwrite")
    args = parser.parse_args(argv)
    if args.write:
        report = json.loads(DEFAULT_REPORT.read_text(encoding="utf-8"))
        with DEFAULT_QUEUE.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(build_queue(report))
    print(validate_queue())


if __name__ == "__main__":
    main()
