"""Offline provenance gates for provisional labels and explanation review."""

import csv
from pathlib import Path

import pytest

from evaluations import explanation_fidelity, review_checks


def test_review_checklist_has_all_32_source_cases_and_empty_human_fields():
    result = review_checks.validate_checklist()
    assert result == {"cases": 32, "disputed": 12, "human_labels_filled": 0}
    rows = review_checks.expected_rows()
    assert len({row["案例ID"] for row in rows}) == 32
    assert all(row["人工最终标签"] == row["人工复核意见"] == "" for row in rows)
    assert all(row["建议标签"] == row["原始标签"] for row in rows)
    assert next(row for row in rows if row["案例ID"] == "N006")["实际输入证据原文"] == "（无输入证据）"
    priority = {f"U{i:03d}" for i in range(1, 7)} | {"H-U01", "H-U02"}
    assert all(next(row for row in rows if row["案例ID"] == case_id)["是否存在争议"] == "是"
               for case_id in priority)


def test_checklist_rejects_changed_evidence_or_agent_entry_in_human_column(tmp_path):
    rows = review_checks.expected_rows()
    rows[0]["实际输入证据原文"] = "未经输入的事实"
    path = tmp_path / "changed.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=review_checks.FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="differs"):
        review_checks.validate_checklist(path)
    rows = review_checks.expected_rows()
    rows[0]["人工最终标签"] = "supported"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=review_checks.FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="human fields"):
        review_checks.validate_checklist(path)


def test_fidelity_candidates_flag_h_n02_added_benefit_without_scoring_it():
    claim = "只要每周锻炼两天，就能保证每个人长寿。"
    facts = ["成年人每周还应安排至少2天涉及主要肌群的肌肉力量训练。"]
    reason = "证据只给出训练频率，但也说对成年人有益；主张保证长寿缺乏支持。"
    flags = explanation_fidelity.candidate_additions(claim, facts, reason)
    assert "效果词:有益" in flags
    assert "效果词:保证" not in flags  # This word came from the claim, not a new fact.
    assert explanation_fidelity.candidate_additions(claim, facts, "证据只给出训练频率。") == []


def test_archived_fidelity_queue_keeps_three_axes_separate_and_human_score_blank():
    result = explanation_fidelity.validate_queue()
    assert result == {"rows": 16, "candidate_rows": 2, "human_scores_filled": 0}
    with Path(explanation_fidelity.DEFAULT_QUEUE).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    h_n02_v2 = next(row for row in rows if row["案例ID"] == "H-N02" and row["策略"] == "v2")
    assert h_n02_v2["分类与初标一致"] == "是"
    assert h_n02_v2["引用状态"] == "无引用"
    assert "效果词:有益" in h_n02_v2["候选新增内容（仅提示）"]
    assert h_n02_v2["人工解释忠实度"] == ""
    h_s01_v1 = next(row for row in rows if row["案例ID"] == "H-S01" and row["策略"] == "v1")
    assert "数字:1.50分钟" in h_s01_v1["候选新增内容（仅提示）"]
