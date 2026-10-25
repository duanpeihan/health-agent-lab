"""Quantity binding regressions use fixed input facts; no model calls."""

from copy import deepcopy

import pytest

from app.evidence_quantities import check_quantity_fidelity
from app.knowledge import load_knowledge_base


DOCUMENTS = load_knowledge_base()
SLEEP = [document for document in DOCUMENTS if document["id"] == "CDC001"]
ACTIVITY = [document for document in DOCUMENTS if document["id"] == "WHO001"]


@pytest.mark.parametrize("document", DOCUMENTS)
def test_all_original_facts_keep_bound_range_unit_and_context(document):
    for fact in document["facts"]:
        assert check_quantity_fidelity(fact, [document]) == []


@pytest.mark.parametrize("claim", [
    "18至6:0岁成年人每晚建议睡眠至少7小时。",
    "18至6：0岁成年人每晚建议睡眠至少7小时。",
    "18至6.0岁成年人每晚建议睡眠至少7小时。",
    "18至600岁成年人每晚建议睡眠至少7小时。",
    "18至60岁成年人每晚建议睡眠至少8小时。",
    "18至60岁成年人每晚建议睡眠至少7分钟。",
    "18至60岁成年人每晚建议睡眠至少7毫升。",
    "18至60岁成年人每周建议睡眠至少7小时。",
    "18至60岁成年人每小时建议睡眠至少7小时。",
    "18至60岁成年人每晚建议睡眠至多7小时。",
    "18至60岁成年人每晚建议睡眠7小时。",
    "18至60岁成年人每晚必须睡眠至少7小时。",
    "18至60岁成年人每晚建议睡眠至少7。",
])
def test_distorted_digits_units_frequency_bounds_and_modality_are_rejected(claim):
    result = check_quantity_fidelity(claim, SLEEP)
    assert result
    assert all(set(issue) >= {"code", "message", "observed", "expected"} for issue in result)


def test_colon_distortion_is_detected_without_changing_input():
    claim = "18至6:0岁成年人每晚建议睡眠至少7小时。"
    evidence = deepcopy(SLEEP)
    result = check_quantity_fidelity(claim, evidence)
    assert "malformed_quantity" in {issue["code"] for issue in result}
    assert claim == "18至6:0岁成年人每晚建议睡眠至少7小时。"
    assert evidence == SLEEP


@pytest.mark.parametrize("claim", [
    "18至64岁成年人每晚建议睡眠至少7小时。",
    "18至60岁成年人每周建议进行150至300分钟中等强度有氧活动。",
    "18至64岁成年人每周建议进行150至300分钟高强度有氧活动。",
    "18至64岁成年人每周建议进行75至300分钟高强度有氧活动。",
    "18至64岁成年人每天建议进行150至300分钟中等强度有氧活动。",
    "成年人每周安排至少2小时涉及主要肌群的肌肉力量训练。",
])
def test_numbers_cannot_be_rebound_from_other_facts_or_other_quantities(claim):
    assert check_quantity_fidelity(claim, DOCUMENTS)


@pytest.mark.parametrize("claim", [
    "18-60岁成年人通常建议每晚睡眠至少7小时。",
    "18岁至60岁成年人通常建议每晚睡眠至少7小时。",
    "１８至６０岁成年人通常建议每晚睡眠至少７小时。",
    "十八至六十岁成年人通常建议每晚睡眠至少七小时。",
    "18到60岁成年人通常建议每晚睡眠不少于7小时。",
])
def test_explicit_notation_equivalences_do_not_change_medical_magnitudes(claim):
    assert check_quantity_fidelity(claim, SLEEP) == []


def test_independent_source_backed_clauses_can_each_use_their_own_fact():
    claim = "18至60岁成年人通常建议每晚睡眠至少7小时，成年人每周还应安排至少2天涉及主要肌群的肌肉力量训练。"
    assert check_quantity_fidelity(claim, DOCUMENTS) == []


def test_no_numeric_claim_needs_no_numeric_gate():
    assert check_quantity_fidelity("健康睡眠需同时满足充足时长与良好质量。", SLEEP) == []


def test_range_endpoints_in_separate_facts_do_not_authorize_a_new_range():
    evidence = [{"facts": ["18至60岁成年人每晚建议睡眠至少7小时。", "64岁人群的睡眠建议另行说明。"]}]
    assert check_quantity_fidelity("18至64岁成年人每晚建议睡眠至少7小时。", evidence)


def test_same_value_different_unit_remains_a_difference():
    evidence = [{"facts": ["测试剂量建议至少7毫克。", "测试时长建议至少8分钟。"]}]
    assert check_quantity_fidelity("测试剂量建议至少7分钟。", evidence)


@pytest.mark.parametrize("claim", [
    "18至64岁成年人，每晚建议睡眠至少7小时。",
    "18至64岁成年人，他们每晚建议睡眠至少7小时。",
    "18至60岁成年人，每周建议进行150至300分钟中等强度有氧活动。",
])
def test_comma_does_not_let_a_quantity_escape_its_age_scope(claim):
    issues = check_quantity_fidelity(claim, DOCUMENTS)
    assert "quantity_context_mismatch" in {issue["code"] for issue in issues}


def test_source_backed_age_scope_survives_comma_without_new_subject():
    assert check_quantity_fidelity("18至60岁成年人，每晚通常建议睡眠至少7小时。", DOCUMENTS) == []


def test_notation_equivalent_frequency_is_not_a_change_of_schedule():
    assert check_quantity_fidelity("成年人每星期还应安排至少2天涉及主要肌群的肌肉力量训练。", ACTIVITY) == []


def test_repeating_generic_adults_does_not_reset_the_age_population():
    issues = check_quantity_fidelity("18至64岁成年人，成年人每晚建议睡眠至少7小时。", DOCUMENTS)
    assert "quantity_context_mismatch" in {issue["code"] for issue in issues}


def test_explicit_frequency_is_inherited_across_comma():
    assert check_quantity_fidelity("每天运动，150至300分钟中等强度有氧活动。", ACTIVITY)
    assert check_quantity_fidelity("每周运动，150至300分钟中等强度有氧活动。", ACTIVITY) == []


@pytest.mark.parametrize("claim", [
    "18至60岁成年人建议每晚至少睡7小时。",
    "18至60岁成年人建议每晚至少睡眠7小时。",
])
def test_bound_can_precede_a_literal_sleep_predicate(claim):
    assert check_quantity_fidelity(claim, SLEEP) == []
