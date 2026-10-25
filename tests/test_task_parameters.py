"""Task references are copied faithfully without becoming medical evidence."""

import pytest

from app.task_parameters import check_task_parameter_fidelity as check


TASK = "请依据睡眠资料评估睡前拉伸20分钟能否让所有人的入睡时间减半；资料不足时明确说明。"


@pytest.mark.parametrize("answer", [
    "资料未提及睡前拉伸20分钟对入睡时间的影响，因此无法确认睡前拉伸20分钟的效果。",
    "尚无证据表明拉伸20分钟有效。",
    "无法确认拉伸２０分钟有效。",
    "无法确认伸展二十分钟有效。",
    "无法确认拉伸20.0分钟有效。",
    "无法确认20分钟拉伸有效。",
    "现有资料不足，无法确认该效果。",
    "无法确认拉伸是否有效。",
    "18至60岁成年人通常建议每晚睡眠至少7小时。",
    "参考[20] https://example.test/20/2.0；资料不足，无法确认该效果。",
])
def test_safe_exact_restatement_or_omission_passes(answer):
    result = check(TASK, answer)
    assert result["passed"]
    assert result["evidence_support_assessed"] is False


@pytest.mark.parametrize("answer", [
    "资料未提及拉伸20分钟的效果，因此无法确认拉伸2.0分钟有效。",
    "无法确认拉伸2.0分钟有效。",
    "尚无证据表明拉伸20小时有效。",
    "资料未提及拉伸1200秒的效果。",
    "无法确认拉伸2:0分钟有效。",
    "无法确认拉伸2.0.0分钟有效。",
    "无法确认拉伸20有效。",
    "无法确认2.0分钟有效。",
])
def test_restatement_distortions_block_even_under_negation(answer):
    result = check(TASK, answer)
    assert not result["passed"]
    assert result["issues"]
    assert all(issue["code"].startswith("task_parameter_") for issue in result["issues"])


@pytest.mark.parametrize("old,new", [
    ("18至60岁", "18至64岁"),
    ("18至60岁", "18岁"),
    ("10至20分钟", "10至30分钟"),
    ("10至20分钟", "20分钟"),
    ("至少20分钟", "至多20分钟"),
])
def test_ranges_units_and_bounds_cannot_change(old, new):
    task = f"请评估{old}成年人拉伸的效果。" if "岁" in old else f"请评估拉伸{old}的效果。"
    answer = f"无法确认{new}成年人拉伸的效果。" if "岁" in old else f"无法确认拉伸{new}的效果。"
    assert not check(task, answer)["passed"]


@pytest.mark.parametrize("answer", [
    "无法确认18到60岁成年人拉伸的效果。",
    "无法确认１８—６０岁成年人拉伸的效果。",
])
def test_literal_range_notation_equivalence_is_allowed(answer):
    assert check("请评估18至60岁成年人拉伸的效果。", answer)["passed"]


def test_two_parameters_cannot_swap_numbers():
    task = "评估拉伸20分钟、散步10分钟的效果。"
    result = check(task, "无法确认拉伸10分钟、散步20分钟有效。")
    assert not result["passed"] and len(result["issues"]) == 2


def test_different_task_parameters_with_the_same_values_are_bound_to_units():
    result = check("评估拉伸20分钟和睡眠7小时的效果。", "无法确认拉伸7分钟和睡眠20小时有效。")
    assert not result["passed"]


def test_group_specific_parameters_cannot_swap_even_when_concepts_match():
    task = "A组拉伸20分钟，B组拉伸10分钟。"
    result = check(task, "无法确认A组拉伸10分钟和B组拉伸20分钟的效果。")
    assert not result["passed"]


def test_group_specific_parameters_can_be_repeated_in_one_legal_sentence():
    task = "A组拉伸20分钟，B组拉伸10分钟。"
    result = check(task, "无法确认A组拉伸20分钟和B组拉伸10分钟的效果。")
    assert result["passed"]


def test_ambiguous_generic_reference_with_multiple_task_quantities_blocks():
    result = check("评估拉伸20分钟、散步10分钟的效果。", "无法确认该20分钟安排的效果。")
    assert not result["passed"]
    assert result["issues"][0]["code"] == "task_parameter_ambiguous"


@pytest.mark.parametrize("answer", [
    "无法确认该方案20小时是否有效。",
    "无法确认上述方法2.0小时是否有效。",
])
def test_explicit_task_reference_cannot_change_the_unit(answer):
    assert not check(TASK, answer)["passed"]


def test_explicit_reference_to_multiple_parameters_cannot_guess_ownership():
    result = check("评估拉伸20分钟、散步10分钟的效果。", "无法确认该参数20小时的效果。")
    assert not result["passed"]
    assert result["issues"][0]["code"] == "task_parameter_ambiguous"


@pytest.mark.parametrize("answer", [
    "无法确认上述运动2.0分钟的效果。",
    "无法确认该锻炼20小时的效果。",
    "该活动2.0分钟能否让人入睡时间减半，资料不足。",
])
def test_generic_task_activity_reference_overrides_a_misleading_nearby_concept(answer):
    assert not check(TASK, answer)["passed"]


def test_generic_reference_can_preserve_or_omit_the_parameter():
    assert check(TASK, "无法确认上述运动20分钟有效。")['passed']
    assert check(TASK, "无法确认上述运动有效。")['passed']
    assert check(TASK, "无法确认上述运动20分钟有效，成年人每晚睡眠至少7小时。")['passed']
    assert check(TASK, "无法确认上述运动20分钟有效且成年人每晚睡眠至少7小时。")['passed']


@pytest.mark.parametrize("answer", ["无法确认按摩2分钟是否有效。", "按摩2分钟有效。"])
def test_an_identically_named_unknown_chinese_activity_cannot_change_its_quantity(answer):
    assert not check("请评估按摩20分钟的效果。", answer)["passed"]


def test_unknown_chinese_activities_keep_independent_literal_parameters():
    assert check("请评估按摩20分钟的效果。", "无法确认按摩20分钟有效，成年人每晚睡眠至少7小时。")['passed']
    assert not check("评估按摩20分钟、热敷10分钟。", "无法确认按摩10分钟、热敷20分钟有效。")['passed']


def test_no_numeric_task_does_not_claim_all_new_numbers_are_verified():
    result = check("请写睡眠科普。", "成年人每晚至少7小时。")
    assert result["passed"] and result["task_parameters"] == []
    assert result["other_answer_parameters"]


def test_editorial_length_and_citations_are_not_copied_medical_parameters():
    result = check("请写200字睡眠科普。", "18至60岁成年人建议每晚睡眠至少7小时[1]。")
    assert result["passed"] and result["task_parameters"] == []


def test_incompatible_inputs_fail_closed():
    assert not check(None, "资料不足")["passed"]
