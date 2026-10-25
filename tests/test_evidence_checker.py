"""Offline regressions for clause roles, safety gates and coverage."""

from copy import deepcopy
from unittest.mock import Mock

import pytest

from app.evidence_checker import MAX_MEDICAL_CLAIMS, check_evidence, classify_segments
from app.knowledge import load_knowledge_base


EVIDENCE = [doc for doc in load_knowledge_base() if doc["id"] == "CDC001"]
FACTS = EVIDENCE[0]["facts"]
BASE = "。".join(fact.rstrip("。") for fact in FACTS) + "。[1]"


def state(text, task="请依据睡眠资料评估睡前拉伸20分钟能否让所有人的入睡时间减半。"):
    return {"draft": text, "evidence": deepcopy(EVIDENCE), "task": task}


def verdict(value):
    return {"verdict": value, "reason": "离线固定分类，不是医学判断", "evidence_quote": ""}


def test_whole_input_facts_are_supported_without_model_call():
    verifier = Mock(side_effect=AssertionError("Exact facts must not call a model"))
    result = check_evidence(state(BASE), verifier)
    assert result["status"] == "completed"
    assert result["checker_details"]["complete"] is True
    assert len(result["checker_details"]["claims"]) == 3
    assert all(item["method"] == "exact_fact" for item in result["checker_details"]["claims"])
    assert verifier.call_count == 0


@pytest.mark.parametrize("text", [
    "无法确认睡前拉伸20分钟能让入睡时间减半",
    "因此，无法基于现有信息确认睡前拉伸20分钟是否能将所有人的入睡时间减半",
    "没有证据表明睡前喝牛奶能够治愈失眠",
    "该资料未包含任何关于睡前拉伸活动、入睡时间变化或其效果的证据",
    "现有资料中缺乏相关研究数据、实验结果或具体效果指标",
])
def test_negative_evidence_statements_do_not_assert_the_quoted_effect(text):
    verifier = Mock(side_effect=AssertionError("Nonaffirmations must not undergo entailment"))
    result = check_evidence(state(BASE + text + "。"), verifier)
    assert result["status"] == "completed"
    assert verifier.call_count == 0
    assert "unsupported_number" not in result["checker_details"]["issue_codes"]
    assert all(item["kind"] == "evidence_limit" for item in classify_segments(text))


@pytest.mark.parametrize("text", [
    "本文未添加未经证实的医学数据",
    "本科普严格依据CDC提供的信息，未添加任何未经证实的医学数据、年龄范围或效果，亦不提供个体化健康建议",
    "健康科普写作必须严格遵循可验证事实，避免自行补充医学数字、年龄范围或效果",
    "（248字）",
])
def test_pure_writing_metadata_does_not_undergo_medical_entailment(text):
    verifier = Mock(side_effect=AssertionError("Pure metadata must not call a model"))
    result = check_evidence(state(BASE + text + "。"), verifier)
    assert result["status"] == "completed"
    assert verifier.call_count == 0
    assert all(item["kind"] == "meta" for item in classify_segments(text))


@pytest.mark.parametrize("text", [
    "睡前喝牛奶能够治愈失眠",
    "本文说明牛奶能治愈失眠",
    "无法确认拉伸效果，但牛奶能治愈失眠",
    "本文未添加未经证实的医学数据，但是牛奶能治愈失眠",
    "无法确认拉伸效果，牛奶能治愈失眠",
    "无法确认拉伸效果但是牛奶能治愈失眠",
    "无法确认拉伸效果但牛奶能治愈失眠",
    "无法确认拉伸效果并且牛奶能治愈失眠",
    "无法确认拉伸效果所以牛奶能治愈失眠",
    "无法确认拉伸效果其实牛奶能治愈失眠",
    "现有资料不能否认睡前喝牛奶能够治愈失眠",
    "现有资料无法推翻睡前喝牛奶能够治愈失眠",
    "现有资料不足以反驳睡前喝牛奶能够治愈失眠",
    "现有资料不能排除睡前喝牛奶能够治愈失眠",
    "规律作息和避免睡前电子设备使用可以提升睡眠质量",
    "健康睡眠是维持成年人身体机能的基础",
])
def test_affirmative_medical_clauses_cannot_hide_behind_disclaimers(text):
    verifier = Mock(return_value=verdict("not_supported"))
    result = check_evidence(state(BASE + text + "。"), verifier)
    assert result["status"] == "needs_revision"
    assert verifier.call_count >= 1
    sent = [call.args[0] for call in verifier.call_args_list]
    assert any(item["kind"] == "medical_claim" for item in classify_segments(text))
    assert any("牛奶" in claim or "规律作息" in claim or "身体机能" in claim for claim in sent)


def test_numbers_in_new_affirmative_claims_are_not_exempted_by_the_user_task():
    verifier = Mock(return_value=verdict("not_supported"))
    result = check_evidence(state(BASE + "睡前拉伸20分钟能让所有人的入睡时间减半。"), verifier)
    assert result["status"] == "needs_revision"
    assert "quantity_mismatch" in result["checker_details"]["issue_codes"]
    assert "unsupported_claim" in result["checker_details"]["issue_codes"]


def test_only_complete_verbatim_input_facts_get_the_exact_fact_shortcut():
    text = BASE + "并非18至60岁成年人通常建议每晚睡眠至少7小时。"
    verifier = Mock(return_value=verdict("contradicted"))
    result = check_evidence(state(text), verifier)
    assert result["status"] == "needs_revision"
    assert verifier.call_count == 1
    assert verifier.call_args.args[0].startswith("并非")


@pytest.mark.parametrize("text", [
    "关于睡眠是身体机能的基础",
    "本文说明规律作息和避免睡前电子设备能提升睡眠质量",
    "本文未添加未经证实的医学数据且牛奶能治愈失眠",
])
def test_metadata_labels_do_not_whitelist_fact_assertions(text):
    assert any(segment["kind"] == "medical_claim" for segment in classify_segments(text))


def test_more_than_six_medical_clauses_fail_coverage_instead_of_silently_passing():
    claims = [f"医学主张测试{index}需要验证" for index in range(MAX_MEDICAL_CLAIMS + 1)]
    verifier = Mock(return_value=verdict("supported"))
    result = check_evidence(state("。".join(claims) + "。[1]"), verifier)
    assert verifier.call_count == MAX_MEDICAL_CLAIMS
    assert result["status"] == "needs_revision"
    assert result["checker_details"]["complete"] is False
    assert result["checker_details"]["unchecked_claims"] == [claims[-1]]
    assert "claim_coverage_limit" in result["checker_details"]["issue_codes"]


def test_all_medical_clauses_beyond_old_top_three_are_checked():
    text = BASE + "正文声称规律作息能改善健康。"
    verifier = Mock(return_value=verdict("not_supported"))
    result = check_evidence(state(text), verifier)
    assert result["status"] == "needs_revision"
    assert len(result["checker_details"]["claims"]) == 4
    assert verifier.call_args.args[0] == "正文声称规律作息能改善健康"


@pytest.mark.parametrize("bad", [{"verdict": "ok"}, {}, None])
def test_invalid_verifier_outputs_fail_closed(bad):
    result = check_evidence(state(BASE + "牛奶能治愈失眠。"), Mock(return_value=bad))
    assert result["status"] == "needs_revision"
    assert result["answer"] == ""
    assert result["checker_details"]["complete"] is False
    assert "verifier_error" in result["checker_details"]["issue_codes"]


def test_verifier_timeout_is_recorded_and_cannot_complete():
    result = check_evidence(state(BASE + "牛奶能治愈失眠。"), Mock(side_effect=TimeoutError("fixture timeout")))
    assert result["status"] == "needs_revision"
    assert result["checker_details"]["claims"][-1]["error"]["type"] == "TimeoutError"
    assert result["checker_details"]["complete"] is False


def test_citation_safety_is_preserved_for_metadata_and_nonaffirmations():
    verifier = Mock()
    text = BASE.replace("[1]", "[9]") + "无法确认拉伸20分钟有效。"
    result = check_evidence(state(text), verifier)
    assert result["status"] == "needs_revision"
    assert "invalid_citation" in result["checker_details"]["issue_codes"]
    result = check_evidence(state(BASE.replace("[1]", "")), verifier)
    assert "missing_citation" in result["checker_details"]["issue_codes"]


def test_all_metadata_cannot_pass_as_an_article():
    text = "本文未添加未经证实的医学数据。" * 5 + "[1]"
    result = check_evidence(state(text), Mock())
    assert result["status"] == "needs_revision"
    assert "no_substantive_content" in result["checker_details"]["issue_codes"]


def test_empty_evidence_does_not_trigger_models_or_pass():
    data = state(BASE)
    data["evidence"] = []
    verifier = Mock(side_effect=AssertionError("No facts must not call model"))
    result = check_evidence(data, verifier)
    assert result["status"] == "needs_revision"
    assert result["checker_details"]["complete"] is False
    assert verifier.call_count == 0


@pytest.mark.parametrize("text", [
    "健康睡眠需同时满足充足时长与良好质量",
    "成年人保持充足睡眠时长，同时也需要关注良好睡眠质量",
    "健康睡眠不仅需要足够的睡眠时长，也需要良好的睡眠质量",
    "18至60岁成年人通常建议每晚睡眠至少7小时，健康睡眠同时需要足够时长以及良好质量",
])
def test_shared_subject_and_simultaneous_predicates_remain_a_complete_sentence(text):
    parts = classify_segments(text + "。[1]")
    assert len(parts) == 1
    assert parts[0]["text"] == text
    assert parts[0]["kind"] == "medical_claim"
    verifier = Mock(return_value=verdict("supported"))
    check_evidence(state(text + "。[1]"), verifier)
    if text != FACTS[-1].rstrip("。"):  # A complete verbatim input fact needs no model.
        verifier.assert_called_once_with(text, EVIDENCE)


@pytest.mark.parametrize("text", [
    "睡眠时长需要关注，但睡前喝牛奶能够治愈失眠",
    "无法确认拉伸效果，但是睡前喝牛奶能够治愈失眠",
    "无法确认拉伸效果同时睡前喝牛奶能够治愈失眠",
    "本文未添加未经证实的医学数据，睡前喝牛奶能够治愈失眠",
    "没有证据表明拉伸有效并且牛奶能治愈失眠",
    "当前资料不足以支持拉伸效果，却已经证明牛奶能治愈失眠",
    "本文未添加未经证实的医学数据且这种饮品可使失眠消失",
])
def test_a_later_medical_assertion_in_the_same_sentence_is_never_dropped(text):
    parts = classify_segments(text)
    claims = [part["text"] for part in parts if part["kind"] == "medical_claim"]
    assert claims
    assert any("牛奶" in claim or "这种饮品" in claim for claim in claims)
    assert all(part["source_sentence"] == text for part in parts)
    verifier = Mock(return_value=verdict("not_supported"))
    result = check_evidence(state(BASE + text + "。"), verifier)
    assert result["status"] == "needs_revision"
    assert [call.args[0] for call in verifier.call_args_list] == claims


@pytest.mark.parametrize("text", [
    "睡眠是一种身体过程，休息环境与个人感受有关，这种饮品可使失眠消失",
    "睡眠是一种身体过程，而这种饮品可使失眠消失",
    "这种饮品可使失眠消失且不会出现任何不良影响",
])
def test_long_sentences_and_unknown_wording_are_fully_verified(text):
    verifier = Mock(return_value=verdict("not_supported"))
    result = check_evidence(state(BASE + text + "。"), verifier)
    assert result["status"] == "needs_revision"
    verifier.assert_called_once_with(text, EVIDENCE)
    assert all("健康睡眠需" != item["claim"] for item in result["checker_details"]["claims"])


def test_sentence_punctuation_preserves_every_verification_unit_in_order():
    text = "健康睡眠需同时满足时长与质量。睡前喝牛奶能够治愈失眠；这种饮品可使疾病消失！"
    parts = classify_segments(text)
    assert [item["text"] for item in parts] == [
        "健康睡眠需同时满足时长与质量", "睡前喝牛奶能够治愈失眠", "这种饮品可使疾病消失"]
    assert all(item["kind"] == "medical_claim" for item in parts)


def test_a_pure_negative_sentence_with_shared_subject_remains_nonaffirmative():
    text = "无法确认睡前拉伸20分钟能让入睡时间减半，也不能据此证明该效果"
    parts = classify_segments(text)
    assert len(parts) == 1 and parts[0]["kind"] == "evidence_limit"
    verifier = Mock(side_effect=AssertionError("Pure nonaffirmation must not call a model"))
    result = check_evidence(state(BASE + text + "。"), verifier)
    assert result["status"] == "completed"
    verifier.assert_not_called()


@pytest.mark.parametrize("connector", ["，但", "但", "，同时", "并且", "，", "而", "可是", "反而"])
def test_task_numbers_stay_in_negative_scope_before_a_new_supported_assertion(connector):
    denied = "无法确认拉伸20分钟有效"
    asserted = "18至60岁成年人通常建议每晚睡眠至少7小时"
    parts = classify_segments(denied + connector + asserted)
    assert [(part["text"], part["kind"]) for part in parts] == [
        (denied, "evidence_limit"), (asserted, "medical_claim")]
    verifier = Mock(side_effect=AssertionError("The affirmative part is an exact input fact"))
    result = check_evidence(state(BASE + denied + connector + asserted + "。"), verifier)
    assert result["status"] == "completed"
    assert "unsupported_number" not in result["checker_details"]["issue_codes"]
    verifier.assert_not_called()


def test_clear_boundary_does_not_exempt_an_unsupported_affirmative_number():
    text = "无法确认拉伸20分钟有效，但18至60岁成年人每晚必须睡12小时"
    parts = classify_segments(text)
    assert parts[-1]["kind"] == "medical_claim"
    assert "12小时" in parts[-1]["text"] and "20分钟" not in parts[-1]["text"]
    verifier = Mock(return_value=verdict("not_supported"))
    result = check_evidence(state(BASE + text + "。"), verifier)
    assert result["status"] == "needs_revision"
    assert "quantity_mismatch" in result["checker_details"]["issue_codes"]
    verifier.assert_called_once_with(parts[-1]["text"], EVIDENCE)


@pytest.mark.parametrize("cue", [
    "这证明", "这意味着", "这表明", "这证实", "结果表明", "结果显示", "由此证明", "事实证明",
])
def test_a_new_affirmative_inference_after_a_negative_scope_is_never_exempt(cue):
    prefix = "无法确认拉伸20分钟有效"
    tail = "牛奶能治愈失眠"
    text = prefix + cue + tail
    parts = classify_segments(text)
    assert [(part["text"], part["kind"]) for part in parts] == [
        (prefix, "evidence_limit"), (tail, "medical_claim")]
    verifier = Mock(return_value=verdict("not_supported"))
    result = check_evidence(state(BASE + text + "。"), verifier)
    assert result["status"] == "needs_revision"
    assert "unsupported_claim" in result["checker_details"]["issue_codes"]
    assert not any(issue["code"] in {"malformed_quantity", "quantity_mismatch", "quantity_unit_missing"}
                   for item in result["checker_details"]["claims"] for issue in item.get("quantity_issues", []))
    verifier.assert_called_once_with(tail, EVIDENCE)


def test_a_metadata_prefix_cannot_hide_an_affirmative_inference():
    text = "本文未添加未经证实的医学数据这意味着这种饮品可使失眠消失"
    parts = classify_segments(text)
    assert parts[0]["kind"] == "meta"
    assert parts[1]["text"] == "这种饮品可使失眠消失" and parts[1]["kind"] == "medical_claim"
    verifier = Mock(return_value=verdict("not_supported"))
    result = check_evidence(state(BASE + text + "。"), verifier)
    assert result["status"] == "needs_revision"
    verifier.assert_called_once_with("这种饮品可使失眠消失", EVIDENCE)


def test_pure_twenty_minute_nonaffirmation_still_skips_quantity_affirmation_checks():
    text = "无法确认睡前拉伸20分钟能让入睡时间减半。"
    verifier = Mock(side_effect=AssertionError("A pure negative scope must not call the model"))
    result = check_evidence(state(BASE + text), verifier)
    assert result["status"] == "completed"
    assert classify_segments(text)[0]["kind"] == "evidence_limit"
    assert all(not item.get("quantity_issues") for item in result["checker_details"]["claims"])
    verifier.assert_not_called()
