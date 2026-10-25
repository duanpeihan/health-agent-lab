"""Task coverage regressions are offline rules, not medical quality claims."""

from app.content_requirements import assess_content, assess_evidence_capacity
from app.knowledge import load_knowledge_base


SLEEP = [doc for doc in load_knowledge_base() if doc["id"] == "CDC001"]
EXERCISE = [doc for doc in load_knowledge_base() if doc["id"] == "WHO001"]
TASK = "请依据给定资料写18至60岁成年人睡眠科普，说明推荐时长与睡眠质量，避免增加无证据结论。"
SHORT = "18至60岁成年人每晚建议睡眠时长至少7小时[1]。健康睡眠需同时满足充足时长与良好质量。"


def segments(text, kind="medical_claim"):
    return [{"text": text, "kind": kind}]


def test_short_answer_can_cover_all_explicit_sleep_requirements():
    assert len(SHORT) < 60
    result = assess_content(TASK, SHORT, SLEEP, segments(SHORT))
    assert result["status"] == "ok"
    assert result["missing"] == []
    assert set(result["required"]) == {"sleep_duration", "sleep_quality", "age_scope"}
    assert result["manual_review_required"] is True


def test_missing_quality_is_incomplete_even_if_padded_over_sixty_characters():
    text = "18至60岁成年人每晚建议至少睡7小时。" * 8 + "[1]"
    result = assess_content(TASK, text, SLEEP, segments(text))
    assert len(text) > 60 and result["status"] == "insufficient_content"
    assert result["terminal"] is False  # Available facts may be used in one bounded repair.
    assert result["missing"] == ["sleep_quality"]


def test_missing_duration_or_age_is_explicitly_recorded():
    result = assess_content(TASK, "健康睡眠需要良好质量。[1]", SLEEP, segments("健康睡眠需要良好质量"))
    assert result["status"] == "insufficient_content"
    assert set(result["missing"]) == {"sleep_duration", "age_scope"}


def test_age_scope_presence_does_not_claim_the_age_number_is_correct():
    text = SHORT.replace("60", "64")
    result = assess_content(TASK, text, SLEEP, segments(text))
    assert result["status"] == "ok"  # Numeric fidelity is independently mandatory.
    assert "主题出现不等于事实正确" in result["limitations"][1]


def test_safe_insufficient_evidence_answer_needs_no_minimum_character_count():
    text = "现有资料不足，无法确认拉伸20分钟能让入睡时间减半。[1]"
    result = assess_content("评估拉伸效果；资料不足则明确无法确认。", text, SLEEP, segments(text, "evidence_limit"))
    assert len(text) < 60 and result["status"] == "ok"
    assert result["assessment_scope"] == "explicit_evidence_limit_answer"


def test_a_disclaimer_cannot_replace_available_requested_sleep_topics():
    text = "无法确认。[1]"
    result = assess_content(TASK, text, SLEEP, segments(text, "evidence_limit"))
    assert result["status"] == "insufficient_content"
    assert set(result["missing"]) == {"sleep_duration", "sleep_quality", "age_scope"}


def test_metadata_alone_does_not_become_complete_by_repetition():
    text = "本文未添加未经证实的医学数据。" * 10
    result = assess_content("写睡眠科普", text, SLEEP, segments(text, "meta"))
    assert result["status"] == "insufficient_content"
    assert "no_substantive_content" in result["issue_codes"]


def test_placeholder_and_unfinished_predicate_are_incomplete():
    for text in ("内容太短", "正文", "健康睡眠需", "健康睡眠需要", "年龄范围与"):
        result = assess_content("写睡眠科普", text + "。[1]", SLEEP, segments(text))
        assert result["status"] == "insufficient_content"
        assert "incomplete_fragment" in result["issue_codes"]
        assert result["terminal"] is False


def test_unknown_task_exposes_manual_scope_without_claiming_general_completeness():
    result = assess_content("请说明一般信息", "这条信息已有来源。[1]", SLEEP, segments("这条信息已有来源"))
    assert result["status"] == "ok" and result["manual_review_required"]
    assert result["required"] == []
    assert "通用任务满足度仍需人工判断" in result["limitations"][-1]


def test_exercise_explicit_topics_are_checked():
    task = "说明18至64岁成年人有氧运动时长及力量训练安排。"
    complete = "18至64岁成年人每周进行150至300分钟有氧活动，并安排至少2天力量训练。[1]"
    result = assess_content(task, complete, EXERCISE, segments(complete))
    assert result["status"] == "ok" and not result["missing"]
    incomplete = assess_content(task, "成年人安排力量训练。[1]", EXERCISE, segments("成年人安排力量训练"))
    assert set(incomplete["missing"]) == {"activity_duration", "age_scope"}


def test_hard_writer_budget_refuses_impossible_length_before_generation():
    for task in ("写至少1000字睡眠科普", "写约1000字睡眠科普", "写一千字睡眠科普", "写800至1000字睡眠科普"):
        result = assess_evidence_capacity(task, SLEEP)
        assert result["status"] == "insufficient_content"
        assert result["terminal"] is True
        assert result["issue_codes"] == ["requested_length_exceeds_output_budget"]
        assert result["output_character_limit"] == 600


def test_maximum_length_is_not_misread_as_required_minimum():
    for task in ("写不超过1000字睡眠科普", "写1000字以内睡眠科普", "写最多1000字睡眠科普"):
        result = assess_evidence_capacity(task, SLEEP)
        assert result["status"] == "ok" and result["requested_length"] is None


def test_known_summary_sources_do_not_support_a_confirmed_long_form_capacity():
    result = assess_evidence_capacity("写一篇长篇成年人睡眠文章", SLEEP)
    assert result["status"] == "insufficient_content"
    assert result["terminal"] is True
    assert result["issue_codes"] == ["summary_evidence_insufficient_for_long_form"]
    assert result["evidence_fact_count"] == 3


def test_unknown_sources_do_not_invent_a_character_capacity():
    docs = [{"id": "CUSTOM", "facts": ["一条资料。"]}]
    result = assess_evidence_capacity("写一篇长篇文章", docs)
    assert result["status"] == "ok" and result["manual_review_required"]
    assert "未触发已知容量边界" in result["limitations"][-1]


def test_explicit_length_not_met_is_not_repaired_by_inventing_facts():
    result = assess_content("写300字睡眠科普，说明推荐时长和质量。", SHORT, SLEEP, segments(SHORT))
    assert result["status"] == "insufficient_content"
    assert "requested_length_not_met" in result["issue_codes"]
    assert result["terminal"] is True
    assert "不能靠添加无依据事实凑字数" in result["issues"][-1]


def test_current_r_normal_does_not_trigger_long_form_preflight():
    result = assess_evidence_capacity(TASK, SLEEP)
    assert result["status"] == "ok" and result["requested_length"] is None


def test_content_gate_does_not_modify_draft_or_evidence():
    import copy
    docs = copy.deepcopy(SLEEP)
    before = copy.deepcopy(docs)
    assess_content(TASK, SHORT, docs, segments(SHORT))
    assert docs == before and "6:0" not in SHORT


def test_compact_parallel_quality_statement_is_covered():
    text = "18至60岁成年人建议每晚至少睡7小时[1]。时长、质量缺一不可。"
    result = assess_content(TASK, text, SLEEP, segments(text))
    assert result["status"] == "ok" and result["missing"] == []


def test_evaluate_word_does_not_excuse_answering_available_explicit_topics():
    task = "请评估18至60岁成年人的睡眠需求，说明推荐时长与睡眠质量。"
    result = assess_content(task, "无法确认。[1]", SLEEP, segments("无法确认", "evidence_limit"))
    assert result["status"] == "insufficient_content"
    assert set(result["missing"]) == {"sleep_duration", "sleep_quality", "age_scope"}
