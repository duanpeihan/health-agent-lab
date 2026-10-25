"""Small, explicit content gates for the local sleep/exercise tasks.

The former 60-character threshold was inherited from the legacy agent's
``检查文章长度`` rule, without a documented completeness criterion. Character
count cannot establish factual or task completeness. These rules check explicit
task topics and obvious fragments, and expose their limited scope for review.
"""

import re

from .writer import MAX_ARTICLE_CHARS


LONG_FORM = re.compile(r"长篇|长文章|长文|完整专题|深度专题")
TASK_NUMBER = r"[0-9０-９零〇一二两三四五六七八九十百千万]+"
LENGTH_RANGE = re.compile(rf"({TASK_NUMBER})\s*(?:至|到|[-—~～])\s*({TASK_NUMBER})\s*字")
LENGTH_SINGLE = re.compile(rf"({TASK_NUMBER})\s*字")
KNOWN_SUMMARY_IDS = {"CDC001", "WHO001"}
LIMITATIONS = [
    "仅检查当前睡眠/运动任务的显式主题和明显残缺，不是通用内容完整性判定。",
    "主题出现不等于事实正确，所有实质主张仍须完整证据核查；内容质量待人工审核。",
]
FRAGMENT_END = re.compile(r"(?:需要|需|应当|应该|应|建议|至少|至多|以及|和|与|同时|因为|由于)$")
PLACEHOLDER = re.compile(r"^(?:内容(?:太|过)?短|正文|文章|待补充|略|不知道|待生成|暂无正文)$")
TITLE_ONLY = {"睡眠", "睡眠时长", "睡眠质量", "成年人睡眠", "关于睡眠", "运动", "有氧活动", "力量训练"}


def _integer(text):
    text = text.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    if text.isdigit():
        return int(text)
    digits = {character: value for value, character in enumerate("零一二三四五六七八九")}
    digits.update({"〇": 0, "两": 2})
    if all(character in digits for character in text):
        return int("".join(str(digits[character]) for character in text))
    total = section = current = 0
    for character in text:
        if character in digits:
            current = digits[character]
        elif character == "万":
            total += (section + current) * 10000
            section = current = 0
        elif character in "十百千":
            section += (current or 1) * {"十": 10, "百": 100, "千": 1000}[character]
            current = 0
        else:
            return None
    return total + section + current


def _length_request(task):
    match = LENGTH_RANGE.search(task)
    if match:
        return {"characters": _integer(match.group(1)), "kind": "range_lower_bound",
                "text": match.group(0)}
    for match in LENGTH_SINGLE.finditer(task):
        before, after = task[max(0, match.start() - 6):match.start()], task[match.end():match.end() + 3]
        if re.search(r"不超过|不要超过|至多|最多|不多于", before) or re.match(r"以内|以下|之内", after):
            continue
        approximate = bool(re.search(r"大约|大概|约", before) or re.match(r"左右|上下", after))
        return {"characters": _integer(match.group(1)),
                "kind": "approximate_target" if approximate else "requested_length",
                "text": match.group(0)}
    return None


def _base(task):
    return {"status": "ok", "terminal": False, "issues": [], "issue_codes": [], "required": [], "missing": [],
            "limitations": list(LIMITATIONS), "manual_review_required": True,
            "assessment_scope": "explicit_local_task_topics_and_obvious_fragments",
            "requested_length": _length_request(task), "long_form_requested": bool(LONG_FORM.search(task))}


def _issue(result, code, message, *, terminal=False):
    result["status"] = "insufficient_content"
    result["terminal"] = result["terminal"] or terminal
    result["issue_codes"].append(code)
    result["issues"].append(message)


def assess_evidence_capacity(task, evidence):
    """Preflight objective output limits and the known summary-only KB scope.

    No arbitrary evidence-character multiplier is used. Other length/topic
    combinations remain outside automatic capacity confirmation.
    """
    result = _base(task)
    facts = list(dict.fromkeys(fact for document in evidence for fact in document.get("facts", [])))
    result.update(evidence_fact_count=len(facts), output_character_limit=MAX_ARTICLE_CHARS,
                  capacity_scope="hard_output_budget_and_known_summary_sources")
    length = result["requested_length"]
    if length and length["characters"] > MAX_ARTICLE_CHARS:
        _issue(result, "requested_length_exceeds_output_budget",
               f"任务要求{length['characters']}字篇幅，超过当前正文{MAX_ARTICLE_CHARS}字符上限；内容不足，需调整篇幅或另行准备资料，不能强行扩写。", terminal=True)
    elif (result["long_form_requested"] and evidence
          and all(document.get("id") in KNOWN_SUMMARY_IDS for document in evidence)):
        _issue(result, "summary_evidence_insufficient_for_long_form",
               "任务明确要求长篇内容，但当前仅有CDC/WHO摘要事实，未建立长篇内容所需的证据覆盖；内容不足，请补充可核查资料或接受简短说明。", terminal=True)
    else:
        result["limitations"].append("未触发已知容量边界，不代表已自动证明任意篇幅均可由现有证据满足。")
    return result


def _required_topics(task):
    required = []
    if "睡眠" in task:
        if "时长" in task or "睡多久" in task or "睡几小时" in task:
            required.append("sleep_duration")
        if "质量" in task:
            required.append("sleep_quality")
        if "年龄" in task or "岁" in task:
            required.append("age_scope")
        if "不同年龄" in task:
            required.append("age_variation")
    elif any(word in task for word in ("运动", "锻炼", "有氧", "身体活动", "力量训练")):
        if "时长" in task or "多久" in task:
            required.append("activity_duration")
        if "力量" in task or "肌肉" in task:
            required.append("strength_training")
        if "年龄" in task or "岁" in task:
            required.append("age_scope")
    return required


def _has_topic(topic, text):
    if topic == "sleep_duration":
        return bool(re.search(r"(?:\d+(?:\.\d+)?|[一二三四五六七八九十两]+)\s*(?:小时|钟头)", text))
    if topic == "sleep_quality":
        return bool(re.search(r"(?:睡眠|休息|良好|足够|需要|需|与|和).{0,12}质量|"
                              r"(?:时长|时间)\s*[、与和及]\s*(?:睡眠)?质量|"
                              r"质量\s*[、与和及]\s*(?:睡眠)?(?:时长|时间)", text))
    if topic == "age_scope":
        return bool(re.search(r"\d+\s*(?:至|到|[-—~～])\s*\d+\s*岁|\d+\s*岁(?:以上|以下|及以上|及以下)", text))
    if topic == "age_variation":
        return bool(re.search(r"不同年龄|年龄.{0,12}(?:不同|差异)", text))
    if topic == "activity_duration":
        return bool(re.search(r"(?:\d+(?:\.\d+)?|[一二三四五六七八九十百两]+)\s*(?:分钟|小时)", text))
    if topic == "strength_training":
        return bool(re.search(r"力量训练|肌肉.{0,8}训练|肌群.{0,8}训练", text))
    return False


def assess_content(task, draft, evidence, segments):
    """Check explicit coverage instead of treating 60 characters as quality."""
    result = assess_evidence_capacity(task, evidence)
    result.update(draft_character_count=len(draft), body_character_count=len(re.sub(r"\[\d+\]|\s", "", draft)))
    if result["status"] == "insufficient_content":
        return result
    substantive = [segment for segment in segments if segment.get("kind") == "medical_claim"]
    limits = [segment for segment in segments if segment.get("kind") == "evidence_limit"]
    if not substantive and not limits:
        _issue(result, "no_substantive_content", "正文缺少实质内容或明确的证据不足说明。")
        return result
    for segment in substantive:
        text = segment.get("text", "").strip("。！？!?；; \t\n")
        if not text or PLACEHOLDER.fullmatch(text) or text in TITLE_ONLY or FRAGMENT_END.search(text):
            _issue(result, "incomplete_fragment", f"正文含占位文字或未完成陈述：{text or '空白'}")
    # A short, explicit non-answer is a legitimate result for an evidence-limit
    # question; it does not require filling space with unrelated facts.
    required = _required_topics(task)
    safe_limit_answer = (bool(limits) and not substantive and not required
                         and bool(re.search(r"能否|是否|评估|效果|疗效|资料不足|无法确认", task)))
    if safe_limit_answer:
        result["assessment_scope"] = "explicit_evidence_limit_answer"
        result["required"] = ["explicit_evidence_limit"]
        return result
    text = "。".join(segment.get("text", "") for segment in substantive)
    missing = [topic for topic in required if not _has_topic(topic, text)]
    result.update(required=required, missing=missing)
    if missing:
        _issue(result, "missing_requested_topics", "正文尚未覆盖任务明确要求的内容：" + "、".join(missing))
    length = result["requested_length"]
    if length and length["kind"] != "approximate_target" and result["body_character_count"] < length["characters"]:
        _issue(result, "requested_length_not_met",
               f"正文未达到任务要求的{length['characters']}字篇幅；内容不足，请补充可核查资料或调整为短摘要，不能靠添加无依据事实凑字数。", terminal=True)
    if not required:
        result["limitations"].append("此任务没有命中已实现的显式主题覆盖规则，通用任务满足度仍需人工判断。")
    return result
