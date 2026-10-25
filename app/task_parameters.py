"""Check copies of task quantities without treating the task as evidence.

Omission is allowed. Literal quantity copies are bound to a local concept and,
where present, a group/population. A matched copy must retain its value, range,
unit and bound. This is a bounded text gate, not medical entailment or general
semantic equivalence; the ordinary Evidence Checker remains independent.
"""

import re

from .evidence_quantities import MALFORMED, NUMBER, _extract, _normalise


CONCEPTS = {
    "sleep_onset": r"入睡时间|入睡时长|入睡潜伏期",
    "stretch": r"拉伸|伸展",
    "walk": r"散步|步行",
    "run": r"跑步|慢跑",
    "swim": r"游泳",
    "cycle": r"骑行|骑车",
    "strength": r"力量训练|抗阻训练",
    "aerobic": r"有氧活动|有氧运动",
    "sleep": r"睡眠时长|睡眠时间|睡眠|睡觉|睡多久|睡几小时",
    "exercise": r"运动|锻炼|身体活动",
    "milk": r"牛奶",
    "dose": r"剂量|服用|用药|摄入",
    "blood_pressure": r"血压",
    "temperature": r"体温|温度",
}
ACTORS = re.compile(r"[A-Za-z甲乙丙丁]组|干预组|对照组|成年人|成人|儿童|青少年|老年人|孕妇|"
                    r"(?:药物?|药品)[A-Za-z甲乙丙丁]")
ACTOR_ALIAS = {"成人": "成年人"}
SENTENCES = re.compile(r"[。！？!?；;\n，,]")
REFERENCE = re.compile(r"https?://[^\s]+|\[\d+\]")
EDITORIAL_NUMBER = re.compile(r"\s*(?:字|词|段|篇|条|个(?:案例|场景))")
DEICTIC = re.compile(r"该|上述|这种|这项|这一|所述|提问|问题|无法|不能|资料|证据|确认")
TASK_REFERENCE = re.compile(
    r"(?:该|上述|这种|这项|这一|所述)(?:方案|方法|安排|干预|参数|做法|数值|时长|用量|剂量|活动|运动|锻炼)|"
    r"(?:用户|问题|任务|提问)(?:中|里)?(?:的|所述的?)?(?:参数|数字|数值|时长|用量|剂量)"
)
NON_ANCHOR = {"资料", "证据", "材料", "信息", "确认", "判断", "说明", "建议", "评估", "效果",
              "不足", "时长", "时间", "分钟", "小时", "人数", "年龄", "每次", "持续", "每晚", "至少", "至多"}
LIMITATIONS = [
    "仅核对可解析的显式数字、范围、单位及上下界；不证明医学事实或任务假设成立。",
    "参数归属使用局部概念与主体规则，不是通用指代或语义解析；减半等隐含数量不在本门禁范围。",
    "合法省略参数可通过；未识别为任务复述的其他医学数值仍由原有证据门禁核查。",
]


def _clean(text):
    text = REFERENCE.sub("", text)
    text = re.sub(r"(?m)^\s*\d+[.)、]\s*", "", text)
    text = re.sub(r"(?:任务|问题|测试)\s*\d+\s*[：:]", "", text)
    return _normalise(text)


def _nearest(matches, start, end):
    """Prefer a nearby governing concept, including postfix '20分钟拉伸'."""
    candidates = []
    for label, match in matches:
        if match.end() <= start:
            gap, side = start - match.end(), 0
        elif match.start() >= end:
            gap, side = match.start() - end, 1
        else:
            continue
        if gap <= 24:
            candidates.append((gap, side, -len(match[0]), label, match))
    return min(candidates, key=lambda item: item[:3])[-2:] if candidates else (None, None)


def _role(clause, quantity):
    start, end = quantity["span"]
    matches = [(label, match) for label, pattern in CONCEPTS.items()
               for match in re.finditer(pattern, clause)]
    concept, _ = _nearest(matches, start, end)
    if quantity["unit"] == "岁":
        concept = "age"
    actors = [(ACTOR_ALIAS.get(match[0], match[0]), match) for match in ACTORS.finditer(clause)]
    preceding_actors = [(label, match) for label, match in actors if 0 <= start - match.end() <= 24]
    if preceding_actors and quantity["unit"] != "岁":
        actor = preceding_actors[-1][0]
    else:
        actor, _ = _nearest(actors, start, end)
    if concept is None:
        # An adjacent named medicine/group or a literal repeated noun gives a
        # bounded fallback. Discourse words are deliberately not noun anchors.
        prefix = clause[max(0, start - 10):start]
        noun = re.search(r"(?:药物?|药品)[A-Za-z甲乙丙丁]$|[A-Za-z]{1,12}$", prefix)
        if noun:
            concept = "literal:" + noun[0]
        else:
            prefix = re.sub(r"(?:(?:每次|持续|总计|大约|至少|至多|不少于|不超过|时长|时间|约|为|是|的))+$", "", prefix)
            prefix = re.sub(r"(?:(?:资料|证据|材料|信息|当前|现有|无法|不能|尚无|没有|确认|判断|说明|建议|"
                            r"评估|提及|提到|提供|该|上述|这种|这项|这一|所述|请|未|不))+$", "", prefix)
            chinese = re.search(r"[\u4e00-\u9fff]{2,}$", prefix)
            suffix = chinese[0][-2:] if chinese else None
            concept = "literal:" + suffix if suffix and suffix not in NON_ANCHOR else None
    return concept, actor


def _task_reference(clause, quantity):
    start, _ = quantity["span"]
    for match in TASK_REFERENCE.finditer(clause):
        if not 0 <= start - match.end() <= 12:
            continue
        gap = clause[match.end():start]
        # Do not let a reference to an earlier quantity capture a later,
        # independently stated CDC parameter in the same sentence.
        if NUMBER.search(gap) or ACTORS.search(gap):
            continue
        if any(re.search(pattern, gap) for pattern in CONCEPTS.values()):
            continue
        return True
    return False


def _parameters(text):
    result = []
    for clause in SENTENCES.split(_clean(text)):
        if not clause.strip():
            continue
        quantities = _extract(clause)
        covered = {index for quantity in quantities for index in range(*quantity["span"])}
        for match in NUMBER.finditer(clause):
            if any(index in covered for index in range(*match.span())):
                continue
            if EDITORIAL_NUMBER.match(clause[match.end():]):
                continue
            # Bare/malformed copied numbers must not disappear just because
            # their unit was deleted. No missing unit is inferred or repaired.
            quantities.append({"text": match[0], "values": (_normalise(match[0]),), "unit": None,
                               "bound": "exact", "first_unit_valid": True, "span": match.span()})
        for quantity in sorted(quantities, key=lambda value: value["span"]):
            concept, actor = _role(clause, quantity)
            result.append({"text": quantity["text"], "values": list(quantity["values"]),
                           "unit": quantity["unit"], "bound": quantity["bound"],
                           "first_unit_valid": quantity["first_unit_valid"],
                           "concept": concept, "actor": actor, "clause": clause,
                           "task_reference": _task_reference(clause, quantity),
                           "malformed_clause": bool(MALFORMED.search(clause))})
    return result


def _signature(parameter):
    return (tuple(parameter["values"]), parameter["unit"], parameter["bound"])


def _candidates(observed, expected):
    # A local explicit reference can change unit or be assigned an unrelated
    # nearby concept; neither is a reason to skip the task-copy check.
    if observed["task_reference"]:
        return list(enumerate(expected)), False
    exact = [(index, item) for index, item in enumerate(expected)
             if observed["concept"] is not None and item["concept"] == observed["concept"]]
    if observed["actor"]:
        actor_matches = [(index, item) for index, item in exact if item["actor"] == observed["actor"]]
        if actor_matches:
            return actor_matches, False
        # A known *different* actor is not a license to copy another actor's
        # quantity. Leave the concept candidates for conservative rejection.
        if exact and any(item["actor"] for _, item in exact):
            return exact, True
    if exact:
        return exact, False
    if observed["concept"] is None:
        # A bare nonaffirmative copy can still be related by the same unit.
        # Explicit task references were already handled above across units.
        by_unit = [(index, item) for index, item in enumerate(expected)
                   if observed["unit"] == item["unit"] or observed["unit"] is None]
        if by_unit and DEICTIC.search(observed["clause"]):
            return by_unit, False
    return [], False


def check_task_parameter_fidelity(task, answer):
    """Return diagnostics only; never replace numbers or certify a hypothesis."""
    if not isinstance(task, str) or not isinstance(answer, str):
        return {"passed": False, "status": "uncertain", "issues": [{
            "code": "task_parameter_invalid_input", "message": "任务参数检查输入无效，不能确认发布保真。"}],
            "task_parameters": [], "checked_copies": [], "omitted_parameters": [],
            "limitations": list(LIMITATIONS), "evidence_support_assessed": False}
    expected, observed = _parameters(task), _parameters(answer)
    issues, checked, used, unrelated = [], [], set(), []
    for item in observed:
        candidates, actor_conflict = _candidates(item, expected)
        if not candidates:
            unrelated.append(item)
            continue
        unique = {_signature(candidate) for _, candidate in candidates}
        ambiguous = len(unique) > 1
        match = (not actor_conflict and not ambiguous and not item["malformed_clause"]
                 and item["first_unit_valid"] and item["unit"] is not None
                 and any(_signature(item) == _signature(candidate) for _, candidate in candidates))
        checked.append({"observed": item, "expected": [candidate for _, candidate in candidates], "passed": match})
        if match:
            used.update(index for index, candidate in candidates if _signature(item) == _signature(candidate))
            continue
        code = "task_parameter_ambiguous" if ambiguous or actor_conflict else "task_parameter_mismatch"
        issues.append({"code": code,
                       "message": f"任务参数复述不一致或归属不明：{item['text']}；须保留原任务数值、范围、单位或省略该参数，不能猜改。",
                       "observed": item, "expected": [candidate for _, candidate in candidates]})
    return {"passed": not issues, "status": "passed" if not issues else "failed",
            "issues": issues, "task_parameters": expected, "checked_copies": checked,
            "omitted_parameters": [item for index, item in enumerate(expected) if index not in used],
            "other_answer_parameters": unrelated, "limitations": list(LIMITATIONS),
            "evidence_support_assessed": False}
