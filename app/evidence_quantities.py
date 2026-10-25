"""Conservative quantity fidelity checks against the actual supplied facts.

This gate compares values *together* with their unit, range, bound, frequency and
local context. It never repairs a number or converts a medical recommendation.
Entailment classification still checks the complete medical statement.
"""

from decimal import Decimal
import re
import unicodedata


UNIT_ALIASES = {
    "小时": "小时", "分钟": "分钟", "秒": "秒", "岁": "岁",
    "毫克": "毫克", "微克": "微克", "克": "克", "公斤": "公斤", "千克": "公斤",
    "毫升": "毫升", "升": "升", "天": "天", "周": "周", "月": "月", "年": "年",
    "次": "次", "%": "%", "毫米汞柱": "毫米汞柱", "mmHg": "毫米汞柱",
    "摄氏度": "摄氏度", "°C": "摄氏度",
}
UNITS = "|".join(re.escape(value) for value in sorted(UNIT_ALIASES, key=len, reverse=True))
ARABIC = r"[-+]?\d+(?:\.\d+)?"
BOUNDS = {"至少": "min", "不少于": "min", "不低于": "min", "最低": "min",
          "至多": "max", "最多": "max", "不超过": "max", "不高于": "max",
          "超过": "gt", "大于": "gt", "低于": "lt", "少于": "lt", "小于": "lt"}
BOUND_PATTERN = "|".join(sorted(BOUNDS, key=len, reverse=True))
QUANTITY = re.compile(
    rf"(?P<bound>{BOUND_PATTERN})?\s*(?P<first>{ARABIC})"
    rf"(?:(?P<first_unit>{UNITS})?\s*(?:至|到|~|～|—|–|-)\s*(?P<second>{ARABIC}))?"
    rf"\s*(?P<unit>{UNITS})"
)
NUMBER = re.compile(ARABIC)
FREQUENCY = re.compile(r"每小时|每分钟|每星期|每晚|每夜|每周|每天|每日|每月|每年|每秒")
FREQUENCY_ALIASES = {"每日": "每天", "每夜": "每晚", "每星期": "每周"}
CLAUSE = re.compile(r"([。！？!?；;，,\n])")
NEW_SUBJECT = re.compile(r"^(?:成年人|成人|不同年龄人群|其他人群|另一组|老年人|儿童|青少年)")
DISTINCT_SUBJECT = re.compile(r"^(?:不同年龄人群|其他人群|另一组|老年人|儿童|青少年)")
BOUND_BEFORE_PREDICATE = re.compile(
    rf"(?P<bound>{BOUND_PATTERN})(?:睡眠?|休息|锻炼|运动|进行|安排|持续|摄入|服用)?\s*$"
)
MALFORMED = re.compile(r"\d\s*[:：]\s*\d|\d+(?:\.\d+){2,}|\d\s+\d")
CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
             "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
CN_SCALES = {"十": 10, "百": 100, "千": 1000, "万": 10000}
CN_NUMBER = r"[零〇一二两三四五六七八九十百千万]+"
DOMAINS = {"sleep": re.compile(r"睡眠|入睡|睡觉|睡前|睡后|睡眠时长"),
           "activity": re.compile(r"运动|锻炼|有氧|肌肉|力量训练|身体活动|训练")}


def _cn_integer(text):
    if all(char in CN_DIGITS for char in text):
        return str(int("".join(str(CN_DIGITS[char]) for char in text)))
    total = section = digit = 0
    for char in text:
        if char in CN_DIGITS:
            digit = CN_DIGITS[char]
        else:
            scale = CN_SCALES[char]
            if scale == 10000:
                total += (section + digit) * scale
                section = digit = 0
            else:
                section += (digit or 1) * scale
                digit = 0
    return str(total + section + digit)


def _normalise(text):
    # Full-width digits and conventional Chinese integer spelling are explicit
    # notation equivalences. Colons, decimals and unit magnitudes are untouched.
    text = unicodedata.normalize("NFKC", re.sub(r"\[\d+\]", "", text))
    text = re.sub(rf"百分之({CN_NUMBER})", lambda match: _cn_integer(match[1]) + "%", text)
    return re.sub(rf"({CN_NUMBER})(?=\s*(?:{UNITS}|至|到|~|～|—|–|-))",
                  lambda match: _cn_integer(match[1]), text)


def _canonical_number(value):
    number = Decimal(value)
    return format(number.normalize(), "f")


def _domains(text):
    return {domain for domain, pattern in DOMAINS.items() if pattern.search(text)}


def _near_intensity(text, start, end):
    after = re.search(r"中等强度|高强度", text[end:end + 16])
    if after:
        return after[0]
    before = list(re.finditer(r"中等强度|高强度", text[max(0, start - 16):start]))
    return before[-1][0] if before else None


def _extract(text, fallback_frequency=None):
    records = []
    for match in QUANTITY.finditer(text):
        preceding = list(FREQUENCY.finditer(text[:match.start()]))
        frequency = preceding[-1][0] if preceding else fallback_frequency
        frequency = FREQUENCY_ALIASES.get(frequency, frequency)
        unit = UNIT_ALIASES[match["unit"]]
        bound = match["bound"]
        if bound is None:
            # A small literal verb may separate a bound from its quantity, as
            # in 至少睡7小时. Do not treat arbitrary preceding prose as a bound.
            preceding_bound = BOUND_BEFORE_PREDICATE.search(text[max(0, match.start() - 18):match.start()])
            bound = preceding_bound["bound"] if preceding_bound else None
        records.append({
            "text": match[0].strip(), "values": tuple(_canonical_number(value)
                for value in (match["first"], match["second"]) if value is not None),
            "unit": unit, "bound": BOUNDS.get(bound, "exact"),
            "frequency": frequency,
            "intensity": None if unit == "岁" else _near_intensity(text, match.start(), match.end()),
            "first_unit_valid": match["first_unit"] is None or UNIT_ALIASES[match["first_unit"]] == unit,
            "span": match.span(),
        })
    return records


def _compatible(observed, expected):
    return (observed["values"] == expected["values"]
            and observed["unit"] == expected["unit"]
            and observed["bound"] == expected["bound"]
            and observed["first_unit_valid"]
            and (observed["frequency"] is None or observed["frequency"] == expected["frequency"])
            and (observed["intensity"] is None or observed["intensity"] == expected["intensity"]))


def check_quantity_fidelity(claim, evidence):
    """Return deterministic blocking issues; never return a corrected claim.

    Apply only to affirmative medical statements. Nonaffirmation/metadata role
    handling belongs to the caller, so quoted user numbers in a safe evidence-
    insufficiency statement do not accidentally become medical affirmations.
    """
    text = _normalise(claim)
    facts = []
    for document in evidence:
        for fact in document.get("facts", []):
            normalised = _normalise(fact)
            frequencies = set(FREQUENCY.findall(normalised))
            fallback = next(iter(frequencies)) if len(frequencies) == 1 else None
            facts.append({"text": fact, "normalised": normalised,
                          "domains": _domains(normalised + " " + document.get("topic", "")),
                          "quantities": _extract(normalised, fallback)})
    issues = []

    def add(code, message, observed, expected):
        issue = {"code": code, "message": message, "observed": observed, "expected": expected}
        if issue not in issues:
            issues.append(issue)

    if MALFORMED.search(text):
        add("malformed_quantity", f"正文出现畸形或含糊数字，不能猜测修正：{claim}",
            claim, [fact["text"] for fact in facts])
    active_age_candidates = None
    active_frequency = None
    asserted_domains = set()
    for clause in (part.strip() for part in CLAUSE.split(text)):
        if not clause:
            continue
        if clause in "。！？!?；;\n":
            active_age_candidates = None
            active_frequency = None
            asserted_domains = set()
            continue
        if clause in "，,":
            continue
        domains = _domains(clause)
        if DISTINCT_SUBJECT.match(clause) or (NEW_SUBJECT.match(clause) and domains and asserted_domains
                                               and domains.isdisjoint(asserted_domains)):
            # Repeating the generic 成年人 is not a new population. Reset only
            # for an explicit different subject/topic; otherwise a comma could
            # launder WHO's age range into a CDC sleep recommendation.
            active_age_candidates = None
            active_frequency = None
        frequencies = list(FREQUENCY.finditer(clause))
        if frequencies:
            active_frequency = FREQUENCY_ALIASES.get(frequencies[-1][0], frequencies[-1][0])
        quantities = _extract(clause, active_frequency)
        covered = set(index for item in quantities for index in range(*item["span"]))
        for number in NUMBER.finditer(clause):
            if not all(index in covered for index in range(*number.span())):
                add("quantity_unit_missing", f"数字缺少可核对的范围或单位，不能确认保真：{claim}",
                    number[0], [fact["text"] for fact in facts])
        candidate_facts = []
        age_candidates = []
        for quantity in quantities:
            candidates = {index for index, fact in enumerate(facts)
                          if (not domains or domains.issubset(fact["domains"]))
                          and any(_compatible(quantity, item) for item in fact["quantities"])}
            candidate_facts.append(candidates)
            if quantity["unit"] == "岁":
                age_candidates.append(candidates)
            if not candidates:
                add("quantity_mismatch", f"数字、范围、单位或建议上下界与输入事实不一致：{quantity['text']}（{claim}）",
                    {key: value for key, value in quantity.items() if key != "span"},
                    [fact["text"] for fact in facts])
        if len(candidate_facts) > 1 and all(candidate_facts) and not set.intersection(*candidate_facts):
            add("quantity_context_mismatch", f"同句年龄、数量或单位来自不同事实，不能拼接成原证据未支持的建议：{claim}",
                clause, [fact["text"] for fact in facts])
        if age_candidates:
            active_age_candidates = set.intersection(*age_candidates)
        elif active_age_candidates is not None:
            for candidates in candidate_facts:
                if candidates and not active_age_candidates.intersection(candidates):
                    add("quantity_context_mismatch", f"逗号后的数量建议不受前述年龄范围对应的同一输入事实支持：{claim}",
                        clause, [facts[index]["text"] for index in sorted(active_age_candidates)])
        if quantities and re.search(r"必须|一定要|强制|只能", clause):
            matching = set.union(*candidate_facts) if candidate_facts else set()
            if matching and not any(re.search(r"必须|一定要|强制|只能", facts[index]["normalised"]) for index in matching):
                add("quantity_modality_mismatch", f"正文将原输入建议改为强制数量要求：{claim}",
                    clause, [facts[index]["text"] for index in sorted(matching)])
        if domains:
            asserted_domains = domains
    return issues
