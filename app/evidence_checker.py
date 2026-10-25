"""Bounded, clause-aware checks against the facts actually supplied to a writer.

Statement roles are conservative text rules, not an independent medical judge.
Only narrowly recognised evidence-limit statements and writing metadata skip
entailment classification. Every other substantive clause is checked or causes
a coverage failure; no unchecked tail can silently become ``completed``.
"""

import re

from .content_requirements import assess_content
from .evidence_quantities import check_quantity_fidelity
from .task_parameters import check_task_parameter_fidelity


MAX_MEDICAL_CLAIMS = 6
LABELS = {"supported", "not_supported", "contradicted", "uncertain"}
SAFE_STOP = "本次内容未通过证据核查，已停止输出未经核查的正文。"
DISCOURSE = re.compile(r"^(?:(?:但是|然而|不过|因此|所以|同时|此外|另外|而且|并且|且|故|亦|也|但)[：:\s]*)+")
# Only sentence punctuation creates a verification unit. Commas/conjunctions
# are used below to assess whether the *entire* sentence is safe to exclude;
# they never turn a subject/predicate into independently verified fragments.
SENTENCE_BREAK = re.compile(r"[。！？!?\n；;]")
ROLE_PART_BREAK = re.compile(
    r"，|(?<!\d),(?!\d)|(?:并不妨碍|并不影响|我确定|可以确定|"
    r"但是|然而|不过|此外|另外|而且|并且|同时|因此|所以|"
    r"这说明|这证明|这意味着|这表明|这证实|结果表明|结果显示|由此证明|由此可见|事实证明|"
    r"事实上|其实|可是|反而|相反|但|却|且|而)"
)
INDEPENDENT_MEDICAL_SUBJECT = re.compile(
    r"^(?:\d|成年人|成人|老年人|儿童|青少年|孕妇|患者|人们|人体|健康睡眠|睡眠|"
    r"睡前|牛奶|拉伸|运动|规律作息|这种饮品|这种方法|该药|此药|药物|该治疗|该干预|它)"
)
ASSERTION_PREDICATE = re.compile(
    r"治愈|治疗|预防|降低|提高|提升|改善|有益|保证|能够|能让|可以让|导致|维持|有助于|"
    r"每晚|每周|每天|有效|可使|会|是|能|必须|建议|需要|需|造成|影响|存在|改变|达到|减少|增加|消失"
)
MEDICAL_PREDICATE = re.compile(
    r"治愈|治疗|预防|降低|提高|提升|改善|有益|保证|能够|能让|可以让|导致|维持|有助于|每晚|每周|每天|有效"
)


def _strip_lead(text):
    return DISCOURSE.sub("", text.strip(" \t\r\n，,：:"))


def _is_meta(text):
    """Allow a narrow writing-only vocabulary, never just a '本文' prefix."""
    if re.fullmatch(r"[（(]?(?:字数[：:]?)?\s*\d+\s*字[）)]?", text):
        return True
    if re.fullmatch(r"(?:不进行|不提供|亦不提供|并不提供|也不提供)(?:任何)?(?:个体化|个人|个性化)?(?:健康|医学|医疗)?(?:(?:诊断|治疗|诊断或治疗)(?:建议|判断)?|建议)", text):
        return True
    if re.fullmatch(r"(?:仅供|只供)(?:一般)?(?:健康)?(?:科普|信息)(?:参考)?", text):
        return True
    if re.fullmatch(r"不(?:能)?(?:代替|替代)(?:专业)?(?:医疗|医学|诊疗)(?:建议|诊断|判断)", text):
        return True
    if re.fullmatch(r"(?:关于|有关|围绕)(?:成年人|成人)?(?:健康)?睡眠(?:的)?(?:科普|资料|信息|说明)?", text):
        return True
    if re.fullmatch(r"(?:健康科普)?写作(?:必须|应当|应)?(?:严格)?遵循可验证事实", text):
        return True
    if re.fullmatch(r"(?:避免|不得|不要|不应)(?:自行)?(?:补充|编造|添加)(?:任何)?(?:医学)?(?:数字|数据)(?:(?:、|或|和)(?:年龄范围|效果|医学数字|数据))*", text):
        return True
    if re.fullmatch(r"(?:本文|本篇|本回答|本说明|本科普|文章|这篇(?:睡眠)?说明|以上内容|此处)?(?:未|不)(?:添加|引入|补充|编造)(?:任何|其他|额外的?)?(?:未经证实的?)?(?:医学|干预)?(?:数据|数字|年龄范围|效果|结论|信息)(?:(?:、|或|和)(?:医学数字|数据|年龄范围|效果|结论))*", text):
        return True
    if re.fullmatch(r"(?:本文|本篇|本回答|本说明|本科普|文章|这篇(?:睡眠)?说明|以上内容|此处)(?:仅|只)?(?:限于|限|说明|总结)(?:所提供的?|给定的?|当前的?)?(?:资料|证据|信息|范围)(?:范围)?", text):
        return True
    if not MEDICAL_PREDICATE.search(text):
        if re.fullmatch(r"(?:本文|本篇|本回答|本说明|本科普|文章|这篇(?:睡眠)?说明|以上内容)(?:严格)?(?:依据|根据|基于).{1,60}(?:事实|资料|信息|证据)(?:整理|撰写)?", text):
            return True
        if re.fullmatch(r"(?:关注|保留)(?:资料允许表达的范围|年龄范围和不同条件|证据范围)", text):
            return True
        if re.fullmatch(r"建议(?:读者)?(?:参考|查阅)(?:原始|公开|所提供的?|给定的?|权威|相关)?(?:资料|信息|来源)", text):
            return True
        if re.fullmatch(r"建议(?:读者)?咨询(?:合格的?|专业的?)?(?:医疗|健康)?专业人员", text):
            return True
    return False


def _is_evidence_limit(text):
    # A positive statement attributed to '本文说明' is deliberately absent here.
    if re.match(r"^(?:无法|不能|尚不能|未能|无从|难以)(?:据此|仅凭.{0,20}|基于.{0,30})?(?:确认|判断|证实|证明|推断|得出|支持|评估)", text):
        return True
    # “未提及” describes an absence in the supplied evidence, not an endorsed
    # effect. Whole-sentence role checks still inspect independent assertions;
    # quoted task numbers are exempt only inside this nonaffirmative scope.
    if re.match(r"^(?:现有|当前|给定|所提供的?|该|本次)?(?:资料|证据|材料|信息|记录)(?:中)?(?:并未|尚未|未|没有|不曾)(?:提供|涉及|包含|给出|讨论|提到|提及|说明)", text):
        return True
    if re.match(r"^(?:没有|尚无|缺乏|未有)(?:足够的?|充分的?|相关的?)?(?:证据|资料|数据)(?:来)?(?:表明|证明|支持|确认|显示|证实)", text):
        return True
    # A source's inability to *deny/refute* an effect is not evidence that the
    # effect holds. Only explicit missing-data/confirmation predicates qualify.
    if re.match(
        r"^(?:现有|当前|给定|所提供的?|该|本次)?(?:资料|证据|材料|信息)(?:中)?"
        r"(?:(?:缺乏|不足|不够)(?:$|(?:足够的?|充分的?|相关的?|具体的?)?"
        r"(?:研究|实验|证据|资料|数据|信息|记录|效果指标))|"
        r"(?:无法|不能|未能|不足以)(?:据此)?(?:确认|判断|证实|证明|推断|得出|支持|评估))", text
    ):
        return True
    if re.match(r"^(?:不支持|不足以支持|不足以确认|不能据此得出)(?:任何)?(?:量化|具体|上述|该|此)?(?:结论|效果)", text):
        return True
    return False


def classify_segments(draft):
    """Return complete sentences and conservatively assess their roles.

    A sentence is exempt only when every nonempty role-assessment part is a
    narrowly recognised disclaimer or writing note. A clear safe introduction
    followed by a new independent assertion may be separated to retain numeric
    scope; otherwise a mixed sentence is sent in full for entailment, preserving
    conjunctions, negation, shared subjects, and all affirmative assertions.
    """
    clean = re.sub(r"\[\d+\]", "", draft)
    result = []
    for sentence in SENTENCE_BREAK.split(clean):
        text = sentence.strip()
        if not text:
            continue
        units = [text]
        # An explicit safe introduction may share a sentence with a new full
        # assertion. Keep quoted task numbers in its negative scope, while
        # verifying the new assertion. Never split a normal predicate such as
        # “健康睡眠需同时满足...” merely because it contains a conjunction.
        for boundary in ROLE_PART_BREAK.finditer(text):
            prefix = text[:boundary.start()].strip(" ，,")
            suffix = _strip_lead(text[boundary.end():])
            prefix_kind, _ = _sentence_role(prefix)
            suffix_kind, _ = _sentence_role(suffix)
            if (prefix_kind in {"meta", "evidence_limit"}
                    and suffix_kind == "medical_claim"
                    and INDEPENDENT_MEDICAL_SUBJECT.match(suffix)
                    and ASSERTION_PREDICATE.search(suffix)):
                units = [prefix, suffix]
                break
        for unit in units:
            kind, reason = _sentence_role(unit)
            result.append({"index": len(result), "text": unit, "kind": kind,
                           "reason": reason, "source_sentence": text})
    return result


def _sentence_role(text):
    parts = [_strip_lead(part) for part in ROLE_PART_BREAK.split(text)]
    parts = [part for part in parts if part]
    roles = ["meta" if _is_meta(part) else "evidence_limit" if _is_evidence_limit(part)
             else "medical_claim" for part in parts]
    if roles and all(role == "meta" for role in roles):
        return "meta", "whole_sentence_contains_only_narrow_writing_metadata"
    if roles and all(role in {"meta", "evidence_limit"} for role in roles):
        return "evidence_limit", "whole_sentence_contains_only_explicit_nonaffirmation"
    return "medical_claim", "whole_substantive_or_mixed_sentence_requires_evidence"


def _fact_key(text):
    # Punctuation at the sentence boundary is the only normalisation allowed.
    return text.strip().rstrip("。！？!?；;").strip()


def check_evidence(state, verify_fn):
    """Return graph-compatible state updates and complete coverage diagnostics."""
    draft, evidence = state["draft"], state["evidence"]
    segments = classify_segments(draft)
    facts = [(doc.get("id"), fact) for doc in evidence for fact in doc.get("facts", [])]
    claims = [segment for segment in segments if segment["kind"] == "medical_claim"]
    issues, codes, checked = [], [], []
    content = assess_content(state.get("task", ""), draft, evidence, segments)

    def issue(code, message):
        if message not in issues:
            codes.append(code)
            issues.append(message)

    for code, message in zip(content["issue_codes"], content["issues"]):
        issue(code, message)
    cited = {int(number) for number in re.findall(r"\[(\d+)\]", draft)}
    if not cited:
        issue("missing_citation", "文章缺少资料引用")
    elif not cited.issubset(set(range(1, len(evidence) + 1))):
        issue("invalid_citation", "文章使用了无效引用编号")
    if not facts:
        issue("no_input_facts", "没有可用于核查正文的输入事实")
    if not claims and not any(item["kind"] == "evidence_limit" for item in segments):
        issue("no_substantive_content", "正文只有元描述，缺少实质内容或明确的证据不足说明")
    if len(claims) > MAX_MEDICAL_CLAIMS:
        issue("claim_coverage_limit", f"正文包含{len(claims)}条待核查事实，超过单轮{MAX_MEDICAL_CLAIMS}条上限，不能确认全文通过")

    selected = claims[:MAX_MEDICAL_CLAIMS] if facts else []
    for segment in selected:
        claim = segment["text"]
        quantities = check_quantity_fidelity(claim, evidence)
        for quantity_issue in quantities:
            issue(quantity_issue["code"], quantity_issue["message"])
        item = {"claim": claim, "segment_index": segment["index"],
                "method": None, "verification": None, "error": None,
                "quantity_issues": quantities}
        # Clause splitting must not turn a verbatim whole input fact containing
        # commas into an unnecessary model request. Only an exact *whole source
        # sentence* qualifies; a substring under an added negation does not.
        exact = [(doc_id, fact) for doc_id, fact in facts
                 if _fact_key(fact) in {_fact_key(claim), _fact_key(segment["source_sentence"])}]
        try:
            if exact:
                item["method"] = "exact_fact"
                verification = {"verdict": "supported", "reason": "整条输入事实的逐字复用，仅忽略句末标点",
                                "evidence_quote": exact[0][1], "document_id": exact[0][0]}
            else:
                item["method"] = "claim_verifier"
                verification = verify_fn(claim, evidence)
            if not isinstance(verification, dict) or verification.get("verdict") not in LABELS:
                raise ValueError("核查器返回无效四分类结果")
            item["verification"] = verification
            if verification["verdict"] != "supported":
                issue("unsupported_claim", f"医学主张没有获得充分证据支持：{claim}（{verification['verdict']}）")
        except Exception as exc:
            item["error"] = {"type": type(exc).__name__, "message": str(exc)}
            issue("verifier_error", f"医学主张核查失败，不能确认支持：{claim}（{type(exc).__name__}）")
        checked.append(item)
        if item["error"]:
            # A failed verifier cannot establish coverage; do not spend the
            # remaining model budget on repeated timeouts in this same round.
            break

    sources = "\n".join(f"[{index}] {doc.get('source', '')}：{doc.get('url', '')}"
                        for index, doc in enumerate(evidence, 1))
    answer_candidate = (draft + "\n\n参考资料：\n" + sources
                        + "\n\n注：仅供一般健康科普参考，不代替专业医疗建议。")
    # Task hypotheses remain unverified. This independent publication gate only
    # checks that parameters *when restated* have not changed, even in a safe
    # evidence-limit sentence; omissions need no invented numeric restatement.
    parameter_check_error = False
    try:
        parameter_check = check_task_parameter_fidelity(state.get("task", ""), answer_candidate)
        if (not isinstance(parameter_check, dict) or type(parameter_check.get("passed")) is not bool
                or not isinstance(parameter_check.get("issues"), list)
                or (not parameter_check["passed"] and not parameter_check["issues"])
                or any(not isinstance(item, dict) or not item.get("code") or not item.get("message")
                       for item in parameter_check["issues"])):
            raise ValueError("任务参数检查返回无效结果")
    except Exception as exc:
        parameter_check_error = True
        parameter_check = {"passed": False, "status": "check_error", "issues": [{
            "code": "task_parameter_check_error",
            "message": f"任务参数保真检查失败，不能确认发布安全：{type(exc).__name__}",
        }], "error": {"type": type(exc).__name__, "message": str(exc)}}
    for parameter_issue in parameter_check["issues"]:
        issue(parameter_issue["code"], parameter_issue["message"])

    unchecked = [segment["text"] for segment in claims[len(checked):]]
    details = {
        "segments": segments, "claims": checked, "unchecked_claims": unchecked,
        "complete": (not unchecked and not any(item["error"] for item in checked)
                     and not parameter_check_error and bool(facts)),
        "medical_claim_count": len(claims), "max_medical_claims": MAX_MEDICAL_CLAIMS,
        "issue_codes": codes, "content_assessment": content,
        "task_parameter_fidelity": parameter_check,
        "coverage_scope": "All classified substantive clauses checked or blocked; role classification is heuristic and not medical validation",
    }
    if facts and content["status"] == "insufficient_content" and content["terminal"]:
        return {"issues": issues, "status": "insufficient_content", "checker_details": details,
                "answer": "当前资料或内容不足以完整满足任务，请补充可核查资料或缩小篇幅要求。"}
    if issues:
        return {"issues": issues, "status": "needs_revision", "answer": "", "checker_details": details}
    return {"issues": [], "status": "completed", "checker_details": details,
            "answer": answer_candidate}
