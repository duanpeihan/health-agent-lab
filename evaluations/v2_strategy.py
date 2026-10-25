"""Experimental verifier. Kept outside app so production imports remain V1."""

import json

from ollama import chat

from app.claim_verifier import MODEL
from .dataset import LABELS


CHECK_SCHEMA_V2 = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": list(LABELS)},
        "reason": {"type": "string"},
        "numeric_check": {"type": "string", "enum": ["none", "compatible", "unsupported", "direct_conflict", "ambiguous"]},
        "quantifier_check": {"type": "string", "enum": ["none", "compatible", "unsupported", "direct_conflict", "ambiguous"]},
        "evidence_ids": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["verdict", "reason", "numeric_check", "quantifier_check", "evidence_ids"],
    "additionalProperties": False,
}


SYSTEM_PROMPT_V2 = (
    "你是严格的给定证据主张核查器。只依据用户提供的事实，不用外部医学知识。"
    "先核对主张和证据的人群、时间、单位、数字、上下界、量词、否定、条件及建议/观察的性质，再分类。"
    "supported：所给事实直接蕴含主张的所有关键限定，可接受保留原范围的同义改写。"
    "contradicted：在相同对象和条件下，证据与主张不能同时为真；明确否定、互斥的确切数值或区间、"
    "明确的‘而不是’替换、全部相同对不同、应做对不应做，都属直接冲突。"
    "not_supported：证据缺少关键内容，或主张扩大人群、加强数值/量词、增加疗效或因果而推不出，"
    "但两者可能同时为真。尤其注意：证据‘通常建议至少7小时’与主张‘必须12小时’并非逻辑互斥，"
    "因为12满足至少7；主张只是把下界和建议强度擅自提高，应判not_supported，数字和量词检查均不能写direct_conflict。"
    "只有证据明确排除12小时、明确给出互斥的唯一值，或主张明确用‘而不是至少7小时’否定原建议时才是contradicted。"
    "uncertain：证据本身明确保留了与主张有关的未决版本、歧义指代、量词范围、数据性质或相互冲突的解释，"
    "且不同解释会改变结论。若证据明确写两稿待选而最终稿未定，主张断言选了其中一稿，应判uncertain；"
    "若‘成年人’在证据中明确可能指全部也可能仅指部分，主张说‘所有成年人’，应判uncertain。"
    "一个解释可能支持主张不等于证据已经支持主张；证据明确未决的情况优先判uncertain。"
    "只有证据根本没有涉及该主张的关键属性、也没有相关的未决信息时才用not_supported。"
    "数字检查和量词检查分别写入numeric_check、quantifier_check：none/compatible/unsupported/direct_conflict/ambiguous。"
    "先检查数值和量词，再给最终verdict；direct_conflict仅在两者不可能同时成立时使用并判contradicted，"
    "ambiguous且影响结论应判uncertain，"
    "除非另一条确切证据已经消除歧义。"
    "evidence_ids为引用的证据编号数组，从1开始；supported和contradicted必须引用直接证据；"
    "uncertain必须引用产生歧义的证据；not_supported可引用相关但不足的证据，也可为空。"
    "没有证据时返回not_supported、空引用。不要引用不存在的编号，也不要编造证据原文。"
)


def verify_claim_v2(claim, documents):
    if not isinstance(claim, str) or not claim.strip():
        raise ValueError("claim must be nonempty text")
    facts = []
    for doc in documents:
        for index, fact in enumerate(doc["facts"], 1):
            facts.append({"id": len(facts) + 1, "document_id": doc.get("id"),
                          "fact_index": index, "quote": fact})
    if not facts:
        return {"verdict": "not_supported", "reason": "没有可用于判断该主张的证据",
                "evidence_quote": "", "evidence_references": [],
                "numeric_check": "none", "quantifier_check": "none"}
    evidence_text = "\n".join(f"[{item['id']}] {item['quote']}" for item in facts)
    response = chat(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT_V2},
                  {"role": "user", "content": f"证据列表：\n{evidence_text}\n\n待核查主张：{claim}"}],
        format=CHECK_SCHEMA_V2, think=False, options={"temperature": 0},
    )
    result = json.loads(response.message.content or "{}")
    verdict, reason = result["verdict"], result["reason"]
    ids = result["evidence_ids"]
    if verdict not in LABELS or not isinstance(reason, str) or not reason.strip():
        raise ValueError("invalid V2 verdict or reason")
    if not isinstance(ids, list) or any(type(i) is not int or not 1 <= i <= len(facts) for i in ids):
        raise ValueError("V2 returned invalid evidence IDs")
    if len(ids) != len(set(ids)) or (verdict in ("supported", "contradicted", "uncertain") and not ids):
        raise ValueError("V2 returned missing or duplicate evidence IDs")
    checks = (result["numeric_check"], result["quantifier_check"])
    valid_checks = {"none", "compatible", "unsupported", "direct_conflict", "ambiguous"}
    if any(check not in valid_checks for check in checks):
        raise ValueError("invalid V2 conflict check")
    if "direct_conflict" in checks and verdict != "contradicted":
        raise ValueError("V2 conflict check contradicts verdict")
    references = [facts[i - 1] for i in ids]
    return {"verdict": verdict, "reason": reason,
            "evidence_quote": references[0]["quote"] if references else "",
            "evidence_references": references,
            "numeric_check": checks[0], "quantifier_check": checks[1]}
