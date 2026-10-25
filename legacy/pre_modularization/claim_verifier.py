
import json

from ollama import chat


MODEL = "qwen3:4b"


# 1. 加载健康知识库
with open("health_knowledge.json", encoding="utf-8") as f:
    knowledge_base = json.load(f)


# 2. 定义结构化输出格式
CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": [
                "supported",
                "not_supported",
                "contradicted",
                "uncertain"
            ]
        },
        "reason": {
            "type": "string"
        },
        "evidence_id": {
            "type": "integer"
        }
    },
    "required": [
        "verdict",
        "reason",
        "evidence_id"
    ],
    "additionalProperties": False
}


# 3. 医学主张核查函数
def verify_claim(claim, documents):

    # 提取全部证据事实
    facts = [
        fact
        for doc in documents
        for fact in doc["facts"]
    ]

    # 没有资料则不能确认主张
    if not facts:
        return {
            "verdict": "not_supported",
            "reason": "没有可用于支持该主张的证据",
            "evidence_quote": ""
        }

    # 给每条证据编号
    evidence_text = "\n".join(
        f"[{i}] {fact}"
        for i, fact in enumerate(facts, 1)
    )

    # 交给模型进行分类
    response = chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是严格的医学科普证据核查助手。"
                    "只能根据提供的证据判断主张。"
                    "不允许使用外部医学知识。"

                    "分类规则："
                    "supported：证据直接支持主张，"
                    "包括含义相同但措辞不同的表达。"

                    "not_supported：证据没有提到主张，"
                    "或者无法从证据推出该主张。"

                    "contradicted：证据明确反驳主张，"
                    "双方存在直接逻辑矛盾。"

                    "uncertain：证据含糊，"
                    "无法可靠判断。"

                    "注意语义等价："
                    "例如证据说不仅需要A，也需要B，"
                    "主张说需要A和B，属于supported。"

                    "证据说至少7小时，"
                    "主张说必须12小时，"
                    "属于not_supported。"

                    "证据没有提到牛奶治疗失眠，"
                    "主张说牛奶能治愈失眠，"
                    "属于not_supported。"

                    "请返回evidence_id。"
                    "当分类是supported或contradicted时，"
                    "evidence_id填写最直接的证据编号，"
                    "从1开始。"

                    "当分类是not_supported或uncertain时，"
                    "evidence_id填写0。"

                    "不要自行编造证据编号。"
                )
            },
            {
                "role": "user",
                "content": (
                    f"证据列表：\n{evidence_text}\n\n"
                    f"待核查主张：{claim}"
                )
            }
        ],
        format=CHECK_SCHEMA,
        think=False,
        options={"temperature": 0}
    )

    result = json.loads(
        response.message.content or "{}"
    )

    verdict = result["verdict"]
    reason = result["reason"]
    evidence_id = result["evidence_id"]

    # 4. Python验证证据编号
    if verdict in ["supported", "contradicted"]:

        # 证据编号不合法，不能直接通过
        if (
            type(evidence_id) is not int
            or evidence_id < 1
            or evidence_id > len(facts)
        ):
            return {
                "verdict": "uncertain",
                "reason": "模型没有提供有效证据编号",
                "evidence_quote": ""
            }

        # Python直接提取知识库原文
        evidence_quote = facts[evidence_id - 1]

    else:
        evidence_quote = ""

    # 5. 返回最终核查结果
    return {
        "verdict": verdict,
        "reason": reason,
        "evidence_quote": evidence_quote
    }


# 6. 测试
if __name__ == "__main__":

    documents = [
        doc for doc in knowledge_base
        if doc["topic"] == "成年人睡眠"
    ]

    test_claims = [
        "18至60岁成年人每晚建议至少睡7小时。",
        "健康睡眠需要足够时长，也需要良好的睡眠质量。",
        "成年人每晚必须睡12小时。",
        "睡前喝牛奶能够治愈失眠。"
    ]

    for claim in test_claims:

        print("\n========== 医学主张核查 ==========")
        print("待核查：", claim)

        result = verify_claim(claim, documents)

        print("核查结论：", result["verdict"])
        print("判断依据：", result["reason"])
        print("引用证据：", result["evidence_quote"])
