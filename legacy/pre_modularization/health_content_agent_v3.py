
import json
import os
import re
import time
from typing import TypedDict
from claim_verifier import verify_claim

from ollama import chat
from langgraph.graph import StateGraph, START, END


MODEL = "qwen3:4b"
MAX_REVISIONS = 2

# 加载现有健康知识库
with open("health_knowledge.json", "r", encoding="utf-8") as f:
    knowledge_base = json.load(f)


# 1. 定义状态
class HealthState(TypedDict):
    task: str
    evidence: list[dict]
    draft: str
    issues: list[str]
    revision_count: int
    answer: str
    status: str


# 2. Retriever：检索资料
def retriever_node(state: HealthState):

    print("\n========== Retriever资料检索 ==========")

    evidence = [
        doc for doc in knowledge_base
        if any(word in state["task"] for word in doc["keywords"])
    ]

    if not evidence:
        return {
            "evidence": [],
            "status": "no_evidence",
            "answer": "知识库中没有找到相关资料，暂不生成内容。"
        }

    for doc in evidence:
        print("检索到：", doc["source"], doc["topic"])

    return {
        "evidence": evidence,
        "status": "retrieved"
    }


# 3. Writer：生成或修正文章
def writer_node(state: HealthState):

    print("\n========== Writer内容生成 ==========")

    evidence_text = ""

    for i, doc in enumerate(state["evidence"], 1):
        evidence_text += (
            f"\n资料[{i}]：{doc['source']}\n"
            + "\n".join(doc["facts"])
            + "\n"
        )

    instruction = (
        "你是严谨的健康科普写作助手。"
        "只能根据给定资料中的事实写文章。"
        "不能编造医学数字、年龄范围或效果。"
        "缺少证据时不要自行补充具体结论。"
        "在正文中使用[1]等编号引用来源。"
        "文章控制在200至300字。"
        "不进行个人诊断或治疗建议。"
    )

    user_content = (
        f"写作任务：{state['task']}\n"
        f"参考证据：{evidence_text}"
    )

    # 如果是修正轮次，加入之前的错误反馈
    if state["revision_count"] > 0:
        print("收到核查反馈，正在修正文章...")

        user_content += (
            f"\n上一版文章：{state['draft']}\n"
            f"发现的问题：{state['issues']}\n"
            "请修正上述问题，删除所有没有证据支持的数字。"
            "修正后直接输出完整文章。"
        )

    response = chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": user_content}
        ],
        think=False,
        options={"temperature": 0}
    )

    draft = (response.message.content or "").strip()
    draft = draft.split("</think>")[-1].strip()

    # 测试专用：故意插入一个错误医学数字
    if (
        os.getenv("INJECT_BAD_FACT", "0") == "1"
        and state["revision_count"] == 0
        and "睡眠" in state["task"]
    ):
        draft += "\n成年人每晚必须睡12小时。[1]"
        print("测试模式：已注入不受支持的12小时声明")

    if (
        os.getenv("INJECT_FALSE_CLAIM", "0") == "1"
        and state["revision_count"] == 0
    ):
        draft += "\n睡前喝牛奶能够治愈失眠。[1]"
        print("测试模式：注入无依据医学主张")

    print("文章初稿生成完成")

    return {
        "draft": draft,
        "status": "drafted"
    }


# 4. Evidence Checker：证据核查
def evidence_checker_node(state: HealthState):

    print("\n========== Evidence Checker ==========")

    draft = state["draft"]
    evidence = state["evidence"]
    issues = []

    # 检查文章长度
    if len(draft) < 60:
        issues.append("文章内容过短")

    # 检查引用编号是否有效
    cited_ids = {
        int(n) for n in re.findall(r"\[(\d+)\]", draft)
    }

    valid_ids = set(range(1, len(evidence) + 1))

    if not cited_ids:
        issues.append("文章缺少资料引用")

    elif not cited_ids.issubset(valid_ids):
        issues.append("文章使用了无效引用编号")

    # 从知识库事实中提取允许出现的数字
    fact_text = " ".join(
        fact
        for doc in evidence
        for fact in doc["facts"]
    )

    supported_numbers = set(
        re.findall(r"\d+(?:\.\d+)?", fact_text)
    )

    # 移除引用编号，避免把[1]当成医学数字
    clean_draft = re.sub(r"\[\d+\]", "", draft)

    # 检查带有健康数量单位的句子
    sentences = re.split(r"[。！？\n]", clean_draft)

    for sentence in sentences:

        health_units = ["小时", "分钟", "岁", "天", "周", "次", "%"]

        if not any(unit in sentence for unit in health_units):
            continue

        numbers = re.findall(r"\d+(?:\.\d+)?", sentence)

        for number in numbers:
            if number not in supported_numbers:
                issues.append(
                    f"数字 {number} 未出现在检索证据中："
                    f"{sentence.strip()}"
                )

    issues = list(dict.fromkeys(issues))



    # 医学主张核查：在数字和引用检查之外，
    # 对具有明显健康结论的句子进行证据验证
    sentences = [
        sentence.strip()
        for sentence in re.split(r"[。！？\n]", clean_draft)
        if sentence.strip()
    ]

    claim_keywords = [
        "小时", "分钟", "必须", "建议",
        "治愈", "预防", "降低风险", "提高"
    ]

    claims = [
        sentence for sentence in sentences
        if any(word in sentence for word in claim_keywords)
    ]

    # 优先检查强断言，限制模型调用数量
    claims.sort(
        key=lambda text: 0
        if ("必须" in text or "治愈" in text)
        else 1
    )

    for claim in claims[:3]:

        print("正在核查医学主张：", claim)

        try:
            verification = verify_claim(
                claim,
                evidence
            )

            verdict = verification["verdict"]

        except (ValueError, KeyError, TypeError, OSError) as e:
            verdict = "uncertain"
            print("主张核查发生异常：", e)

        print("证据核查结论：", verdict)

        if verdict != "supported":
            issues.append(
                f"医学主张没有获得充分证据支持："
                f"{claim}（{verdict}）"
            )

    # 核查通过
    if not issues:

        print("引用及数字核查通过")

        sources = "\n".join(
            f"[{i}] {doc['source']}：{doc['url']}"
            for i, doc in enumerate(evidence, 1)
        )

        answer = (
            draft
            + "\n\n参考资料：\n"
            + sources
            + "\n\n注：仅供一般健康科普参考，不代替专业医疗建议。"
        )

        return {
            "issues": [],
            "answer": answer,
            "status": "completed"
        }

    # 发现问题
    print("证据核查未通过：")

    for issue in issues:
        print("-", issue)

    return {
        "issues": issues,
        "status": "needs_revision"
    }


# 5. Reflection：记录反馈、允许重新生成
def reflection_node(state: HealthState):

    count = state["revision_count"] + 1

    print(f"\n========== Reflection第{count}次 ==========")

    print("根据证据核查反馈修正文章")

    return {
        "revision_count": count,
        "status": "revising"
    }


# 6. 修正失败保护
def stop_node(state: HealthState):

    return {
        "status": "evaluation_failed",
        "answer": (
            "文章经过多次修正仍未通过证据核查，"
            "因此不输出未经核查的科普内容。\n"
            "未解决的问题："
            + "；".join(state["issues"])
        )
    }


# 7. 条件路由
def route_retriever(state: HealthState):
    return "finish" if state["status"] == "no_evidence" else "write"


def route_checker(state: HealthState):

    if state["status"] == "completed":
        return "finish"

    if state["revision_count"] >= MAX_REVISIONS:
        return "stop"

    return "reflect"


# 8. 构建 LangGraph
builder = StateGraph(HealthState)

builder.add_node("retriever", retriever_node)
builder.add_node("writer", writer_node)
builder.add_node("checker", evidence_checker_node)
builder.add_node("reflection", reflection_node)
builder.add_node("stop", stop_node)

builder.add_edge(START, "retriever")

builder.add_conditional_edges(
    "retriever",
    route_retriever,
    {
        "write": "writer",
        "finish": END
    }
)

builder.add_edge("writer", "checker")

builder.add_conditional_edges(
    "checker",
    route_checker,
    {
        "finish": END,
        "reflect": "reflection",
        "stop": "stop"
    }
)

builder.add_edge("reflection", "writer")
builder.add_edge("stop", END)

graph = builder.compile()



# =====================================
# 9. 封装Agent运行函数
# =====================================

def run_health_agent(task: str) -> dict:

    initial_state = {
        "task": task,
        "evidence": [],
        "draft": "",
        "issues": [],
        "revision_count": 0,
        "answer": "",
        "status": "running"
    }

    start_time = time.perf_counter()

    result = graph.invoke(
        initial_state,
        config={"recursion_limit": 20}
    )

    duration = round(
        time.perf_counter() - start_time,
        2
    )

    return {
        "status": result["status"],
        "answer": result["answer"],
        "evidence_count": len(result["evidence"]),
        "revision_count": result["revision_count"],
        "duration_seconds": duration
    }


# =====================================
# 10. 终端运行入口
# =====================================

if __name__ == "__main__":

    task = input("请输入健康科普任务：")

    result = run_health_agent(task)

    print("\n========== HealthAgent ==========")
    print("任务状态：", result["status"])
    print("检索资料数：", result["evidence_count"])
    print("Reflection次数：", result["revision_count"])
    print("\n最终结果：")
    print(result["answer"])
    print("\n任务耗时：", result["duration_seconds"], "秒")
