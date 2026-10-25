import os
import time
from typing import TypedDict
from .claim_verifier import verify_claim
from .knowledge import load_knowledge_base

from .writer import generate_article, default_chat as chat, SAFE_STOP
from .evidence_checker import check_evidence
from .content_requirements import assess_evidence_capacity
from langgraph.graph import StateGraph, START, END


MODEL = "qwen3:4b"
MAX_REVISIONS = 1

# 加载现有健康知识库
knowledge_base = load_knowledge_base()


# 1. 定义状态
class HealthState(TypedDict):
    task: str
    evidence: list[dict]
    draft: str
    issues: list[str]
    revision_count: int
    answer: str
    status: str
    writer_runs: list[dict]
    generation_failed: bool
    checker_details: dict


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

    capacity = assess_evidence_capacity(state["task"], evidence)
    if capacity["status"] == "insufficient_content":
        return {"evidence": evidence, "status": "insufficient_content",
                "issues": capacity["issues"],
                "checker_details": {"content_assessment": capacity, "complete": False},
                "answer": "当前资料与生成篇幅不足以满足任务，请补充可核查资料或调整为简短说明。"}

    return {
        "evidence": evidence,
        "status": "retrieved"
    }


# 3. Writer：生成或修正文章
def writer_node(state: HealthState):
    result = generate_article(state, chat)
    if result["generation_failed"]:
        return result
    draft = result["draft"]
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

    return {**result, "draft": draft}


# 4. Evidence Checker：证据核查
def evidence_checker_node(state: HealthState):
    return check_evidence(state, verify_claim)


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
    if state["status"] == "insufficient_content":
        return {"status": "insufficient_content", "answer": state["answer"]}
    return {"status": "evaluation_failed", "answer": SAFE_STOP}


# 7. 条件路由
def route_retriever(state: HealthState):
    return "finish" if state["status"] in ("no_evidence", "insufficient_content") else "write"


def route_checker(state: HealthState):

    if state["status"] == "completed":
        return "finish"

    if state["status"] == "insufficient_content":
        return "stop"

    if not state.get("checker_details", {}).get("complete", False):
        return "stop"

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

builder.add_conditional_edges(
    "writer",
    lambda state: "stop" if state.get("generation_failed") else "check",
    {"stop": "stop", "check": "checker"},
)

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
        "status": "running",
        "writer_runs": [],
        "generation_failed": False,
        "checker_details": {},
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

def main():

    task = input("请输入健康科普任务：")

    result = run_health_agent(task)

    print("\n========== HealthAgent ==========")
    print("任务状态：", result["status"])
    print("检索资料数：", result["evidence_count"])
    print("Reflection次数：", result["revision_count"])
    print("\n最终结果：")
    print(result["answer"])
    print("\n任务耗时：", result["duration_seconds"], "秒")


if __name__ == "__main__":
    main()
