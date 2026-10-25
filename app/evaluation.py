import json
import time

from .claim_verifier import verify_claim
from .knowledge import load_knowledge_base


# 加载知识库
knowledge_base = load_knowledge_base()


# 评测数据：每条主张都有预期分类
test_cases = [
    {
        "id": "T01",
        "topic": "成年人睡眠",
        "claim": "18至60岁成年人每晚建议至少睡7小时。",
        "expected": "supported"
    },
    {
        "id": "T02",
        "topic": "成年人睡眠",
        "claim": "健康睡眠需要足够时长，也需要良好的睡眠质量。",
        "expected": "supported"
    },
    {
        "id": "T03",
        "topic": "成年人睡眠",
        "claim": "成年人每晚必须睡12小时。",
        "expected": "not_supported"
    },
    {
        "id": "T04",
        "topic": "成年人睡眠",
        "claim": "睡前喝牛奶能够治愈失眠。",
        "expected": "not_supported"
    },
    {
        "id": "T05",
        "topic": "成年人运动",
        "claim": "成年人每周应至少安排2天主要肌群的力量训练。",
        "expected": "supported"
    },
    {
        "id": "T06",
        "topic": "成年人运动",
        "claim": "完全不运动比进行一些身体活动更健康。",
        "expected": "contradicted"
    }
]


def main():
    results = []

    print("========== Claim Verifier评测 ==========")

    for case in test_cases:

        documents = [
            doc for doc in knowledge_base
            if doc["topic"] == case["topic"]
        ]

        start = time.perf_counter()

        try:
            verification = verify_claim(
                case["claim"],
                documents
            )

            predicted = verification["verdict"]
            error = None

        except Exception as e:
            predicted = "error"
            error = str(e)

        elapsed_ms = round(
            (time.perf_counter() - start) * 1000,
            2
        )

        passed = predicted == case["expected"]

        result = {
            "id": case["id"],
            "claim": case["claim"],
            "expected": case["expected"],
            "predicted": predicted,
            "passed": passed,
            "elapsed_ms": elapsed_ms,
            "error": error
        }

        results.append(result)

        print(f"\n测试：{case['id']}")
        print("预期分类：", case["expected"])
        print("模型分类：", predicted)
        print("结果：", "通过" if passed else "未通过")


    # 统计结果
    total = len(results)
    correct = sum(r["passed"] for r in results)

    accuracy = correct / total if total else 0

    print("\n========== 评测报告 ==========")
    print("测试数量：", total)
    print("正确分类数量：", correct)
    print(f"分类准确率：{accuracy:.2%}")


    # 保存结果
    with open("claim_eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n评测结果已保存到 claim_eval_results.json")


if __name__ == "__main__":
    main()
