"""Offline integrity check for the 32-case human review worksheet."""

import argparse
import csv
from pathlib import Path

from .dataset import LABELS, load_dataset, resolve_documents


ROOT = Path(__file__).resolve().parent
CHECKLIST = ROOT / "review_checklist.csv"
FIELDS = (
    "数据集", "案例ID", "待核查主张", "实际输入证据原文", "原始标签", "原始标注理由",
    "建议标签", "建议理由", "是否存在争议", "争议说明", "人工最终标签", "人工复核意见",
)

# Separate from the seed datasets: these are Agent proposals, not human adjudications.
# Every ID is explicit so additions to either dataset require a new review pass.
PROPOSALS = {
    "S001": ("supported", "年龄、建议口径和至少7小时的下界均保留。", "否", "无明显语义争议；仍需人工最终确认。"),
    "C001": ("contradicted", "主张否定证据明确要求的良好睡眠质量。", "否", "相同健康睡眠条件下直接冲突。"),
    "N001": ("not_supported", "至少7小时不排除12小时，但不能推出所有成年人必须12小时。", "是", "数字下界、年龄范围和建议/必须强度易与直接矛盾混淆。"),
    "S002": ("supported", "两种活动强度、区间、或关系和人群均与原文一致。", "否", "无明显语义争议；仍需人工最终确认。"),
    "S003": ("supported", "至少2天和主要肌群力量训练均保留。", "否", "无明显语义争议；仍需人工最终确认。"),
    "S004": ("supported", "更好于不活动与少量起步、逐步增加均受原文支持。", "否", "无明显语义争议；仍需人工最终确认。"),
    "S005": ("supported", "只改写不同年龄人群的推荐时长不同。", "否", "无明显语义争议；仍需人工最终确认。"),
    "S006": ("supported", "足够时长与良好质量两个必要条件均保留。", "否", "无明显语义争议；仍需人工最终确认。"),
    "C002": ("contradicted", "主张以‘而不是’明确否定并替换证据的150至300分钟推荐区间。", "是", "需人工确认推荐区间替换是否构成直接冲突，而非仅缺少支持。"),
    "C003": ("contradicted", "0天且不应至少2天与证据的至少2天建议直接冲突。", "否", "同一训练类型和周频率。"),
    "C004": ("contradicted", "主张否定证据明确给出的活动优于不活动的比较。", "否", "无明显语义争议；仍需人工最终确认。"),
    "C005": ("contradicted", "相同人群下直接否定至少7小时的通常建议。", "否", "无明显语义争议；仍需人工最终确认。"),
    "C006": ("contradicted", "所有年龄推荐都相同与原文明确的不同直接冲突。", "否", "无明显语义争议；仍需人工最终确认。"),
    "N002": ("not_supported", "证据没有牛奶或治愈失眠的相关内容。", "否", "不得用外部医学知识补足。"),
    "N003": ("not_supported", "证据没有补钙、50%效果或所有成年人的保证。", "否", "无直接反驳该具体疗效的证据。"),
    "N004": ("not_supported", "18至60岁建议不能外推到65岁以上。", "否", "年龄越界是证据不足，不是直接冲突。"),
    "N005": ("not_supported", "睡眠建议不能推出一周内90%的失眠治愈率。", "否", "疗效、时间和比例均缺证据。"),
    "N006": ("not_supported", "实际输入为空证据，按规约为不支持。", "否", "空证据不表示主张现实中为假。"),
    "U001": ("uncertain", "7小时和9小时两个最终草稿候选均在证据中，最终采用哪稿未定。", "是", "需人工确认显式待选版本与普通证据不足的项目边界。"),
    "U002": ("uncertain", "两个活动区间待选，最终档位未记录。", "是", "需人工确认显式待选版本与普通证据不足的项目边界。"),
    "U003": ("uncertain", "‘他们’在证据中明确可能指A组或B组。", "是", "指代未消歧；需确认主张关于A组的断言如何标注。"),
    "U004": ("uncertain", "证据明确保留全部与部分两个量词范围。", "是", "‘可能是全部’不能直接推出‘所有’，需确认未决范围标签。"),
    "U005": ("uncertain", "7小时可能是推荐值也可能是观测均值。", "是", "需确认建议/观测性质的显式未决解释。"),
    "U006": ("uncertain", "‘和’的同时满足与任一满足两种解释明确未决。", "是", "需确认连接词歧义是否足以归入uncertain。"),
    "H-S01": ("supported", "高强度活动75至150分钟是原文明确给出的一个选项。", "否", "只抽取一个选项，不排除另一个。"),
    "H-S02": ("supported", "少量起步并逐步增加是原文同义改写。", "否", "未新增疗效。"),
    "H-C01": ("contradicted", "少于7小时而不是至少7小时与同口径推荐下界互斥。", "是", "需确认主张始终被理解为推荐，而非实际睡眠时长。"),
    "H-C02": ("contradicted", "主张否定原文明确要求的良好睡眠质量。", "否", "相同健康睡眠条件下直接冲突。"),
    "H-N01": ("not_supported", "证据没有拉伸20分钟、入睡时间减半或所有人。", "否", "不评估现实医疗效果。"),
    "H-N02": ("not_supported", "训练频率建议不能推出每个人长寿的保证。", "是", "标签倾向明确，但既有V2理由额外说‘有益’，须另审解释忠实度。"),
    "H-U01": ("uncertain", "2天与3天草案待选，最终版本未记录。", "是", "需人工确认显式未决版本按uncertain而非not_supported。"),
    "H-U02": ("uncertain", "20%的分母明确可能是全体或甲组。", "是", "日常阅读可能默认全体，但编者注明确保留两种解释。"),
}


def expected_rows():
    datasets = (
        ("原始24条", load_dataset(ROOT / "claims.v1.json")),
        ("独立保留8条", load_dataset(ROOT / "claims.holdout.v2.json")),
    )
    ids = {case["id"] for _, data in datasets for case in data["cases"]}
    if len(ids) != 32 or ids != set(PROPOSALS):
        raise ValueError("Review proposals must cover exactly the 32 unique cases")
    rows = []
    for source, data in datasets:
        for case in data["cases"]:
            suggested, basis, disputed, dispute_note = PROPOSALS[case["id"]]
            if suggested not in LABELS:
                raise ValueError(f"Invalid proposal for {case['id']}")
            documents = resolve_documents(case, data["_knowledge"])
            evidence = "\n".join(f"[{doc['id']}] {fact}" for doc in documents for fact in doc["facts"])
            rows.append(dict(zip(FIELDS, (
                source, case["id"], case["claim"], evidence or "（无输入证据）",
                case["expected_label"], case["rationale"], suggested, basis,
                disputed, dispute_note, "", "",
            ))))
    return rows


def validate_checklist(path=CHECKLIST):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError("Review checklist columns differ from the required schema")
        actual = list(reader)
    if actual != expected_rows():
        raise ValueError("Review checklist differs from source evidence/proposals or has filled human fields")
    return {"cases": len(actual), "disputed": sum(row["是否存在争议"] == "是" for row in actual),
            "human_labels_filled": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Create candidate CSV once; never overwrite")
    args = parser.parse_args(argv)
    if args.write:
        with CHECKLIST.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(expected_rows())
    print(validate_checklist())


if __name__ == "__main__":
    main()
