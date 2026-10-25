# 第一阶段 Harness：Claim Verifier 离线评测

本目录独立于生产流程，不修改 `app/`、知识库或原有测试。入口直接调用
`app.claim_verifier.verify_claim(claim, documents)`，不经 FastAPI、不生成整篇文章、
不运行检索/Reflection。这里的“离线”指固定本地数据与证据，真实评测仍调用本机
Ollama 的 `qwen3:4b`。本目录未添加 Mock 模型 CLI。

## 数据和人工复核

`claims.v1.json` 有 24 条初始标注，四类各 6 条，**全部待人工复核**。
这些是根据现有事实拟定的种子标签，尚未经过人工审核，不能称为金标准。

- `supported`：给定证据直接支持主张，包含语义改写。
- `not_supported`：没提到或推不出；“至少 7 小时”推不出“必须 12 小时”，
  后者按现有生产提示词归为本类，不能仅因数字不同就归为矛盾。
- `contradicted`：证据与主张直接逻辑冲突，包含对原事实的否定。
- `uncertain`：给定证据自身的指代、范围或性质模糊，无法可靠确定。
  本知识库只有确定性事实，本类用独立的 `synthetic_ambiguous` 语言夹具；
  夹具不是 WHO/CDC 原文，也不是新增医学事实。

仅使用该条提供的证据，不用外部医学知识补足。`evidence_refs` 的 `fact_indices`
从 1 开始；运行前解析为选定原文。`expected_evidence_quotes` 为可接受的逐字引文。
KB SHA256 防止知识库改了但标签未跟进；引用不一致会拒绝运行。
人工审查可直接按案例 `id` 检查主张、标签、理由、引用和来源。

优先复核 `uncertain` 与 `not_supported` 的边界、量词、年龄范围、虚假数字以及
是否存在多种合理标签。需要医学事实更新时另行核对权威来源；本轮只核查当前本地
证据的蕴含关系。当前脚本有意要求保留 `pending_human_review`，不要未经审核改成
`reviewed`。完成逐条复核后，应发布新数据版本并相应放宽/更新状态校验，保留 v1。

## 命令

以下命令均在项目根目录执行，使用已有虚拟环境；没有新增运行依赖。

```bash
cd health-agent-lab

# 不调用模型，检查24条数据及固定证据
.venv/bin/python -m evaluations.run --validate-only

# 原有离线回归 + 新增Harness单测（所有网络连接被测试夹具阻断）
.venv/bin/python -m pytest -q tests evaluations/tests

# 语法检查
.venv/bin/python -m compileall -q app tests evaluations

# 真实模型，只跑固定前三条：supported / contradicted / not_supported
.venv/bin/python -m evaluations.run --smoke --timeout-seconds 180

# 可选：人工复核后再运行全部24条；本轮不会自动执行
.venv/bin/python -m evaluations.run --full --timeout-seconds 180
```

必须显式选择 `--validate-only`、`--smoke` 或 `--full`；帮助、导入和数据校验均不调用
模型。Smoke 三条均有证据，都会走真实模型路径；不覆盖 uncertain，不能据此评价
全部四类。完整集含一条空证据短路，它应为 `not_supported`，实际模型调用数为 0。

真实评测串行运行，每条使用独立子进程，最多等待设定秒数。超时会结束客户端工作
进程并记录失败；不保证 Ollama 服务端同步取消推理。建议本机不要同时跑其他模型
任务。若超时，先查看结果与本地模型状态，再决定是否增加时限；脚本不会自动重试。
每条 180 秒时，smoke 的调用等待上限约 9 分钟，24 条约 72 分钟；不是实际耗时预测。

## 结果与指标解释

每次写入新的 `evaluations/results/{mode}-{UTC时间}-{随机编号}/`，不覆盖历史结果：

- `progress.jsonl`：逐条完成后写入，意外中断仍保留已完成案例；只有它而没有完整
  `report.json` 表示运行未完成，不得按完整评测引用。
- `report.json`：全部结果、错误、证据原文、引用匹配、逐条耗时、数据/知识库/核查器/
  Harness 摘要、Git 状态、Python/Ollama SDK 版本与时间。
- `report.md`：便于审阅的 accuracy、各类 precision/recall/F1、混淆矩阵、耗时和错误。

Accuracy = 正确数 / 所有选中案例数；异常、超时、无效输出为 `error`，计入分母，
在对应真实类别中计为 FN。混淆矩阵行是初始标签，列是四类预测及 `error`。
Precision/Recall/F1 的零分母记为 0，support 同时显示。Macro F1 为四类的简单平均；
缺类的 smoke 不适合用此数比较模型。p50 用中位数，p95 用 nearest-rank；耗时含
子进程启动与导入，不等于模型纯推理时间。

JSON 另含 `metrics.by_evidence_origin`，把知识库、合成歧义和空证据分开；
`metrics.observed_model_cases` 只含成功返回且观测到真实模型调用的样本，可能排除
失败，不能代替含失败的总指标。分类正确与引用匹配分别报告，引用差异不暗改分类分数。

worker 在独立进程内透传原始 `chat` 调用并记录真实调用数/模型名/是否完成，
不改参数和返回值。`model_requests_expected` 只是按证据推算的预期调用数；
`observed_model_call_count` 才是收到的观测值。超时/子进程失败可能使调用数未知，
以 `unknown_model_call_count_cases` 单列，不假装从未调用。模型文件 digest 未查询时
为 null；模型标签或运行时可能变化，temperature=0 也不保证跨环境一致。

内部单测通过注入固定响应验证计分和错误处理，报告模式强制为 `test_double`；
这些测试不提供真实模型准确率。真实 CLI 报告标记 `real_model`，但由于人工标签
尚未复核，其数字也只能称为“对初始标签的一致率”。24 条种子集很小，不代表医疗
可靠性、完整 Agent 质量或线上泛化表现。

## 下一步 A / B

**A（建议）**：先逐条人工复核标签，重点看歧义与无依据边界；结合3条真实 smoke
检查格式、引用和失败原因，再决定是否调整下一版评测数据。保留当前生产行为。

**B**：接受完整评测的耗时后，显式执行 `--full` 建立24条初始基线，按错误明细复核，
随后扩展独立测试集。如果未来修改核查提示词，应另做版本对比，避免在这24条上
反复调参后把同一集合的结果当泛化能力。本轮不自动全量、不自动 commit。

## V2 独立策略实验（2026-10-24）

生产 `app/claim_verifier.py`、FastAPI 与本文件上面的 V1 运行路径保持原样。
离线代码可调用 `evaluations.strategy.get_verifier("v1"|"v2")` 显式选策略。
V2 位于 `v2_strategy.py`；`worker_v2.py` 只运行 V2，并记录本次实际模型调用。
V2 对同一批传入事实重新编号，要求模型分别标记数字、量词关系，返回引用编号；
Python 将编号映射为 `document_id`、该文档内 1 起始的 `fact_index` 和逐字原文，
并拒绝越界、重复、缺失或与 `direct_conflict` 判定自相矛盾的响应。数字和量词的
语义判断仍由模型作出，结构校验不等于确定性医学或逻辑证明。

小样本对照命令：

```bash
.venv/bin/python -m pytest -q tests evaluations/tests
.venv/bin/python -m evaluations.run --validate-only --dataset evaluations/claims.holdout.v2.json
.venv/bin/python -m evaluations.compare_v2 --timeout-seconds 60
```

对照脚本默认只选原24条中的6条，覆盖四类：`S001 C002 C006 N001 U001 U004`。
它复用 `evaluations/results/full-20261024T073456Z-c8fa2fe6/report.json` 的 V1 原始
逐条结果，本次只调用 V2；必须核对数据集 SHA256 和生产 V1 文件 SHA256。
`--case-ids` 可显式选择原始集内最多8条，不接受更大批量。V1 和 V2 对同一批
主张、证据比较，但不是同时运行；耗时差异仅作描述。

`claims.holdout.v2.json` 是独立保留的8条新案例，四类各2条，全部待人工复核。
这些案例未用于 V2 Prompt 编写，也未参加本轮真实 smoke；知识库相关案例仅评估
当前给定证据的语义关系，合成歧义案例不构成医学建议。请先独立复核标签，再将
保留集用于后续评测。不要把本轮六条探索性结果外推成真实部署准确率。

## 四分类标注与解释复核（纯离线）

`annotation_guidelines.md` 是项目自定义的四分类规约。`review_checklist.csv` 将原
24条与保留8条的主张、实际输入证据、原始标签/理由、Agent建议标签和争议说明并列；
人工最终标签与意见保持空白。建议复制清单到一个新的人工审核版本后填写，保留本
候选版和原始 JSON 数据。所有32条最终标签都需要独立人工确认，优先审查八条
`uncertain`、`N001`、`C002`、`H-C01` 及 `H-N02` 的解释问题。

`explanation_fidelity.md` 区分分类一致、引用有效与解释忠实度。
`explanation_fidelity_review.csv` 从已保存的8条配对真实结果整理16条理由，
只提示可能新增的效果词或数字；人工忠实度栏为空，不存在自动忠实度分数。

以下检查只读现有数据与结果，不调用 Ollama：

```bash
.venv/bin/python -m evaluations.review_checks
.venv/bin/python -m evaluations.explanation_fidelity
.venv/bin/python -m pytest -q tests evaluations/tests
```

两个生成器只在显式传入 `--write` 时首次创建候选 CSV，使用独占创建模式拒绝覆盖。
完成独立人工审核后，应另存带审核者、日期、裁决依据和版本号的新标签集，再基于
已保存的逐条预测重算探索性指标；不得覆盖历史初标或报告。


## 公开克隆与历史记录

公开仓库使用 `evaluations/fixtures/` 中最小历史投影进行离线回归，原始 `results/`、完整源码快照和运行环境日志保留在本地，不随默认 Git 提交发布。投影有源报告与派生文件 SHA256，属于历史测试输入，不是新增实验。公开最终结果见 `docs/reports/final_reflection_ab.md`。

32 条 AI 建议的默认校验严格核对公开的原始案例、知识库、标签和建议文件；它明确报告旧全仓快照的变动/缺失，不声称完整历史工作区一致。`--strict-snapshot` 仍要求原始全仓快照逐项一致，只适合有完整本地历史归档的审计环境。医学正确性和独立人工复核状态保持未完成。
