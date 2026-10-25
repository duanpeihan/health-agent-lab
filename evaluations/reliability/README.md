# 修复后工程可靠性小规模验证

该模块独立观察当前 `app.agent` 的 Retriever、Writer、Evidence Checker、Reflection 节点；不改变生产 API，不复用旧 A/B 的提示或参数覆盖逻辑。

固定顺序为 R-NORMAL、R-INSUFFICIENT、R-INJECT；任务与故障逐字读取原 `reflection_ab/cases.json`。只运行修复后的 Reflection ON，最多自动修正一次，修正后完整重查。故障只在成功生成的首次草稿追加一次，并暂时清除外部注入环境变量。截断、超时、Writer 失效及核查失败均安全停止。

离线验证（不连接模型）：

```sh
.venv/bin/python -m evaluations.reliability.run --validate-only
.venv/bin/python -m pytest evaluations/tests/test_reliability.py
```

通过全部离线测试后的小规模真实验证（仅这三条，不重试）：

```sh
.venv/bin/python -m evaluations.reliability.run --run-small
```

默认单次请求 45 秒、单任务 180 秒；可显式设定但分别不得超过 60 秒、240 秒。增加预算不能代替修复。命令不提供任意任务或批量规模参数。

每次新建 `evaluations/results/reliability-UTC-随机ID/`，保存 report.json/report.md、human_review.json、freeze.json、source_snapshot.zip、progress.jsonl、每任务 events。原始响应、token 数、结束原因、截断、调用和各阶段时间均来自观察；纯 exact_fact 核查的判断方式见 checker_details，不会虚构模型调用。Reflection 节点只计数与路由，修正生成耗时计入 Writer。

冻结所有 app Python 源码、新 runner、知识库、原案例、测试、旧基准报告、Git 状态、Ollama 模型 manifest、有效模板及哈希，并核对历史 results 文件哈希。所有输出均新建，不覆盖 V1/V2 或旧 A/B，也不创建 Git commit。进程被终止时只报告事件日志可观测调用下界，末轮问题数保持未知。

历史对照仅比较旧 `reflection-ab-20261024T130302Z-2d4a399f` 的 B 与新 ON。旧 B 是共享初稿后分叉，新 ON 是独立顺序运行；时间、Writer 提示与生成长度、核查规则已变，因此不是新的配对 A/B，也不提供单因素因果结论。

检查器通过只计入工程通过率，不证明医学正确。三个内容质量记录均待独立人工审核，医学质量、证据忠实度、任务完成度、误拦截确认与审核者保持空值。真实误拦截只列诊断候选，不直接计为已确认误拦截。

2026-10-24 后续状态：首次三例修复后真实运行 `reliability-20261024T151335Z-cb238e68` 已封存，三例均在512 token上限发生Writer截断并安全停止。其后Writer切换为约束JSON对象的 `article` 字段；该新协议本轮只做离线验证，尚无真实模型结果，不能将此前三例计作新协议的实测。新运行的freeze会单独记录当前协议，原始请求同时保存实际format。
