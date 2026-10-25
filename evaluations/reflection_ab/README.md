# 最终 HealthAgent Reflection OFF / ON 小规模配对评测

当前入口针对 `cases.json` 中四个固定场景，各独立执行 OFF 和 ON 一次，共最多8次端到端评测图调用。评测图复用生产节点及真实路由函数；没有直接调用公共 `run_health_agent` 函数。A/OFF唯一改变是把生产路由中的自动重写转为安全停止，B/ON保留最多一次 Reflection。

两组均保留完整 Evidence Checker、数字/范围/单位检查、任务参数保真和同一发布门禁。保留生产语义的 `completed`、`evaluation_failed`、`insufficient_content`、`no_evidence`；检查器异常、Writer截断或超时均不得发布草稿。

模型、Prompt、知识库及JSON正文协议从当前 `app/` 读取。Writer为qwen3:4b、json_article_v1、think=False、temperature=0、seed=42、num_predict=512、num_ctx=4096、请求超时45秒。Verifier配置通过禁网捕获实际 `verify_claim` 请求读取，原样传给模型；不能用Writer选项覆盖Verifier。CLI禁止将请求超时改成旧90秒。

初稿分别真实生成，温度和种子相同不保证跨运行逐字一致。执行顺序在调用前冻结：按案例文件顺序交替AB、BA。固定错误注入只发生于R-INJECT的首次Writer完整初稿之后，两组追加相同文本；修正轮不再注入，环境中的旧注入标志在评测期间归零。

旧历史的 `shared_prefix_paired` 设计及2048 token/90秒结果保留，它们不是本轮8次独立评测，不能混池比较。

在项目目录运行：

```sh
.venv/bin/python -m pytest -q tests evaluations/tests --junitxml=/absolute/path/final_ab_pytest.xml
.venv/bin/python -m evaluations.reflection_ab.run --validate-only
.venv/bin/python -m evaluations.reflection_ab.run --run-small --pytest-junit /absolute/path/final_ab_pytest.xml
```

`--validate-only`不访问网络、不调用模型。真实入口只接受四个未修改案例，固定8项计划，串行运行，无自动重试。每个独立任务最多300秒；单次模型请求仍为生产45秒。OFF最多7次模型调用，ON最多14次；无证据案例预期0次。四场景总上限63次模型调用，是最坏覆盖上限而非实际调用预算目标。

遇到独立任务超时或worker失败，保留已落盘事件、已观测调用下界、完整已取得响应和未知状态，继续剩余计划。未取得最终worker结果时不伪记零调用，也不声称最终发布行为已确认；如果日志中已有完整最终结果则可恢复。父进程终止不保证Ollama服务端立刻取消计算。

每轮创建新的 `evaluations/results/reflection-final-ab-.../`，不覆盖历史。保存配置/代码/Prompt/模型manifest哈希、源码快照、Git状态、历史数据和标签哈希，结束后重新核验。工作区已有未提交内容，因此Git HEAD之外还必须依赖源码快照识别本次实验。

输出包括：

- `report.json`：全部节点输入输出、模型请求响应、每轮门禁、状态、耗时与汇总。
- `report.md`：两组对照结论、失败原因、每任务最终回答及解释边界。
- `task_comparison.csv/.md`：每条任务对照表。
- `configuration_and_version.md`、`freeze.json`、`source_snapshot.zip`：配置和版本。
- `human_review.json/.md`：8条待独立人工审核记录，医学质量和任务满足度均为null。
- `remaining_delivery_checklist.md`：当前项目剩余交付。
- `events/`、`progress.jsonl`：逐任务增量记录。

指标区分：completed程序门禁通过率（全部4项及有证据3项口径）、将R-NONE正确无证据停止计入的程序定义任务成功率、安全停止率、可观测门禁错误发布、Reflection修正后通过次数、医学证据与任务参数门禁、调用和耗时。独立人工任务成功率/医学通过率当前未知。P50/P95按(n-1)*p线性插值，每组仅4项且含零调用任务，只作描述，不作显著性或总体正确率结论。

不自动提交Git，不自动开展额外模型实验。
