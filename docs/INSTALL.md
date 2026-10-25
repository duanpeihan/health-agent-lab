# Mac 安装与运行

已验证环境为 **CPython 3.12.10 / macOS**。`.python-version` 为支持该文件的版本管理器固定此版本；普通 shell 仍需自行确认 `python3.12` 指向正确解释器。其他 Python 小版本未在本次交付验证。

`requirements.txt` 固定直接运行依赖；`requirements-dev.txt` 增加 pytest 与 TestClient 所需依赖；`requirements-lock.txt` 约束当前测试环境的传递依赖版本。该约束文件不包含本地工具、账号信息或可执行文件路径，也不是带下载哈希的 wheel 锁文件。

## 从克隆到离线验证

将下方示例仓库地址中的 `YOUR_GITHUB_USERNAME` 替换为发布者账号。项目目前没有自动创建远程仓库，也没有自动提交或推送。

```sh
git clone https://github.com/YOUR_GITHUB_USERNAME/health-agent-lab.git
cd health-agent-lab

python3.12 --version
# 本次验证版本：Python 3.12.10
python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install -r requirements-dev.txt
python -m pip check
python -m pytest -q tests evaluations/tests
```

测试使用固定输出、Mock 或冻结实验的公开测试夹具，并阻断模型网络访问。离线测试不要求安装、启动 Ollama，也不需要下载模型。只运行服务而不运行测试时，可安装 `requirements.txt`。

本轮交付优先检查现有环境与离线测试，没有联网重装依赖；上面的首次安装会从 Python 包索引下载所列版本。网络不可用时应提前准备对应 Python/macOS 的 wheel，而不是随意放宽版本约束。

## 本地 Ollama

需要先自行安装提供 `ollama` 命令的 macOS Ollama 客户端。项目不包含模型权重，也不需要云模型 API Key。使用独立终端启动本地服务；若 Ollama 桌面应用已经提供服务，不要再启动第二个同端口服务。

终端一：

```sh
ollama serve
```

终端二，首次使用时下载模型：

```sh
ollama pull qwen3:4b
ollama list
```

`ollama pull` 会下载模型；不要用它覆盖或替换已冻结实验的模型后再声称是原版本复现。`qwen3:4b` 标签可能随模型仓库更新，精确实验复现还应核对公开配置说明中的原模型标识与 manifest 指纹。

本项目冻结配置为 `json_article_v1`，Writer 使用 `qwen3:4b`、`think=False`、`temperature=0`、`seed=42`、`num_predict=512`、`num_ctx=4096`，单次请求超时 45 秒。Verifier 从生产函数读取参数，不借用 Writer 的输出预算。Reflection 最多一次；测试通过或 `completed` 不能代替独立医学审核。

## 启动 FastAPI

终端三，在克隆后的仓库根目录：

```sh
source .venv/bin/activate
export OLLAMA_HOST=http://127.0.0.1:11434
export INJECT_BAD_FACT=0
export INJECT_FALSE_CLAIM=0
export LANGSMITH_TRACING=false
export LANGCHAIN_TRACING_V2=false
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

浏览器可打开 `http://127.0.0.1:8000/docs`。本地开发需要自动重载时可额外加 `--reload`。兼容入口 `python -m uvicorn api:app` 仍可使用；推荐新入口 `app.api:app`。

```sh
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"task":"请依据资料说明成年人睡眠时长与睡眠质量。"}'
```

`GET /health` 只验证应用存活，不调用模型。`POST /generate` 对检索到证据的任务会真实调用本地模型；无相关证据时按流程直接停止。请求中的 `task` 长度为 5–500 字符。结果字段为 `status`、`answer`、`evidence_count`、`revision_count`、`duration_seconds`。

## 路径与环境变量

- 生产知识库位于 `data/health_knowledge.json`，由 `app/knowledge.py` 相对源码定位，不使用开发者个人目录。
- 包内使用相对导入。在仓库根目录执行上述命令；若从别处启动，先切换到自己的克隆目录，不要复制个人绝对路径。
- 生产 Ollama 客户端由当前 Ollama SDK 读取 `OLLAMA_HOST`。最终 A/B worker 为保持原实验条件固定使用本机 `http://127.0.0.1:11434`，并读取当前用户家目录下的本地模型 manifest。远程 Ollama 或修改模型存储位置不属于本轮已验证的冻结实验条件。
- 模型名、JSON 协议、长度与超时是当前生产配置；没有 `MODEL_NAME`、`NUM_PREDICT`、`REQUEST_TIMEOUT` 等环境变量开关，勿写入无效设置。
- `.env.example` 只是无凭据示例。应用不会自动读取 `.env`；可显式在 shell 设置变量，或运行 `set -a; source .env.example; set +a`。不要把真实凭据写进可提交文件。
- `INJECT_BAD_FACT` 与 `INJECT_FALSE_CLAIM` 仅供受控回归或评测。普通 API 使用必须为 `0` 或未设置。

本轮仅做离线 TestClient/API 存活验证，没有执行 `ollama pull`、会调用模型的生成请求或额外模型实验。最终 A/B 结果只来自已保存的八次独立运行。
