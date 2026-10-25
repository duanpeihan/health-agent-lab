# GitHub 展示项目发布步骤

本说明只提供后续人工步骤。工程整理过程没有执行 Git 暂存、提交、远程创建或推送。

## 已准备公开的内容

- `app/`、`data/`：生产代码和少量带来源的本地知识库。
- `tests/`、`evaluations/`：离线回归、评测代码、合成案例、AI建议和最小历史测试夹具。AI建议不冒充独立人工标签。
- `docs/reports/`：最终 A/B 的脱敏公开投影、精确指标、配置及来源哈希；没有更改原实验。
- `legacy/pre_modularization/` 与根目录兼容入口：保留早期核心实现和导入兼容性。
- `requirements*.txt`、`.python-version`、`.env.example`、README：安装与运行说明。

`.venv/`、缓存、`.env*`（示例除外）、模型权重、`evaluations/results/`、JSONL日志、压缩归档及零散根目录旧demo保持在本地并由Git忽略。忽略不删除文件；已有Git提交历史也不重写。不要使用 `git add -f` 强行加入原始实验目录。

公开夹具提取自已保存结果，只保留离线测试需要的字段；`evaluations/fixtures/provenance.json` 记录原始来源和派生SHA。完整个人环境日志不公开。旧全仓快照审计仍可在持有原始归档的环境使用严格模式；公共标签校验只声明证据/标签输入范围，不冒称全历史快照一致。

## 发布前由维护者确认

1. 使用下列命令检查候选改动，确认没有私人文件、真实凭据或误加入的原始日志。
2. 把README中的 `YOUR_GITHUB_USERNAME` 换成实际账号/组织；在GitHub手动建立空仓库。
3. 确认Git作者身份和希望采用的许可方式。本轮没有替维护者授予开源许可证；公开展示与授予他人复用权是不同决定。
4. 按下列步骤自行暂存、审阅、提交、推送。独立医学审核尚未完成，README必须保留该限制；这不应被宣传为临床可用系统。

## 在项目根目录手动执行

先复核环境和离线测试；不需要启动Ollama，也不调用模型：

```sh
python -m pip check
python -m pytest -q tests evaluations/tests
git status --short
git diff --check
git diff --stat
```

只选公开内容暂存，不使用全目录强制加入：

```sh
git add -- .gitignore .python-version .env.example README.md \
  requirements.txt requirements-dev.txt requirements-lock.txt pytest.ini \
  api.py health_content_agent_v3.py claim_verifier.py evaluate_claims.py \
  health_knowledge.json app data tests evaluations legacy/pre_modularization docs
git diff --cached --name-only
git diff --cached --check
git diff --cached --stat
git diff --cached
```

确认暂存列表没有 `.venv/`、`.env`、`evaluations/results/`、模型文件、JSONL或个人归档。提交前可再检查被忽略记录：

```sh
git status --short --ignored
git check-ignore .env .venv/ evaluations/results/
git diff --cached --name-only -- .env .venv evaluations/results
```

准备完成后由维护者执行（以下地址是待替换的占位符）：

```sh
git commit -m "Prepare HealthAgent showcase delivery"
git remote -v
# 没有origin时添加；已有时先确认，不自动覆盖
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/health-agent-lab.git
git push -u origin HEAD
```

不要为了让演示“全部成功”而删除失败案例或改动 A/B 指标。R-INSUFFICIENT的一次修正仍失败、模型数字失真和知识库范围有限，都属于需要如实保留的工程结果。
