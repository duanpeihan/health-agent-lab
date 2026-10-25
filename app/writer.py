"""Bounded structured writing; accepting a response requires a complete article.

The installed Thinking weights did not obey a plain-text closing-tag prefill.
A JSON response schema now constrains the transport, while Python validates the
article separately. This does not claim native thinking-off support. Raw output
and finish metadata are retained without salvaging a clipped paragraph.
"""

import json
import re
import time

from ollama import Client


MODEL = "qwen3:4b"
OPTIONS = {"temperature": 0, "seed": 42, "num_predict": 512, "num_ctx": 4096}
REQUEST_TIMEOUT_SECONDS = 45
MAX_ARTICLE_CHARS = 600
PROTOCOL = "json_article_v1"
OUTPUT_SCHEMA = {
    "type": "object", "properties": {"article": {"type": "string"}},
    "required": ["article"], "additionalProperties": False,
}
SAFE_STOP = "内容生成或证据核查未通过，因此不输出未经核查的科普内容。"
SYSTEM_PROMPT = (
    '只返回JSON对象，article字段直接写一段简短健康科普正文。事实少时写短，不凑字数。'
    "只用给定事实，优先复用事实原句。年龄、数字、范围、单位及建议口径须与原文一致，不转换、不猜改，使用[1]等来源编号。"
    "问题涉及的效果不在资料中时，明确说资料不足、无法确认。"
    "不加诊疗建议、常识或额外效果。不写分析、提纲、字数、自我评价或修订说明。/no_think"
)
ANALYSIS_MARKERS = re.compile(
    r"(?:^|\n)\s*(?:分析[：:]|思考[：:]|文章结构[：:]|草拟内容[：:]|"
    r"首先[，,]?\s*(?:用户|我需要|我应该)|我(?:需要|应该|将要)(?:写|生成|分析|确保))"
)


def default_chat(*args, **kwargs):
    # A per-request client also bounds the production Writer, not just the harness.
    return Client(timeout=REQUEST_TIMEOUT_SECONDS).chat(*args, **kwargs)


def build_messages(state):
    evidence = "\n".join(
        f"[{index}] " + " ".join(document["facts"])
        for index, document in enumerate(state["evidence"], 1)
    )
    user = f"任务：{state['task']}\n事实：\n{evidence}"
    if state["revision_count"]:
        # Articles are bounded on acceptance, so this retains the whole prior draft.
        previous = state["draft"][:MAX_ARTICLE_CHARS]
        feedback = "；".join(str(item)[:180] for item in state["issues"][:6])
        user += (
            f"\n上一版文章：{previous}\n发现的问题：{feedback}\n"
            "仅删除或改写问题句，保留有据内容，输出完整的短正文。不要复述反馈。"
        )
    user += '\n仅返回规定JSON，article只放最终正文。/no_think'
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def generate_article(state, chat_fn):
    begin = time.perf_counter()
    metadata = {
        "phase": "repair" if state["revision_count"] else "initial",
        "model": MODEL, "options": dict(OPTIONS), "think": False,
        "response_received": False, "eval_count": None, "prompt_eval_count": None,
        "done": None, "done_reason": None, "truncated": False,
        "raw_content": "", "error": None, "analysis_detected": False,
        "assistant_prefill": None, "protocol": PROTOCOL, "elapsed_seconds": None,
    }
    article = ""
    try:
        response = chat_fn(model=MODEL, messages=build_messages(state), think=False,
                           format=OUTPUT_SCHEMA, options=dict(OPTIONS))
        metadata["response_received"] = True
        for field in ("eval_count", "prompt_eval_count", "done", "done_reason"):
            metadata[field] = getattr(response, field, None)
        metadata["raw_content"] = response.message.content or ""
        metadata["truncated"] = metadata["done_reason"] == "length"
        if metadata["truncated"]:
            raise ValueError("Writer output was truncated at the fixed token limit")
        if metadata["done"] is not True:
            raise ValueError("Writer response did not finish")
        if metadata["done_reason"] != "stop":
            raise ValueError("Writer returned an unexpected finish reason")
        parsed = json.loads(metadata["raw_content"])
        if not isinstance(parsed, dict) or set(parsed) != {"article"} or not isinstance(parsed["article"], str):
            raise ValueError("Writer response does not match the article object schema")
        article = parsed["article"].strip()
        if "<think>" in article or "</think>" in article:
            metadata["analysis_detected"] = True
            raise ValueError("Writer article contains thinking markup")
        if ANALYSIS_MARKERS.search(article):
            metadata["analysis_detected"] = True
            raise ValueError("Writer returned analysis instead of a final article")
        if not article or len(article) > MAX_ARTICLE_CHARS:
            raise ValueError("Writer returned an empty or oversized article")
    except Exception as exc:
        metadata["error"] = {"type": type(exc).__name__, "message": str(exc)}
        article = ""
    finally:
        metadata["elapsed_seconds"] = time.perf_counter() - begin
    runs = [*state.get("writer_runs", []), metadata]
    if metadata["error"]:
        return {"draft": "", "status": "draft_generation_failed", "generation_failed": True,
                "issues": ["Writer生成失败：" + metadata["error"]["message"]],
                "answer": SAFE_STOP, "writer_runs": runs}
    return {"draft": article, "status": "drafted", "generation_failed": False,
            "answer": "", "writer_runs": runs}
