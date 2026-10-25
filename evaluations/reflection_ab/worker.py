"""One isolated production-config task/strategy; JSON stdin/stdout protocol."""

import json
import os
from pathlib import Path
import sys

from ollama import Client

from app import writer
from .core import digest, load_cases, run_case


HOST = "http://127.0.0.1:11434"


def main():
    request = json.load(sys.stdin)
    case, strategy = request["case"], request["strategy"]
    frozen = {item["id"]: item for item in load_cases()["cases"]}
    if case.get("id") not in frozen or digest(case) != digest(frozen[case["id"]]):
        raise ValueError("Worker only accepts an unchanged task from the four-case suite")
    if strategy not in ("A", "B"):
        raise ValueError("Worker only accepts Reflection OFF/A or ON/B")
    timeout = request["request_timeout_seconds"]
    if type(timeout) not in (float, int) or timeout != writer.REQUEST_TIMEOUT_SECONDS:
        raise ValueError("Request timeout must equal the current production timeout")
    event_path = Path(request["event_path"]).resolve()
    event_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive journals also prevent accidental reuse of an earlier run.
    with event_path.open("x", encoding="utf-8") as journal:
        def emit(event):
            journal.write(json.dumps(event, ensure_ascii=False) + "\n")
            journal.flush()
            os.fsync(journal.fileno())

        client = Client(host=HOST, timeout=writer.REQUEST_TIMEOUT_SECONDS)
        try:
            result = run_case(case, strategy, client.chat, execution_mode="real_model", event_sink=emit)
        except Exception as exc:
            emit({"event": "worker_failed", "case_id": case["id"], "strategy": strategy,
                  "error": {"type": type(exc).__name__, "message": str(exc)}})
            raise
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
