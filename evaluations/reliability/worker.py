"""Isolated real-model task; exclusive JSONL journal and JSON stdin/stdout."""

from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import sys

from ollama import Client

from .run import HOST, run_case, validate_timeouts


def main():
    request = json.load(sys.stdin)
    timeout = request["request_timeout_seconds"]
    validate_timeouts(180, timeout)
    event_path = Path(request["event_path"]).resolve()
    event_path.parent.mkdir(parents=True, exist_ok=True)
    with event_path.open("x", encoding="utf-8") as journal:
        def emit(event):
            journal.write(json.dumps(event, ensure_ascii=False) + "\n")
            journal.flush()
            os.fsync(journal.fileno())
        client = Client(host=HOST, timeout=timeout)
        # Production nodes print progress; keep protocol stdout valid JSON.
        with redirect_stdout(sys.stderr):
            result = run_case(request["case"], client.chat, execution_mode="real_model", event_sink=emit)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
