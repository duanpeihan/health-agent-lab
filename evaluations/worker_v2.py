"""One isolated experimental V2 invocation; no labels enter the model."""

import json
import sys
import time
from importlib.metadata import version


def main():
    request = json.load(sys.stdin)
    from . import v2_strategy

    original_chat, calls = v2_strategy.chat, []

    def observed_chat(*args, **kwargs):
        call = {"model": kwargs.get("model"), "completed": False}
        calls.append(call)
        start = time.perf_counter()
        try:
            response = original_chat(*args, **kwargs)
            call["completed"] = True
            return response
        finally:
            call["elapsed_seconds"] = time.perf_counter() - start

    v2_strategy.chat = observed_chat
    verification = v2_strategy.verify_claim_v2(request["claim"], request["documents"])
    json.dump({"model": v2_strategy.MODEL, "verification": verification,
               "model_calls": calls, "model_call_count": len(calls),
               "ollama_python_version": version("ollama"), "model_digest": None},
              sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
