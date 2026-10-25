"""One isolated invocation of the explicitly selected frozen strategy."""

import json
import sys
import time
from importlib.metadata import version


def main():
    request = json.load(sys.stdin)
    strategy = request["strategy"]
    if strategy == "v1":
        from app import claim_verifier as module
        verifier = module.verify_claim
    elif strategy == "v2":
        from . import v2_strategy as module
        verifier = module.verify_claim_v2
    else:
        raise ValueError("strategy must be v1 or v2")
    original_chat, calls = module.chat, []

    def observed_chat(*args, **kwargs):
        call = {"model": kwargs.get("model"), "temperature": kwargs.get("options", {}).get("temperature"),
                "think": kwargs.get("think"), "completed": False}
        calls.append(call)
        start = time.perf_counter()
        try:
            response = original_chat(*args, **kwargs)
            call["completed"] = True
            return response
        finally:
            call["elapsed_seconds"] = time.perf_counter() - start

    module.chat = observed_chat
    verification = verifier(request["claim"], request["documents"])
    json.dump({"strategy": strategy, "model": module.MODEL, "verification": verification,
               "model_calls": calls, "model_call_count": len(calls),
               "ollama_python_version": version("ollama")}, sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
