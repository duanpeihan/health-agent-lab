"""One isolated real production-verifier invocation. No label is sent here."""

import json
import sys
import time
from importlib.metadata import version


def main():
    request = json.load(sys.stdin)
    from app import claim_verifier

    original_chat, calls = claim_verifier.chat, []

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

    # Process-local observation only: forward every argument and response unchanged.
    claim_verifier.chat = observed_chat
    verification = claim_verifier.verify_claim(request["claim"], request["documents"])
    json.dump({"model": claim_verifier.MODEL, "verification": verification,
               "model_calls": calls, "model_call_count": len(calls),
               "ollama_python_version": version("ollama"), "model_digest": None},
              sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
