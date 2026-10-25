"""Load the current knowledge base independently of the working directory."""

import json
from pathlib import Path


KNOWLEDGE_PATH = Path(__file__).resolve().parent.parent / "data" / "health_knowledge.json"


def load_knowledge_base() -> list[dict]:
    with KNOWLEDGE_PATH.open(encoding="utf-8") as source:
        return json.load(source)
