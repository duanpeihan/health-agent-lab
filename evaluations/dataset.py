"""Validate reviewable labels against a pinned local knowledge snapshot."""

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path(__file__).with_name("claims.v1.json")
DEFAULT_KNOWLEDGE = ROOT / "data" / "health_knowledge.json"
LABELS = ("supported", "not_supported", "contradicted", "uncertain")


def resolve_documents(case, knowledge):
    """Only expose explicitly selected facts, never implicit retrieval results."""
    if case["evidence_origin"] == "synthetic_ambiguous":
        return copy.deepcopy(case["evidence_fixture"])
    by_id = {doc["id"]: doc for doc in knowledge}
    documents = []
    for ref in case["evidence_refs"]:
        doc = copy.deepcopy(by_id[ref["document_id"]])
        doc["facts"] = [doc["facts"][index - 1] for index in ref["fact_indices"]]
        documents.append(doc)
    return documents


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def load_dataset(path=DEFAULT_DATASET, knowledge_path=DEFAULT_KNOWLEDGE):
    path, knowledge_path = Path(path), Path(knowledge_path)
    knowledge_bytes = knowledge_path.read_bytes()
    knowledge = json.loads(knowledge_bytes)
    dataset = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(dataset, dict), "Dataset must be an object")
    _require(isinstance(knowledge, list) and all(isinstance(doc, dict) for doc in knowledge),
             "Knowledge must be a list of objects")
    _require(dataset.get("schema_version") == "1.0", "Unsupported dataset schema_version")
    _require(bool(dataset.get("dataset_id")), "Missing dataset_id")
    _require(dataset.get("label_status") == "pending_human_review", "Expected provisional labels")
    _require(dataset.get("knowledge_sha256") == hashlib.sha256(knowledge_bytes).hexdigest(),
             "Knowledge SHA256 mismatch: review and re-pin labels before evaluation")
    cases = dataset.get("cases")
    _require(isinstance(cases, list) and len(cases) >= 3, "Dataset requires at least 3 cases")
    by_id = {doc["id"]: doc for doc in knowledge}
    seen = set()
    for case in cases:
        _require(isinstance(case, dict), "Each case must be an object")
        case_id = case.get("id")
        _require(isinstance(case_id, str) and case_id and case_id not in seen, "Duplicate/missing case id")
        seen.add(case_id)
        prefix = f"{case_id}: "
        _require(case.get("expected_label") in LABELS, prefix + "Invalid expected_label")
        for field in ("claim", "rationale"):
            _require(isinstance(case.get(field), str) and case[field].strip(), prefix + f"Missing {field}")
        _require(isinstance(case.get("tags"), list) and case["tags"] and
                 all(isinstance(tag, str) and tag for tag in case["tags"]), prefix + "Missing tags")
        _require(case.get("label_status") == "pending_human_review" and case.get("reviewer") is None,
                 prefix + "This v1 seed must retain pending review status")
        origin = case.get("evidence_origin")
        _require(origin in ("knowledge_base", "synthetic_ambiguous", "empty_evidence"),
                 prefix + "Invalid evidence_origin")
        refs, fixture = case.get("evidence_refs"), case.get("evidence_fixture", [])
        _require(isinstance(refs, list) and isinstance(fixture, list), prefix + "Invalid evidence lists")
        if origin == "synthetic_ambiguous":
            _require(not refs and bool(fixture), prefix + "Synthetic evidence must be isolated")
            for doc in fixture:
                _require(isinstance(doc, dict) and isinstance(doc.get("facts"), list) and doc["facts"]
                         and all(isinstance(fact, str) and fact.strip() for fact in doc["facts"]),
                         prefix + "Invalid synthetic facts")
                _require(doc.get("source") == "synthetic_ambiguous", prefix + "Synthetic source must be explicit")
        elif origin == "empty_evidence":
            _require(not refs and not fixture, prefix + "Empty evidence must be empty")
        else:
            _require(bool(refs) and not fixture, prefix + "KB evidence requires references only")
        ref_ids = set()
        for ref in refs:
            _require(isinstance(ref, dict), prefix + "Each evidence reference must be an object")
            doc_id, indices = ref.get("document_id"), ref.get("fact_indices")
            _require(doc_id in by_id and doc_id not in ref_ids, prefix + "Unknown/duplicate document_id")
            ref_ids.add(doc_id)
            _require(isinstance(indices, list) and bool(indices), prefix + "Missing fact_indices")
            _require(all(type(i) is int and 1 <= i <= len(by_id[doc_id]["facts"]) for i in indices),
                     prefix + "Invalid 1-based fact index")
            _require(len(indices) == len(set(indices)), prefix + "Duplicate fact index")
        quotes = case.get("expected_evidence_quotes")
        _require(isinstance(quotes, list) and all(isinstance(q, str) and q for q in quotes),
                 prefix + "Invalid expected_evidence_quotes")
        facts = [fact for doc in resolve_documents(case, knowledge) for fact in doc["facts"]]
        if case["expected_label"] in ("supported", "contradicted"):
            _require(bool(quotes) and all(q in facts for q in quotes), prefix + "Expected quote is not exact evidence")
        else:
            _require(not quotes, prefix + "Non-evidential labels require empty expected quotes")
    # Private runtime data is excluded from the dataset hash/report source file.
    dataset["_knowledge"] = knowledge
    dataset["_path"] = str(path.resolve())
    dataset["_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dataset
