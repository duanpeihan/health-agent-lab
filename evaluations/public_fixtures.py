"""Small, hash-checked public projections of immutable historical records.

These files support offline regressions in a clean checkout. They are not new
model experiments, and they do not replace the private full records under
``evaluations/results``. Only synthetic tasks and the observations required by
tests are retained; local paths and transport/environment inventories are not.
"""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).with_name("fixtures")
PROVENANCE = FIXTURES / "provenance.json"


def public_fixture_path(key):
    """Resolve a registered fixture and verify its published byte digest."""
    manifest = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    try:
        record = manifest["fixtures"][key]
    except KeyError as exc:
        raise ValueError(f"Unknown public fixture: {key}") from exc
    path = (FIXTURES / record["path"]).resolve()
    if not path.is_relative_to(FIXTURES.resolve()) or not path.is_file():
        raise ValueError(f"Missing or unsafe public fixture: {key}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError(f"Public fixture SHA256 differs: {key}")
    return path


def load_writer_report(run_id):
    """Load the exact saved article bodies needed by a regression replay."""
    report = json.loads(public_fixture_path("writer:" + run_id).read_text(encoding="utf-8"))
    if report.get("fixture_kind") != "historical_projection_not_new_experiment":
        raise ValueError("Writer fixture must be labelled as a historical projection")
    return report


def historical_source_bytes(relative):
    """Return source bytes whose original freeze identity remains unchanged."""
    return public_fixture_path("source:" + relative).read_bytes()


def materialize_historical_source_root(target):
    """Build the original four-file holdout guard input for an offline test.

    Archived Verifier source remains byte-identical. The dataset and knowledge
    files still come from the checkout and must match their frozen SHA256s.
    This helper does not modify or bypass the real runner's source guard.
    """
    target = Path(target).resolve()
    freeze = json.loads((ROOT / "evaluations/frozen/holdout-v2.freeze.json").read_text(encoding="utf-8"))
    archived = {"app/claim_verifier.py", "evaluations/v2_strategy.py"}
    for relative, expected in freeze["source_sha256"].items():
        data = historical_source_bytes(relative) if relative in archived else (ROOT / relative).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Historical source fixture differs from freeze: {relative}")
        path = (target / relative).resolve()
        if not path.is_relative_to(target):
            raise ValueError("Historical fixture path escapes its target")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return target
