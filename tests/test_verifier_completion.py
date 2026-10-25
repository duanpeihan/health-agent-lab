"""Transport finish guards are independent of a parsable verdict object."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app import claim_verifier


@pytest.mark.parametrize("done,reason", [(True, "length"), (False, "stop"), (True, "error"), (None, None)])
def test_complete_json_does_not_override_unfinished_transport(monkeypatch, done, reason):
    content = json.dumps({"verdict": "supported", "reason": "fixed fixture", "evidence_id": 1})
    model = Mock(return_value=SimpleNamespace(message=SimpleNamespace(content=content),
                                              done=done, done_reason=reason))
    monkeypatch.setattr(claim_verifier, "chat", model)
    with pytest.raises(ValueError, match="unfinished or truncated"):
        claim_verifier.verify_claim("fixture", [{"facts": ["fixed evidence"]}])
    model.assert_called_once()
