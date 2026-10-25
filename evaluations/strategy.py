"""Explicit offline-only strategy switch; app imports remain untouched."""


def get_verifier(strategy):
    if strategy == "v1":
        from app.claim_verifier import verify_claim
        return verify_claim
    if strategy == "v2":
        from .v2_strategy import verify_claim_v2
        return verify_claim_v2
    raise ValueError("strategy must be v1 or v2")
