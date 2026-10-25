"""Compatibility entry point; original source is kept under legacy/."""

import sys

from app import claim_verifier as _implementation


if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
