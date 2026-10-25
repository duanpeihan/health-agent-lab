"""Compatibility entry point for ``uvicorn api:app`` and existing API mocks."""

import sys

from app import api as _api


# Use the same module so patch("api.run_health_agent") still affects the routes.
sys.modules[__name__] = _api
