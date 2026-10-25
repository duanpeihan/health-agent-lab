"""Keep the test suite offline, including accidental local Ollama calls."""

import socket
from unittest.mock import patch


_network_patches = []


def _reject_connection(*args, **kwargs):
    raise AssertionError("Tests must mock model calls; network connections are disabled.")


def pytest_sessionstart(session):
    # TestClient uses the ASGI app in memory and needs no network connection.
    for method in ("connect", "connect_ex"):
        network_patch = patch.object(socket.socket, method, _reject_connection)
        network_patch.start()
        _network_patches.append(network_patch)


def pytest_sessionfinish(session, exitstatus):
    while _network_patches:
        _network_patches.pop().stop()
