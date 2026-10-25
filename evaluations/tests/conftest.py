"""The Harness unit suite must never connect to Ollama or another service."""

import socket

import pytest


@pytest.fixture(autouse=True)
def prohibit_network(monkeypatch):
    def reject_connection(*args, **kwargs):
        raise AssertionError("Harness unit tests must not make network/model calls")

    monkeypatch.setattr(socket.socket, "connect", reject_connection)
    monkeypatch.setattr(socket.socket, "connect_ex", reject_connection)
