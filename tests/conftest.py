import os
import socket
import sys
from pathlib import Path

import pytest

# Use litellm's bundled price map instead of downloading it at import time.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

BOT_DIR = Path(__file__).resolve().parent.parent
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))


class NetworkBlocked(RuntimeError):
    pass


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Every test runs with outbound connections disabled."""

    def guard(*args, **kwargs):
        raise NetworkBlocked(f"network access attempted in a unit test: {args[:2]}")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket.socket, "connect_ex", guard)
    monkeypatch.setattr(socket, "create_connection", guard)
    monkeypatch.setattr(socket, "getaddrinfo", guard)
    yield


@pytest.fixture
def clean_env(monkeypatch):
    """Remove every variable bot_config reads, so tests start from nothing."""
    for name in list(os.environ):
        if name.startswith("BOT_") or name in (
            "OPENROUTER_API_KEY",
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "METACULUS_TOKEN",
            "ASKNEWS_CLIENT_ID",
            "ASKNEWS_SECRET",
            "ASKNEWS_API_KEY",
        ):
            monkeypatch.delenv(name, raising=False)
    return monkeypatch
