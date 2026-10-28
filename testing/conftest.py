"""Shared setup for the CoralWatch test suites in testing/ (see testing/README.md)."""

import datetime as dt
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("CORALWATCH_NO_REFRESH", "1")      # no background NOAA refresh during tests
sys.path[:0] = [str(ROOT), str(ROOT / "forecast" / "2_model")]

CASE_DATE = dt.date(2024, 3, 3)       # peak of the 2024 mass bleaching, in the held-out test season (data cached)


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient
    from backend.app import app
    return TestClient(app)


@pytest.fixture
def offline_llm(monkeypatch):
    """Force the template briefing, so tests don't depend on Ollama or an API key."""
    monkeypatch.setenv("CORALWATCH_LLM", "offline")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)


def sse_events(text):
    """Parse a server-sent-events body into [(event, data_json)]."""
    import json
    out = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        if "event" in lines:
            out.append((lines["event"], json.loads(lines.get("data", "null"))))
    return out
