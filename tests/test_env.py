from __future__ import annotations

import pytest

from grapefruit.env import get_xai_api_key, grok_cli_env


def test_get_xai_api_key_prefers_official(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "xai-official")
    monkeypatch.setenv("GROK_API_KEY", "xai-legacy")
    assert get_xai_api_key() == "xai-official"


def test_get_xai_api_key_falls_back_to_legacy(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setenv("GROK_API_KEY", "xai-legacy")
    assert get_xai_api_key() == "xai-legacy"


def test_get_xai_api_key_required_missing(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("GROK_API_KEY", raising=False)
    with pytest.raises(ValueError, match="XAI_API_KEY"):
        get_xai_api_key()
    assert get_xai_api_key(required=False) is None


def test_grok_cli_env_copies_legacy_key(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setenv("GROK_API_KEY", "xai-legacy")
    env = grok_cli_env()
    assert env["XAI_API_KEY"] == "xai-legacy"
    assert env["GROK_API_KEY"] == "xai-legacy"
