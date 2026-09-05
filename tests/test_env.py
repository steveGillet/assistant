from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from grapefruit.env import get_xai_api_key, grok_cli_env, read_grok_cli_token


def test_get_xai_api_key_prefers_official(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "xai-official")
    monkeypatch.setenv("GROK_API_KEY", "xai-legacy")
    monkeypatch.setattr(
        "grapefruit.env.read_grok_cli_token", lambda **_k: "jwt-should-not-win"
    )
    assert get_xai_api_key() == "xai-official"


def test_get_xai_api_key_falls_back_to_legacy(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setenv("GROK_API_KEY", "xai-legacy")
    monkeypatch.setattr("grapefruit.env.read_grok_cli_token", lambda **_k: None)
    assert get_xai_api_key() == "xai-legacy"


def test_get_xai_api_key_required_missing(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("GROK_API_KEY", raising=False)
    monkeypatch.setattr("grapefruit.env.read_grok_cli_token", lambda **_k: None)
    with pytest.raises(ValueError, match="XAI_API_KEY"):
        get_xai_api_key()
    assert get_xai_api_key(required=False) is None


def test_get_xai_api_key_uses_grok_login_when_env_is_key_id(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "00000000-0000-4000-8000-000000000000")
    monkeypatch.delenv("GROK_API_KEY", raising=False)
    monkeypatch.setattr(
        "grapefruit.env.read_grok_cli_token", lambda **_k: "eyJ-grok-login"
    )
    assert get_xai_api_key() == "eyJ-grok-login"


def test_get_xai_api_key_uses_grok_login_when_env_empty(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("GROK_API_KEY", raising=False)
    monkeypatch.setattr(
        "grapefruit.env.read_grok_cli_token", lambda **_k: "eyJ-grok-login"
    )
    assert get_xai_api_key() == "eyJ-grok-login"


def test_read_grok_cli_token_skips_expired(tmp_path):
    now = datetime(2026, 9, 5, 12, tzinfo=timezone.utc)
    path = tmp_path / "auth.json"
    path.write_text(
        '{"https://auth.x.ai::id": {"key": "eyJ-expired", '
        '"expires_at": "2026-09-05T08:00:00Z"}}'
    )
    assert read_grok_cli_token(path, now=now) is None
    path.write_text(
        '{"https://auth.x.ai::id": {"key": "eyJ-live", '
        '"expires_at": "2026-09-05T18:00:00Z"}}'
    )
    assert read_grok_cli_token(path, now=now) == "eyJ-live"
    assert read_grok_cli_token(path, now=now + timedelta(hours=8)) is None


def test_grok_cli_env_copies_legacy_key(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setenv("GROK_API_KEY", "xai-legacy")
    env = grok_cli_env()
    assert env["XAI_API_KEY"] == "xai-legacy"
    assert env["GROK_API_KEY"] == "xai-legacy"
