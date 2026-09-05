"""Resolve xAI credentials for the official Grok API and Grok CLI.

Official env var is XAI_API_KEY (console keys start with ``xai-``).
GROK_API_KEY is accepted as a fallback from the old community CLI.
If those are missing or are not console keys, use the access token from
``grok login`` in ``~/.grok/auth.json``.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

GROK_AUTH_FILE = Path.home() / ".grok" / "auth.json"


def _is_console_api_key(key: str) -> bool:
    return key.startswith("xai-")


def _parse_expires_at(raw: object) -> datetime | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def read_grok_cli_token(
    path: Path | None = None,
    *,
    now: datetime | None = None,
) -> str | None:
    """Unexpired access token from ``grok login``, if any."""
    path = path or GROK_AUTH_FILE
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    now = now or datetime.now(timezone.utc)
    best: str | None = None
    best_exp: datetime | None = None
    for entry in data.values():
        if not isinstance(entry, dict):
            continue
        token = entry.get("key") or entry.get("access_token")
        if not isinstance(token, str) or not token.strip():
            continue
        token = token.strip()
        expires = _parse_expires_at(entry.get("expires_at"))
        if expires is not None and expires <= now:
            continue
        if best is None or (
            expires is not None and (best_exp is None or expires > best_exp)
        ):
            best = token
            best_exp = expires
    return best


def get_xai_api_key(*, required: bool = True) -> str | None:
    official = (os.getenv("XAI_API_KEY") or "").strip() or None
    legacy = (os.getenv("GROK_API_KEY") or "").strip() or None
    if official and _is_console_api_key(official):
        return official
    if legacy and _is_console_api_key(legacy):
        return legacy
    grok_token = read_grok_cli_token()
    if grok_token:
        return grok_token
    key = official or legacy
    if required and not key:
        raise ValueError(
            "Set XAI_API_KEY (a console key starting with xai-) or run `grok login`. "
            "Create a key at https://console.x.ai"
        )
    return key


def grok_cli_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Env for subprocesses of the official `grok` CLI."""
    env = os.environ.copy()
    official = (env.get("XAI_API_KEY") or "").strip()
    legacy = (env.get("GROK_API_KEY") or "").strip()
    if not official and legacy:
        env["XAI_API_KEY"] = legacy
    if extra:
        env.update(extra)
    return env
