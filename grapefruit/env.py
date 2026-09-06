"""Resolve xAI credentials for the official Grok API and Grok CLI.

Official env var is XAI_API_KEY (console keys start with ``xai-``).
GROK_API_KEY is accepted as a fallback from the old community CLI.
If those are missing or are not console keys, use the access token from
``grok login`` in ``~/.grok/auth.json``.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

GROK_AUTH_FILE = Path.home() / ".grok" / "auth.json"
DEFAULT_OIDC_ISSUER = "https://auth.x.ai"
DEFAULT_OIDC_CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"
REFRESH_SKEW = timedelta(minutes=5)
_AUTH_LOCK = threading.Lock()


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


def _entry_token(entry: dict) -> str | None:
    token = entry.get("key") or entry.get("access_token")
    if not isinstance(token, str) or not token.strip():
        return None
    return token.strip()


def _load_auth(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _save_auth(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _token_endpoint(issuer: str) -> str:
    return issuer.rstrip("/") + "/oauth2/token"


def post_refresh_token(
    *,
    token_endpoint: str,
    client_id: str,
    refresh_token: str,
) -> dict:
    body = urllib.parse.urlencode(
        {
            "grant_type": "refresh_token",
            "client_id": client_id,
            "refresh_token": refresh_token,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        token_endpoint,
        data=body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(
            f"grok login refresh failed (HTTP {exc.code}): {detail}"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"grok login refresh failed: {exc}") from exc
    if not isinstance(payload, dict) or not payload.get("access_token"):
        raise RuntimeError("grok login refresh returned no access_token")
    return payload


def _apply_refresh(entry: dict, payload: dict, *, now: datetime) -> None:
    access = str(payload["access_token"]).strip()
    entry["key"] = access
    new_refresh = payload.get("refresh_token")
    if isinstance(new_refresh, str) and new_refresh.strip():
        entry["refresh_token"] = new_refresh.strip()
    expires_in = payload.get("expires_in")
    try:
        seconds = int(expires_in)
    except (TypeError, ValueError):
        seconds = 0
    if seconds > 0:
        expiry = now + timedelta(seconds=seconds)
        entry["expires_at"] = (
            expiry.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        )


def _needs_refresh(expires: datetime | None, now: datetime, *, force: bool) -> bool:
    if force:
        return True
    if expires is None:
        return False
    return expires <= now + REFRESH_SKEW


def read_grok_cli_token(
    path: Path | None = None,
    *,
    now: datetime | None = None,
    refresh: bool = True,
    force_refresh: bool = False,
) -> str | None:
    """Unexpired access token from ``grok login``, refreshing if needed."""
    path = path or GROK_AUTH_FILE
    now = now or datetime.now(timezone.utc)
    with _AUTH_LOCK:
        data = _load_auth(path)
        if data is None:
            return None
        changed = False
        best: str | None = None
        best_exp: datetime | None = None
        for entry in data.values():
            if not isinstance(entry, dict):
                continue
            token = _entry_token(entry)
            refresh_token = entry.get("refresh_token")
            expires = _parse_expires_at(entry.get("expires_at"))
            if refresh and isinstance(refresh_token, str) and refresh_token.strip():
                if _needs_refresh(expires, now, force=force_refresh) or not token:
                    issuer = (
                        entry.get("oidc_issuer")
                        if isinstance(entry.get("oidc_issuer"), str)
                        else DEFAULT_OIDC_ISSUER
                    ) or DEFAULT_OIDC_ISSUER
                    client_id = (
                        entry.get("oidc_client_id")
                        if isinstance(entry.get("oidc_client_id"), str)
                        else DEFAULT_OIDC_CLIENT_ID
                    ) or DEFAULT_OIDC_CLIENT_ID
                    try:
                        payload = post_refresh_token(
                            token_endpoint=_token_endpoint(issuer),
                            client_id=client_id,
                            refresh_token=refresh_token.strip(),
                        )
                    except RuntimeError:
                        if token and expires is not None and expires > now:
                            pass
                        else:
                            continue
                    else:
                        _apply_refresh(entry, payload, now=now)
                        token = _entry_token(entry)
                        expires = _parse_expires_at(entry.get("expires_at"))
                        changed = True
            if not token:
                continue
            if expires is not None and expires <= now:
                continue
            if best is None or (
                expires is not None and (best_exp is None or expires > best_exp)
            ):
                best = token
                best_exp = expires
        if changed:
            try:
                _save_auth(path, data)
            except OSError:
                pass
        return best


def refresh_grok_cli_token(path: Path | None = None) -> str | None:
    """Force-refresh the grok login access token and return it."""
    return read_grok_cli_token(path, refresh=True, force_refresh=True)


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
