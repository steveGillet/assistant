"""Resolve xAI credentials for the official Grok API and Grok CLI.

Official env var is XAI_API_KEY. GROK_API_KEY is accepted as a fallback
from the old community CLI.
"""

from __future__ import annotations

import os


def get_xai_api_key(*, required: bool = True) -> str | None:
    key = os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")
    if required and not key:
        raise ValueError(
            "Set XAI_API_KEY (preferred) or GROK_API_KEY. "
            "Create a key at https://console.x.ai"
        )
    return key


def grok_cli_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Env for subprocesses of the official `grok` CLI."""
    env = os.environ.copy()
    key = get_xai_api_key(required=False)
    if key and not env.get("XAI_API_KEY"):
        env["XAI_API_KEY"] = key
    if extra:
        env.update(extra)
    return env
