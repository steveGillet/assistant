"""Optional network tests. Run with: pytest -m live"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest

from grapefruit.env import get_xai_api_key, grok_cli_env
from grapefruit.grok_cli import find_grok_bin, parse_cli_output
from grapefruit.paths import ROOT
from grapefruit.tts import synthesize

pytestmark = pytest.mark.live


def test_grok_cli_json_pong():
    grok = find_grok_bin()
    if not grok:
        pytest.skip("official grok CLI not installed")
    if not get_xai_api_key(required=False) and not os.path.expanduser("~/.grok/auth.json"):
        pytest.skip("no XAI_API_KEY / grok login")
    result = subprocess.run(
        [
            grok,
            "-p",
            "Reply with exactly the single word pong and nothing else.",
            "--cwd",
            str(ROOT),
            "--output-format",
            "json",
            "--no-auto-update",
            "--yolo",
            "--max-turns",
            "1",
            "--disallowed-tools",
            "run_terminal_cmd,web_search,web_fetch,search_replace,read_file,grep,list_dir",
        ],
        capture_output=True,
        text=True,
        env=grok_cli_env(),
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    text, sid = parse_cli_output(result.stdout)
    assert "pong" in text.lower()
    assert sid


def test_tts_tiny_mp3(tmp_path):
    if not get_xai_api_key(required=False):
        pytest.skip("no XAI_API_KEY")
    audio = synthesize("Ping.", voice="eve", language="en", codec="mp3")
    assert len(audio) > 1000
    out = tmp_path / "ping.mp3"
    out.write_bytes(audio)
    assert out.stat().st_size == len(audio)


def test_which_grok():
    assert shutil.which("grok") or find_grok_bin()
    json.dumps({"ok": True})
