"""Spawn the official Grok CLI for local computer use.

This is the local agent, not MCP. `grok -p` is a full Grok Build session:
shell, files, web, and any MCP servers configured in ~/.grok/config.toml
or this project's .grok/config.toml. The Voice model never talks to those
MCP servers directly.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path

from grapefruit.env import grok_cli_env
from grapefruit.paths import ROOT, SESSION_FILE
from grapefruit import ui

CLI_TIMEOUT_SEC = int(os.getenv("GROK_CLI_TIMEOUT_SEC", "600"))
SUMMARY_RULES = (
    "You are helping Grapefruit, a voice and terminal assistant. When finished, "
    "reply with a short spoken-friendly summary of two to five sentences. No "
    "markdown tables. The user only hears or sees this summary. Write new files "
    "into the generated/ directory unless the user names another path."
)

Runner = Callable[..., subprocess.CompletedProcess]


def find_grok_bin() -> str | None:
    for candidate in (
        shutil.which("grok"),
        os.path.expanduser("~/.grok/bin/grok"),
        "/usr/local/bin/grok",
    ):
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def load_cli_session(path: Path = SESSION_FILE) -> str | None:
    try:
        sid = path.read_text(encoding="utf-8").strip()
        return sid or None
    except FileNotFoundError:
        return None


def save_cli_session(session_id: str, path: Path = SESSION_FILE) -> None:
    path.write_text(session_id, encoding="utf-8")


def parse_cli_output(raw: str) -> tuple[str, str | None]:
    """Return (summary_text, session_id) from `grok --output-format json`."""
    raw = (raw or "").strip()
    if not raw:
        return "", None

    payload = None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        for line in reversed(raw.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    payload = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
        if payload is None:
            return raw[:1500], None

    if not isinstance(payload, dict):
        return raw[:1500], None

    session_id = payload.get("sessionId")
    text = payload.get("text") or payload.get("message") or ""
    if payload.get("type") == "error":
        return f"Grok CLI error: {payload.get('message', text) or raw[:400]}", session_id
    return (text.strip() or "Grok finished, with no summary."), session_id


def apply_stream_line(event: dict, text_parts: list[str]) -> tuple[str | None, bool]:
    """Handle one streaming-json event.

    Returns (session_id_if_end, is_error).
    """
    typ = event.get("type")
    if typ == "text":
        chunk = event.get("data") or ""
        if chunk:
            text_parts.append(chunk)
    elif typ == "tool_call":
        title = event.get("title") or event.get("toolName") or "tool"
        status = event.get("status") or ""
        if status not in {"completed", "failed", "error"}:
            ui.cli(title)
        elif status in {"failed", "error"}:
            ui.cli(f"{title} {status}")
    elif typ == "tool_call_update":
        title = event.get("title") or event.get("toolName") or "tool"
        status = event.get("status") or ""
        if status in {"failed", "error"}:
            ui.cli(f"{title} {status}")
    elif typ == "end":
        final = event.get("text") or event.get("data")
        if final and not text_parts:
            text_parts.append(final)
        return event.get("sessionId"), False
    elif typ == "error":
        msg = event.get("message") or event.get("data") or "Grok CLI error"
        text_parts.append(str(msg))
        return event.get("sessionId"), True
    return None, False


def _run_streaming(cmd: list[str], cwd: Path, env: dict, timeout: int) -> tuple[int, str, str]:
    """Run grok with streaming-json. Returns (returncode, combined_text, raw_stderr)."""
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    text_parts: list[str] = []
    session_id: str | None = None
    stop_hb = threading.Event()
    err_chunks: list[str] = []

    def read_stderr() -> None:
        if proc.stderr:
            err_chunks.append(proc.stderr.read())

    def heartbeat() -> None:
        elapsed = 0
        while not stop_hb.wait(15):
            elapsed += 15
            ui.cli(f"{elapsed}s")

    hb = threading.Thread(target=heartbeat, daemon=True)
    hb.start()
    threading.Thread(target=read_stderr, daemon=True).start()
    start = time.time()
    try:
        while True:
            if time.time() - start > timeout:
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                raise subprocess.TimeoutExpired(cmd, timeout)
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    break
                time.sleep(0.05)
                continue
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            sid, is_err = apply_stream_line(event, text_parts)
            if sid:
                session_id = sid
            if is_err:
                break
        proc.wait(timeout=10)
        stderr = "".join(err_chunks)
        raw = json.dumps(
            {
                "text": "".join(text_parts).strip(),
                "sessionId": session_id,
                "type": "error" if proc.returncode else None,
            }
        )
        return proc.returncode or 0, raw, stderr or ""
    finally:
        stop_hb.set()
        if proc.poll() is None:
            proc.kill()


def build_grok_cmd(
    grok_bin: str,
    task: str,
    *,
    cwd: Path = ROOT,
    session_id: str | None = None,
) -> list[str]:
    cmd = [
        grok_bin,
        "-p",
        task,
        "--cwd",
        str(cwd),
        "--yolo",
        "--output-format",
        "streaming-json",
        "--no-auto-update",
        "--rules",
        SUMMARY_RULES,
    ]
    if session_id:
        cmd.extend(["--resume", session_id])
    return cmd


def run_grok(
    task: str,
    background: bool = False,
    *,
    cwd: Path = ROOT,
    session_path: Path = SESSION_FILE,
    runner: Runner | None = None,
    popen: Callable | None = None,
) -> str:
    grok_bin = find_grok_bin()
    if not grok_bin:
        return (
            "Official Grok CLI not found. Install it with "
            "curl -fsSL https://x.ai/cli/install.sh | bash"
        )

    session_id = load_cli_session(session_path)
    cmd = build_grok_cmd(grok_bin, task, cwd=cwd, session_id=session_id)
    env = grok_cli_env()
    ui.cli(task[:180])

    if background:
        (popen or subprocess.Popen)(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return f"Started in the background: {task}"

    def invoke(command: list[str]):
        if runner is not None:
            return runner(
                command,
                cwd=str(cwd),
                env=env,
                capture_output=True,
                text=True,
                timeout=CLI_TIMEOUT_SEC,
            )
        from types import SimpleNamespace

        code, raw, stderr = _run_streaming(command, cwd, env, CLI_TIMEOUT_SEC)
        return SimpleNamespace(returncode=code, stdout=raw, stderr=stderr)

    try:
        result = invoke(cmd)
    except subprocess.TimeoutExpired:
        minutes = max(1, CLI_TIMEOUT_SEC // 60)
        return f"Grok CLI timed out after {minutes} minutes."
    except FileNotFoundError:
        return "Official Grok CLI is not installed."

    if result.returncode != 0 and session_id:
        err = (result.stderr or result.stdout or "").lower()
        if "session" in err or "resume" in err or "not found" in err:
            ui.cli("resume failed; new session")
            try:
                session_path.unlink(missing_ok=True)
            except TypeError:
                try:
                    session_path.unlink()
                except FileNotFoundError:
                    pass
            cmd = build_grok_cmd(grok_bin, task, cwd=cwd, session_id=None)
            try:
                result = invoke(cmd)
            except subprocess.TimeoutExpired:
                minutes = max(1, CLI_TIMEOUT_SEC // 60)
                return f"Grok CLI timed out after {minutes} minutes."

    if result.returncode != 0 and not (result.stdout or "").strip():
        err = (result.stderr or "").strip()[:800]
        return f"Grok CLI failed: {err or 'unknown error'}"

    raw = (result.stdout or "").strip()
    if not raw:
        return f"Grok CLI returned no output. stderr: {(result.stderr or '')[:400]}"

    text, new_sid = parse_cli_output(raw)
    if new_sid:
        save_cli_session(new_sid, session_path)
    return text
