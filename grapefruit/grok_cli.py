"""Spawn the official Grok CLI for local computer use.

This is the local agent, not MCP. `grok -p` is a full Grok Build session:
shell, files, web, and any MCP servers configured in ~/.grok/config.toml
or this project's .grok/config.toml. The Voice model never talks to those
MCP servers directly.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import select
import shutil
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path

from grapefruit.env import grok_cli_env
from grapefruit.paths import ROOT, SESSION_FILE
from grapefruit import ui

CLI_TIMEOUT_SEC = int(os.getenv("GROK_CLI_TIMEOUT_SEC", "0"))
CLI_STALL_SEC = int(os.getenv("GROK_CLI_STALL_SEC", "0"))
CLI_MAX_TURNS = int(os.getenv("GROK_CLI_MAX_TURNS", "0"))
LEADER_SOCK = ROOT / ".grok" / "grapefruit-leader.sock"
SUMMARY_RULES = (
    "You are helping Grapefruit, a voice and terminal assistant. When finished, "
    "reply with a short spoken-friendly summary of two to five sentences. No "
    "markdown tables. The user only hears or sees this summary. "
    "Grapefruit-local papers, audio, podcasts, and downloads go in generated/. "
    "If the user names a file without a path, look in generated/, then cwd, "
    "then assets/ before downloading. "
    "If this task is in another directory or on another machine, edit in place. "
    "Do not copy those files into generated/ unless the user asked for a local copy."
)
_FOREIGN_HOST = re.compile(
    r"\b(ssh|scp|sftp)\b|\b\w+@[\w.-]+|\bon the pi\b|\bon the raspberry\b",
    re.I,
)
_ABS_PATH = re.compile(r"(?:^|[\s'`\"=(])(/home/[\w./-]+|~/[\w./-]+)")

Runner = Callable[..., subprocess.CompletedProcess]


def looks_foreign_work(task: str, *, root: Path = ROOT) -> bool:
    """True when the task is clearly on another host or outside this repo."""
    text = task or ""
    if _FOREIGN_HOST.search(text):
        return True
    try:
        root_resolved = root.resolve()
    except OSError:
        root_resolved = root
    for match in _ABS_PATH.finditer(text):
        raw = os.path.expanduser(match.group(1))
        try:
            path = Path(raw).resolve()
        except OSError:
            continue
        try:
            path.relative_to(root_resolved)
        except ValueError:
            return True
    return False


def summary_rules_for_task(task: str = "", *, root: Path = ROOT) -> str:
    rules = SUMMARY_RULES
    if looks_foreign_work(task, root=root):
        rules += (
            " This task looks like other-directory or remote work. Edit in place. "
            "Do not copy files into generated/ unless the user asked for a local copy."
        )
    return rules


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
        ui.cli(_tool_event_line(event))
    elif typ == "tool_call_update":
        status = event.get("status") or ""
        if status in {"failed", "error", "completed"}:
            ui.cli(_tool_event_line(event))
    elif typ == "end":
        final = event.get("text") or event.get("data")
        if final and not text_parts:
            text_parts.append(final)
        return event.get("sessionId"), False
    elif typ == "error":
        msg = event.get("message") or event.get("data") or "Grok CLI error"
        if "timeout" in str(msg).lower():
            text_parts.append(_timeout_message(CLI_TIMEOUT_SEC))
        else:
            text_parts.append(str(msg))
        return event.get("sessionId"), True
    return None, False


def _tool_event_line(event: dict) -> str:
    title = event.get("title") or event.get("toolName") or event.get("name") or "tool"
    status = event.get("status") or ""
    detail = (
        event.get("args")
        or event.get("input")
        or event.get("command")
        or event.get("preview")
        or event.get("description")
        or ""
    )
    if isinstance(detail, (dict, list)):
        detail = json.dumps(detail, ensure_ascii=False)
    detail = str(detail).strip()
    parts = [str(title)]
    if status:
        parts.append(status)
    if detail:
        parts.append(detail)
    return "  ".join(parts)


def _timeout_message(seconds: int) -> str:
    minutes = max(1, int(seconds) // 60) if seconds >= 60 else 0
    if minutes:
        return (
            f"Grok CLI timed out after {minutes} minutes and was stopped "
            "so it would not keep using credits. Ask again if you want to continue."
        )
    return (
        f"Grok CLI timed out after {int(seconds)} seconds and was stopped "
        "so it would not keep using credits. Ask again if you want to continue."
    )


def _kill_proc_group(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.terminate()
        except Exception:
            pass
    try:
        proc.wait(timeout=2)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except Exception:
            pass
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass


def _run_streaming(cmd: list[str], cwd: Path, env: dict, timeout: int) -> tuple[int, str, str]:
    """Run grok with streaming-json. Returns (returncode, combined_text, raw_stderr)."""
    LEADER_SOCK.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        start_new_session=True,
    )
    assert proc.stdout is not None
    fd = proc.stdout.fileno()
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    text_parts: list[str] = []
    session_id: str | None = None
    stop_wd = threading.Event()
    err_chunks: list[str] = []
    killed_for: list[str] = []
    last_event = [time.time()]
    start = time.time()
    stall = CLI_STALL_SEC if CLI_STALL_SEC > 0 else 0

    def read_stderr() -> None:
        if proc.stderr:
            err_chunks.append(proc.stderr.read().decode("utf-8", errors="replace"))

    def watchdog() -> None:
        elapsed = 0
        while not stop_wd.wait(15):
            elapsed += 15
            quiet = int(time.time() - last_event[0])
            if quiet >= 30:
                ui.cli(f"{elapsed}s · no new CLI events for {quiet}s")
            else:
                ui.cli(f"{elapsed}s")
            now = time.time()
            if timeout > 0 and now - start >= timeout:
                killed_for.append("timeout")
                _kill_proc_group(proc)
                return
            if stall > 0 and now - last_event[0] >= stall:
                killed_for.append("stall")
                _kill_proc_group(proc)
                return

    threading.Thread(target=watchdog, daemon=True).start()
    threading.Thread(target=read_stderr, daemon=True).start()
    buf = b""
    try:
        while True:
            if proc.poll() is not None and not buf:
                break
            if timeout > 0:
                remaining = timeout - (time.time() - start)
                if remaining <= 0:
                    killed_for.append("timeout")
                    _kill_proc_group(proc)
                    break
                wait = min(0.5, max(0.05, remaining))
            else:
                wait = 0.5
            ready, _, _ = select.select([fd], [], [], wait)
            if ready:
                try:
                    chunk = os.read(fd, 65536)
                except BlockingIOError:
                    chunk = b""
                if chunk:
                    buf += chunk
                    while b"\n" in buf:
                        raw_line, buf = buf.split(b"\n", 1)
                        line = raw_line.decode("utf-8", errors="replace").strip()
                        if not line:
                            continue
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(event, dict):
                            continue
                        last_event[0] = time.time()
                        sid, is_err = apply_stream_line(event, text_parts)
                        if sid:
                            session_id = sid
                        if is_err:
                            _kill_proc_group(proc)
                            buf = b""
                            break
                elif proc.poll() is not None:
                    break
            if proc.poll() is not None:
                if buf.strip():
                    line = buf.decode("utf-8", errors="replace").strip()
                    buf = b""
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        event = None
                    if isinstance(event, dict):
                        sid, _ = apply_stream_line(event, text_parts)
                        if sid:
                            session_id = sid
                break
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            _kill_proc_group(proc)
        if killed_for:
            why = "stall" if "stall" in killed_for else "timeout"
            seconds = stall if why == "stall" else timeout
            raise subprocess.TimeoutExpired(cmd, seconds)
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
        stop_wd.set()
        if proc.poll() is None:
            _kill_proc_group(proc)


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
        "--leader-socket",
        str(LEADER_SOCK),
        "--rules",
        summary_rules_for_task(task, root=cwd),
    ]
    if CLI_MAX_TURNS > 0:
        cmd.extend(["--max-turns", str(CLI_MAX_TURNS)])
    if session_id:
        cmd.extend(["--resume", session_id, "--fork-session"])
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
    ui.cli(task)

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
    except subprocess.TimeoutExpired as exc:
        try:
            session_path.unlink(missing_ok=True)
        except TypeError:
            try:
                session_path.unlink()
            except FileNotFoundError:
                pass
        return _timeout_message(int(exc.timeout or CLI_TIMEOUT_SEC))
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
            except subprocess.TimeoutExpired as exc:
                try:
                    session_path.unlink(missing_ok=True)
                except TypeError:
                    try:
                        session_path.unlink()
                    except FileNotFoundError:
                        pass
                return _timeout_message(int(exc.timeout or CLI_TIMEOUT_SEC))

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
