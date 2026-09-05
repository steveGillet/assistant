"""Silent CLI mode: Grok CLI text, no Voice socket."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
import queue

from grapefruit import ui
from grapefruit.commands import handle_line
from grapefruit.grok_cli import run_grok
from grapefruit.memory import ConversationLog
from grapefruit.paths import ROOT


@dataclass
class SilentState:
    log: ConversationLog
    pending_restore: str = ""


def ensure_silent_log(log: ConversationLog | None = None) -> ConversationLog:
    if log is not None:
        return log
    log = ConversationLog()
    log.start("")
    return log


def handle_silent_line(
    text: str,
    state: SilentState,
    *,
    grok: Callable[..., str] = run_grok,
    cwd: Path | None = None,
) -> str:
    """Handle one typed line while silent.

    Returns continue | quit | unsilent.
    """
    cwd = cwd or ROOT
    raw = (text or "").strip()
    if not raw:
        return "continue"
    action = handle_line(raw, state.log, silent=True)
    if action.kind == "quit":
        ui.session("bye")
        return "quit"
    if action.kind == "unsilent":
        ui.status(action.text)
        return "unsilent"
    if action.kind == "silent":
        ui.status("already silent")
        return "continue"
    if action.kind in {"mute", "unmute", "status"}:
        ui.status("silent · type to grok cli · /unsilent or grapefruit for voice")
        return "continue"
    if action.kind in {"print", "help", "save"}:
        ui.info(action.text)
        return "continue"
    if action.kind == "restore":
        state.pending_restore = action.restore_body
        ui.info(action.text)
        if action.restore_title:
            state.log.append(
                "system",
                f"Restored: {action.restore_title}",
                source="restore",
                kind="restore",
            )
        return "continue"
    task = action.text
    ui.user(task)
    state.log.append("user", task, source="typed")
    extra = ""
    if state.pending_restore:
        extra = state.pending_restore
        state.pending_restore = ""
    result = grok(task, cwd=cwd, extra_rules=extra)
    state.log.append("assistant", (result or "")[:1500], source="typed")
    ui.assistant_delta(result or "")
    ui.assistant_end()
    return "continue"


def run_silent(
    *,
    log: ConversationLog | None = None,
    typed: queue.Queue[str] | None = None,
    lines: Iterable[str] | None = None,
    grok: Callable[..., str] = run_grok,
    cwd: Path | None = None,
) -> str:
    """Blocking silent loop for tests and text-only --silent.

    Returns quit or unsilent. Does not open Grok Voice.
    """
    state = SilentState(log=ensure_silent_log(log))
    line_iter = iter(lines) if lines is not None else None
    ui.session("silent · grok cli · /unsilent or grapefruit for voice · /quit ends")
    while True:
        raw = _next_line(line_iter, typed)
        if raw is None:
            return "quit"
        outcome = handle_silent_line(raw, state, grok=grok, cwd=cwd)
        if outcome in {"quit", "unsilent"}:
            return outcome
    return "quit"


def _next_line(
    line_iter,
    typed: queue.Queue[str] | None,
) -> str | None:
    if line_iter is not None:
        try:
            return next(line_iter)
        except StopIteration:
            return None
    if typed is not None:
        try:
            return typed.get()
        except (EOFError, KeyboardInterrupt):
            return None
    try:
        return input()
    except (EOFError, KeyboardInterrupt):
        return None
