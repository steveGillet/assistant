"""Terminal slash commands. These are handled locally, not sent to Voice."""

from __future__ import annotations

from dataclasses import dataclass

from grapefruit.memory import (
    ConversationLog,
    format_conversation_list,
    list_conversations,
    load_restore_text,
)
from grapefruit.protocol import is_quit_command


@dataclass
class CommandResult:
    kind: str  # passthrough | quit | print | restore | save | help
    text: str = ""
    restore_body: str = ""
    restore_title: str = ""


def parse_slash(text: str) -> tuple[str, str] | None:
    stripped = (text or "").strip()
    if not stripped.startswith("/"):
        return None
    cmd, _, rest = stripped[1:].partition(" ")
    return cmd.lower().strip(), rest.strip()


def help_text() -> str:
    return (
        "Commands:\n"
        "  /quit            end this session (or exit while idle)\n"
        "  /conversations   list saved conversations\n"
        "  /restore         list saved chats\n"
        "  /restore 1       restore by list number\n"
        "  /restore pi      restore by topic words (not the filename)\n"
        "  /save [title]    set the title of the current conversation\n"
        "Anything else is sent to Grok Voice as a user turn."
    )


def handle_line(text: str, log: ConversationLog | None = None) -> CommandResult:
    raw = (text or "").strip()
    if is_quit_command(raw):
        return CommandResult(kind="quit")
    parsed = parse_slash(raw)
    if parsed is None:
        return CommandResult(kind="passthrough", text=raw)
    cmd, arg = parsed
    if cmd in {"quit", "exit", "stop", "goodbye"}:
        return CommandResult(kind="quit")
    if cmd in {"help", "h", "?"}:
        return CommandResult(kind="help", text=help_text())
    if cmd in {"conversations", "list", "history"}:
        return CommandResult(
            kind="print",
            text=format_conversation_list(list_conversations()),
        )
    if cmd in {"restore", "load"}:
        body, meta = load_restore_text(arg)
        if meta is None and not arg:
            return CommandResult(kind="print", text=body)
        if meta is None:
            return CommandResult(kind="print", text=body)
        return CommandResult(
            kind="restore",
            text=f"Restoring {meta.title!r} from {meta.started[:10]}.",
            restore_body=body,
            restore_title=meta.title,
        )
    if cmd == "save":
        if log is None or log.meta is None:
            return CommandResult(kind="print", text="No active conversation to save.")
        if arg:
            log.set_title(arg)
        return CommandResult(
            kind="save",
            text=f"Saved as {log.meta.title!r} ({log.meta.id}).",
        )
    return CommandResult(kind="passthrough", text=raw)
