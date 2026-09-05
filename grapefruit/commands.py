"""Terminal slash commands. These are handled locally, not sent to Voice."""

from __future__ import annotations

from dataclasses import dataclass

from grapefruit.memory import (
    ConversationLog,
    format_conversation_list,
    list_conversations,
    load_restore_text,
    recap_from_restore_text,
)
from grapefruit.protocol import (
    is_mute_command,
    is_quit_command,
    is_silent_command,
    is_unmute_command,
    is_unsilent_command,
    is_wake_line,
)


@dataclass
class CommandResult:
    kind: str  # passthrough | quit | print | restore | save | help | mute | unmute | status | silent | unsilent
    text: str = ""
    restore_body: str = ""
    restore_title: str = ""


def parse_slash(text: str) -> tuple[str, str] | None:
    stripped = (text or "").strip()
    if not stripped.startswith("/"):
        return None
    cmd, _, rest = stripped[1:].partition(" ")
    return cmd.lower().strip(), rest.strip()


def help_text(*, silent: bool = False) -> str:
    extra = (
        "Anything else is sent to Grok CLI."
        if silent
        else "Anything else is sent to Grok Voice as a user turn."
    )
    return (
        "Commands:\n"
        "  /quit            end this session (or exit while idle)\n"
        "  /conversations   list saved conversations\n"
        "  /restore         list saved chats (skips this session)\n"
        "  /restore 1       restore by list number\n"
        "  /restore pi      restore by topic words, never the current chat name\n"
        "  /resume          same as /restore; while muted, resumes this session\n"
        "  /silent          CLI only; Voice parks (say grapefruit or /unsilent to talk)\n"
        "  /unsilent        back to Voice (/loud is the same)\n"
        "  /mute            park Voice and wait; jobs keep running (/pause is the same)\n"
        "  /unmute          reopen Voice and hear a recap\n"
        "  /status          show running jobs\n"
        "  /save [title]    set the title of the current conversation\n"
        f"{extra}"
    )


def handle_line(
    text: str,
    log: ConversationLog | None = None,
    *,
    muted: bool = False,
    silent: bool = False,
) -> CommandResult:
    raw = (text or "").strip()
    if is_quit_command(raw):
        return CommandResult(kind="quit")
    if silent:
        if is_unsilent_command(raw) or is_unmute_command(raw) or is_wake_line(raw):
            return CommandResult(kind="unsilent", text="Returning to Voice.")
        if is_silent_command(raw):
            return CommandResult(kind="print", text="Already in silent CLI mode.")
    if not silent:
        if is_silent_command(raw):
            return CommandResult(kind="silent", text="Switching to silent CLI mode.")
        if is_unmute_command(raw):
            return CommandResult(kind="unmute", text="Unmuting this session.")
        if is_mute_command(raw):
            return CommandResult(kind="mute", text="Muting voice. Jobs keep running.")
    parsed = parse_slash(raw)
    if parsed is None:
        return CommandResult(kind="passthrough", text=raw)
    cmd, arg = parsed
    if cmd in {"quit", "exit", "stop", "goodbye"}:
        return CommandResult(kind="quit")
    if cmd in {"help", "h", "?"}:
        return CommandResult(kind="help", text=help_text(silent=silent))
    if cmd == "silent":
        if silent:
            return CommandResult(kind="print", text="Already in silent CLI mode.")
        return CommandResult(kind="silent", text="Switching to silent CLI mode.")
    if cmd in {"unsilent", "loud"}:
        if not silent:
            return CommandResult(kind="print", text="Voice is already live.")
        return CommandResult(kind="unsilent", text="Returning to Voice.")
    if cmd in {"mute", "pause"}:
        return CommandResult(kind="mute", text="Muting voice. Jobs keep running.")
    if cmd in {"unmute", "unpause"}:
        return CommandResult(kind="unmute", text="Unmuting this session.")
    if cmd == "status":
        return CommandResult(kind="status")
    current_ids = [log.meta.id] if log is not None and log.meta is not None else None
    if cmd in {"conversations", "list", "history"}:
        return CommandResult(
            kind="print",
            text=format_conversation_list(list_conversations(exclude_ids=current_ids)),
        )
    if cmd in {"restore", "load", "resume"}:
        if muted and not arg:
            return CommandResult(kind="unmute", text="Unmuting this session.")
        exclude = current_ids
        if arg and not arg.isdigit() and log is not None:
            exclude = log.exclude_ids()
        body, meta = load_restore_text(arg, exclude_ids=exclude)
        if meta is None:
            return CommandResult(kind="print", text=body)
        if log is not None:
            log.note_restored(meta.id)
        recap = recap_from_restore_text(body)
        notice = f"Restored {meta.title!r} from {meta.started[:10]}."
        if recap:
            notice = f"{notice}\n{recap}"
        return CommandResult(
            kind="restore",
            text=notice,
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
