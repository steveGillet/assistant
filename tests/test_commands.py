from __future__ import annotations

from grapefruit.commands import handle_line, parse_slash
from grapefruit.memory import ConversationLog


def test_parse_slash():
    assert parse_slash("/restore robot arms") == ("restore", "robot arms")
    assert parse_slash("hello") is None


def test_quit_and_passthrough():
    assert handle_line("/quit").kind == "quit"
    assert handle_line("find a paper").kind == "passthrough"
    assert handle_line("find a paper").text == "find a paper"


def test_help():
    result = handle_line("/help")
    assert result.kind == "help"
    assert "/restore" in result.text
    assert "/resume" in result.text
    assert "/mute" in result.text
    assert "same as /restore" in result.text


def test_save_title(tmp_path):
    log = ConversationLog(root=tmp_path)
    log.start("x")
    log.append("user", "first", source="typed")
    result = handle_line("/save manipulator paper", log)
    assert result.kind == "save"
    assert log.meta is not None
    assert log.meta.title == "manipulator paper"


def test_restore_no_match_lists(tmp_path, monkeypatch):
    monkeypatch.setattr("grapefruit.commands.list_conversations", lambda: [])
    monkeypatch.setattr(
        "grapefruit.commands.load_restore_text",
        lambda query, **kwargs: ("No match for 'nope'. Recent:\nNo saved conversations yet.", None),
    )
    result = handle_line("/restore nope")
    assert result.kind == "print"
    assert "No match" in result.text


def test_resume_is_restore_alias():
    result = handle_line("/resume")
    assert result.kind == "print"


def test_mute_and_unmute_commands():
    assert handle_line("/mute").kind == "mute"
    assert handle_line("/pause").kind == "mute"
    assert handle_line("go quiet").kind == "mute"
    assert handle_line("I'll be back").kind == "mute"
    assert handle_line("mute now").kind == "mute"
    assert handle_line("Can you mute now?").kind == "mute"
    assert handle_line("Go into quiet mode.").kind == "mute"
    assert handle_line("I want you to go into mute mode.").kind == "mute"
    assert handle_line("/unmute").kind == "unmute"
    assert handle_line("I'm back").kind == "unmute"
    assert handle_line("/resume", muted=True).kind == "unmute"
    assert handle_line("/status").kind == "status"


def test_restore_skips_current_session(tmp_path, monkeypatch):
    monkeypatch.setattr("grapefruit.memory.CONVERSATIONS", tmp_path)
    older = ConversationLog(root=tmp_path)
    older.start("robots")
    older.append("user", "robotic manipulator controllers", source="typed")
    current = ConversationLog(root=tmp_path)
    current.start("now")
    current.append("user", "talking about resume", source="typed")
    listed = handle_line("/conversations", current)
    assert listed.kind == "print"
    assert "talking about resume" not in listed.text
    assert "manipulator" in listed.text.lower() or "robot" in listed.text.lower()
    restored = handle_line("/restore 1", current)
    assert restored.kind == "restore"
    assert restored.restore_title != current.meta.title
    resumed = handle_line("/resume 1", current)
    assert resumed.kind == restored.kind
    assert resumed.restore_title == restored.restore_title
    assert "Last you:" in restored.text
    assert "robotic manipulator controllers" in restored.text
    by_name = handle_line("/restore talking about resume", current)
    assert by_name.kind == "print"
