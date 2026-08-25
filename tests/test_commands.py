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
        lambda query: ("No match for 'nope'. Recent:\nNo saved conversations yet.", None),
    )
    result = handle_line("/restore nope")
    assert result.kind == "print"
    assert "No match" in result.text
