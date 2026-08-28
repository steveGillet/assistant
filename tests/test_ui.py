from __future__ import annotations

from grapefruit import ui


def test_assistant_line_is_closed_before_other_output(capsys):
    ui.assistant_delta("Hello")
    ui.user("typed after")
    out = capsys.readouterr().out
    assert "Hello" in out
    assert "typed after" in out
    hello_at = out.find("Hello")
    you_at = out.find("you")
    assert hello_at < you_at
    between = out[hello_at : you_at]
    assert "\n" in between


def test_tool_result_is_not_ellipsized(capsys):
    body = "First sentence of a long Grok CLI summary.\nSecond sentence with details.\n" + (
        "word " * 200
    )
    ui.tool_result(body)
    out = capsys.readouterr().out
    assert "First sentence of a long Grok CLI summary." in out
    assert "Second sentence with details." in out
    assert "..." not in out
    assert "word" in out


def test_cli_prints_full_task(capsys):
    task = "Download the five manipulator papers and convert each one to a podcast mp3 in generated/"
    ui.cli(task)
    out = capsys.readouterr().out.replace("\n", " ")
    for word in ("Download", "manipulator", "papers", "podcast", "generated"):
        assert word in out
    assert "..." not in out
