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
