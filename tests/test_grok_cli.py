from __future__ import annotations

import json
from types import SimpleNamespace

from grapefruit.grok_cli import (
    apply_stream_line,
    build_grok_cmd,
    parse_cli_output,
    run_grok,
)


def test_parse_cli_output_happy_path():
    raw = json.dumps({"text": "pong", "sessionId": "abc-123", "stopReason": "end_turn"})
    text, sid = parse_cli_output(raw)
    assert text == "pong"
    assert sid == "abc-123"


def test_parse_cli_output_error_object():
    raw = json.dumps({"type": "error", "message": "Couldn't start session"})
    text, sid = parse_cli_output(raw)
    assert "Couldn't start session" in text
    assert sid is None


def test_parse_cli_output_ignores_leading_junk():
    raw = "update available\n" + json.dumps({"text": "done", "sessionId": "sid"})
    text, sid = parse_cli_output(raw)
    assert text == "done"
    assert sid == "sid"


def test_parse_cli_output_empty():
    text, sid = parse_cli_output("")
    assert text == ""
    assert sid is None


def test_build_grok_cmd_resume_and_yolo(tmp_path):
    cmd = build_grok_cmd("/usr/bin/grok", "list files", cwd=tmp_path, session_id="sid-1")
    assert cmd[:3] == ["/usr/bin/grok", "-p", "list files"]
    assert "--yolo" in cmd
    assert "--output-format" in cmd
    assert cmd[cmd.index("--cwd") + 1] == str(tmp_path)
    assert cmd[cmd.index("--resume") + 1] == "sid-1"


def test_streaming_json_collects_text_and_session():
    parts: list[str] = []
    apply_stream_line({"type": "text", "data": "Found "}, parts)
    apply_stream_line({"type": "tool_call", "title": "WebSearch", "status": "in_progress"}, parts)
    sid, err = apply_stream_line(
        {"type": "end", "sessionId": "sid-9", "text": "Found three papers."},
        parts,
    )
    assert sid == "sid-9"
    assert err is False
    assert "".join(parts).startswith("Found")


def test_build_uses_streaming_json(tmp_path):
    cmd = build_grok_cmd("/usr/bin/grok", "hi", cwd=tmp_path)
    assert cmd[cmd.index("--output-format") + 1] == "streaming-json"


def test_run_grok_missing_binary(monkeypatch):
    monkeypatch.setattr("grapefruit.grok_cli.find_grok_bin", lambda: None)
    assert "not found" in run_grok("hello").lower()


def test_run_grok_parses_runner_stdout(monkeypatch, tmp_path):
    monkeypatch.setattr("grapefruit.grok_cli.find_grok_bin", lambda: "/usr/bin/grok")
    session = tmp_path / "session"

    def fake_run(cmd, **kwargs):
        assert "-p" in cmd
        assert "download the paper" in cmd
        payload = {"text": "Downloaded inverse kinematics.pdf", "sessionId": "new-sid"}
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    result = run_grok(
        "download the paper",
        cwd=tmp_path,
        session_path=session,
        runner=fake_run,
    )
    assert "Downloaded" in result
    assert session.read_text() == "new-sid"


def test_run_grok_background_uses_popen(monkeypatch, tmp_path):
    monkeypatch.setattr("grapefruit.grok_cli.find_grok_bin", lambda: "/usr/bin/grok")
    called = {}

    def fake_popen(cmd, **kwargs):
        called["cmd"] = cmd
        return SimpleNamespace(pid=1)

    msg = run_grok(
        "make a podcast",
        background=True,
        cwd=tmp_path,
        session_path=tmp_path / "session",
        popen=fake_popen,
    )
    assert "background" in msg.lower()
    assert called["cmd"][2] == "make a podcast"
