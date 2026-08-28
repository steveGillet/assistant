from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

from grapefruit.grok_cli import (
    apply_stream_line,
    build_grok_cmd,
    looks_foreign_work,
    parse_cli_output,
    run_grok,
    summary_rules_for_task,
    _timeout_message,
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
    assert "--fork-session" in cmd
    assert "--leader-socket" in cmd
    assert "--max-turns" not in cmd


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


def test_looks_foreign_work_detects_ssh_and_outside_paths(tmp_path):
    assert looks_foreign_work("ssh steve@192.168.1.114 and read gyro.py")
    assert looks_foreign_work("edit the unit file on the Pi")
    assert looks_foreign_work("write into /home/steve/Desktop/twoWheeledRedemption/gyro.py")
    assert not looks_foreign_work("download the manipulator papers")
    inside = tmp_path / "generated" / "paper.pdf"
    assert not looks_foreign_work(f"play {inside}", root=tmp_path)


def test_summary_rules_remind_remote_not_to_copy_home():
    local = summary_rules_for_task("download the paper")
    assert "generated/" in local
    assert "look in generated/" in local
    remote = summary_rules_for_task("ssh into the pi and edit gyro.py")
    assert "Edit in place" in remote
    assert "Do not copy files into generated/" in remote


def test_build_grok_cmd_uses_task_placement_rules(tmp_path):
    cmd = build_grok_cmd(
        "/usr/bin/grok",
        "ssh steve@host and fix controller.py",
        cwd=tmp_path,
    )
    rules = cmd[cmd.index("--rules") + 1]
    assert "Edit in place" in rules


def test_timeout_message_mentions_credits():
    msg = _timeout_message(600)
    assert "timed out" in msg.lower()
    assert "credit" in msg.lower()


def test_run_grok_timeout_is_handled(monkeypatch, tmp_path):
    monkeypatch.setattr("grapefruit.grok_cli.find_grok_bin", lambda: "/usr/bin/grok")

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 600)

    result = run_grok(
        "long job",
        cwd=tmp_path,
        session_path=tmp_path / "session",
        runner=fake_run,
    )
    assert "timed out" in result.lower()
    assert "credit" in result.lower()
