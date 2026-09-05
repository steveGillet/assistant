from __future__ import annotations

from grapefruit.memory import ConversationLog
from grapefruit.silent import run_silent


def test_silent_sends_lines_to_cli(tmp_path):
    seen = []

    def fake_grok(task, **kwargs):
        seen.append(task)
        return f"did {task}"

    log = ConversationLog(root=tmp_path)
    log.start("silent")
    outcome = run_silent(
        log=log,
        lines=["list files", "/quit"],
        grok=fake_grok,
        cwd=tmp_path,
    )
    assert outcome == "quit"
    assert seen == ["list files"]
    roles = [m.role for m in log.messages]
    assert "user" in roles
    assert "assistant" in roles


def test_silent_restore_is_injected_once(tmp_path):
    seen = []

    def fake_grok(task, **kwargs):
        seen.append({"task": task, "extra": kwargs.get("extra_rules", "")})
        return "ok"

    older = ConversationLog(root=tmp_path)
    older.start("robots")
    older.append("user", "robotic manipulator controllers", source="typed")
    older.append("assistant", "I'll look that up.", source="typed")

    current = ConversationLog(root=tmp_path)
    current.start("now")
    current.append("user", "hello", source="typed")
    outcome = run_silent(
        log=current,
        lines=["/restore robotic manipulator", "continue that work", "/quit"],
        grok=fake_grok,
        cwd=tmp_path,
    )
    assert outcome == "quit"
    assert len(seen) == 1
    assert seen[0]["task"] == "continue that work"
    assert "robotic manipulator" in seen[0]["extra"].lower()
    assert "Prior conversation" not in seen[0]["task"]


def test_silent_unsilent_returns_without_quit():
    seen = []

    def fake_grok(task, **kwargs):
        seen.append(task)
        return "ok"

    outcome = run_silent(
        lines=["list files", "/unsilent"],
        grok=fake_grok,
    )
    assert outcome == "unsilent"
    assert seen == ["list files"]


def test_silent_wake_word_is_unsilent():
    outcome = run_silent(
        lines=["grapefruit"],
        grok=lambda task, **kwargs: "nope",
    )
    assert outcome == "unsilent"


def test_silent_ignores_mute_phrases():
    seen = []

    def fake_grok(task, **kwargs):
        seen.append(task)
        return "ok"

    outcome = run_silent(
        lines=["go quiet", "/quit"],
        grok=fake_grok,
    )
    assert outcome == "quit"
    assert seen == ["go quiet"]
