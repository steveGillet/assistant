from __future__ import annotations

from datetime import datetime, timedelta, timezone

from grapefruit.memory import (
    ConversationLog,
    Message,
    compact_messages,
    estimate_tokens,
    extractive_summary,
    find_conversation,
    format_conversation_list,
    load_restore_text,
    score_conversation,
    slugify,
)


def test_slugify():
    assert slugify("Robot Manipulator Controllers!") == "robot-manipulator-controllers"
    assert slugify("") == "conversation"


def test_estimate_tokens():
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 40) == 10


def test_append_and_restore(tmp_path):
    log = ConversationLog(root=tmp_path, token_limit=10_000)
    log.start("warmup")
    log.append("user", "find a paper on robotic manipulator controllers", source="typed")
    log.append("assistant", "I'll have Grok look that up.", source="speech")
    log.append("tool", "Downloaded inverse-kinematics.pdf", name="run_grok")
    assert log.path is not None
    assert log.path.is_file()
    text = log.context_for_model()
    assert "robotic manipulator" in text
    assert "Downloaded" in text
    items = __import__("grapefruit.memory", fromlist=["load_index"]).load_index(tmp_path)
    assert items[0].title.startswith("find a paper")


def test_compact_keeps_recent_and_summary():
    msgs = [
        Message(ts="t", role="user", text=f"turn {i} " + ("x" * 50), source="typed")
        for i in range(20)
    ]
    compacted = compact_messages(msgs, keep_last=5)
    assert compacted[0].kind == "compact"
    assert "Earlier conversation" in compacted[0].text
    assert len(compacted) == 6
    assert compacted[-1].text.startswith("turn 19")


def test_log_compacts_when_over_ratio(tmp_path):
    log = ConversationLog(root=tmp_path, token_limit=80)
    log.start("long")
    for i in range(30):
        log.append("user", f"message number {i} with extra padding " + ("word " * 20))
    kinds = [m.kind for m in log.messages]
    assert "compact" in kinds


def test_find_conversation_by_topic_and_yesterday(tmp_path):
    log = ConversationLog(root=tmp_path)
    log.start("other")
    log.append("user", "weather in boulder", source="typed")

    log2 = ConversationLog(root=tmp_path)
    log2.start("robots")
    log2.append("user", "robotic manipulator controllers", source="typed")
    log2.meta.started = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    log2.meta.updated = log2.meta.started
    log2._write_index()

    found = find_conversation("robot manipulators yesterday", root=tmp_path)
    assert found is not None
    assert "manipulator" in found.title.lower() or "robot" in found.title.lower()

    body, meta = load_restore_text("robot manipulators yesterday", root=tmp_path)
    assert meta is not None
    assert "robotic manipulator" in body.lower()


def test_list_is_numbered_and_restore_by_index(tmp_path):
    log = ConversationLog(root=tmp_path)
    log.start("a")
    log.append("user", "weather in boulder", source="typed")
    log2 = ConversationLog(root=tmp_path)
    log2.start("b")
    log2.append("user", "raspberry pi balancing robot", source="typed")
    listing = format_conversation_list(
        __import__("grapefruit.memory", fromlist=["list_conversations"]).list_conversations(tmp_path)
    )
    assert "/restore 1" in listing
    assert "1  " in listing or "  1  " in listing
    found = find_conversation("1", root=tmp_path)
    assert found is not None
    assert "raspberry" in found.title.lower() or "weather" in found.title.lower()


def test_list_empty(tmp_path, monkeypatch):
    monkeypatch.setattr("grapefruit.memory.CONVERSATIONS", tmp_path)
    assert "No saved" in format_conversation_list([])


def test_extractive_summary_mentions_roles():
    msgs = [
        Message(ts="t", role="user", text="hello", source="typed"),
        Message(ts="t", role="assistant", text="hi there", source="speech"),
    ]
    summary = extractive_summary(msgs)
    assert "user:" in summary
    assert "assistant:" in summary


def test_score_prefers_phrase_match(tmp_path):
    from grapefruit.memory import ConversationMeta

    a = ConversationMeta(
        id="1", path="1.jsonl", title="cooking pasta", started="2026-08-23T00:00:00+00:00",
        updated="2026-08-23T00:00:00+00:00", summary="noodles",
    )
    b = ConversationMeta(
        id="2", path="2.jsonl", title="robotic manipulators", started="2026-08-23T00:00:00+00:00",
        updated="2026-08-23T00:00:00+00:00", summary="controllers paper",
    )
    assert score_conversation(b, "robot manipulators") > score_conversation(a, "robot manipulators")
