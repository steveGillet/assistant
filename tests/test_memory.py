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
    format_restore_recap,
    last_exchange,
    load_restore_text,
    score_conversation,
    slugify,
    spoken_restore_confirm,
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


def test_find_conversation_excludes_current(tmp_path):
    first = ConversationLog(root=tmp_path)
    first.start("robots")
    first.append("user", "robotic manipulator controllers", source="typed")
    current = ConversationLog(root=tmp_path)
    current.start("now")
    current.append("user", "talking about resume", source="typed")
    found = find_conversation(
        "talking about resume",
        root=tmp_path,
        exclude_ids=[current.meta.id],
    )
    assert found is None or found.id != current.meta.id
    by_index = find_conversation("1", root=tmp_path, exclude_ids=[current.meta.id])
    assert by_index is not None
    assert by_index.id == first.meta.id


def test_restore_recap_uses_last_exchange():
    msgs = [
        Message(ts="t", role="user", text="ssh into the pi", source="typed"),
        Message(ts="t", role="assistant", text="I'll have Grok take care of that.", source="speech"),
        Message(ts="t", role="tool", text="SSH worked. hostname steve-desktop.", name="run_grok"),
        Message(ts="t", role="assistant", text="SSH worked. The machine is steve-desktop.", source="speech"),
    ]
    user, asst = last_exchange(msgs)
    assert user == "ssh into the pi"
    assert "steve-desktop" in asst
    recap = format_restore_recap(msgs)
    assert recap.startswith("Last you:")
    assert "Last grok:" in recap
    spoken = spoken_restore_confirm("pi robot", recap)
    assert "Restored the conversation about pi robot" in spoken
    assert "ssh into the pi" in spoken
    assert "steve-desktop" in spoken


def test_slash_does_not_rename_session(tmp_path):
    log = ConversationLog(root=tmp_path)
    log.start("")
    log.append("user", "/resume", source="typed")
    assert log.meta is not None
    assert log.meta.title == "Untitled session"


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


def test_last_exchange_skips_restore_mute_and_garbled():
    msgs = [
        Message(ts="t", role="user", text="Set the beta value to point two.", source="speech"),
        Message(
            ts="t",
            role="assistant",
            text="Done. In gyro.py I added fusion.beta = 0.2 after make_fusion.",
            source="speech",
        ),
        Message(ts="t", role="user", text="You", source="speech"),
        Message(
            ts="t",
            role="assistant",
            text="What were you going to say? I'm listening.",
            source="speech",
        ),
        Message(ts="t", role="user", text="Go into quiet mode.", source="speech"),
        Message(ts="t", role="user", text="God, perfection, grapefruit.", source="speech"),
        Message(
            ts="t",
            role="assistant",
            text="Happy to help. Glad everything's working out.",
            source="speech",
        ),
    ]
    user, asst = last_exchange(msgs)
    assert user == "Set the beta value to point two."
    assert "fusion.beta" in asst
    recap = format_restore_recap(msgs)
    assert "grapefruit" not in recap
    assert "quiet mode" not in recap.lower()


def test_recap_from_restore_text_uses_header_only():
    from grapefruit.memory import recap_from_restore_text

    text = (
        "Restored conversation 'For the Raspberry Pi conversation.' from 2026-08-27 "
        "(id 2026-08-27T050619Z_for-the-raspberry-pi-conversation).\n"
        "Last you: Set the beta value to point two.\n"
        "Last grok: Done. In gyro.py I added fusion.beta = 0.2.\n"
        "\n"
        "User (speech): For the Raspberry Pi conversation.\n"
        "tool [restore_conversation]: Prior conversation loaded.\n"
        "Last you: God, perfection, grapefruit.\n"
        "Last grok: Happy to help. Glad everything's working out.\n"
        "Last you: Remove it from the local. You move it.\n"
        "Last grok: Done. It was still sitting in the local workspace.\n"
    )
    recap = recap_from_restore_text(text)
    assert "fusion.beta" in recap
    assert "God, perfection" not in recap
    assert "Remove it from the local" not in recap


def test_render_for_restore_collapses_nested_dumps():
    from grapefruit.memory import render_for_restore

    msgs = [
        Message(
            ts="t",
            role="user",
            text="Can you raspberry pi conversation?",
            source="speech",
        ),
        Message(ts="t", role="assistant", text="I'll bring that back for you.", source="speech"),
        Message(
            ts="t",
            role="tool",
            text=(
                "Prior conversation loaded. Use it as context.\n"
                "Last you: God, perfection, grapefruit.\n"
                "Last grok: Happy to help."
            ),
            name="restore_conversation",
        ),
        Message(
            ts="t",
            role="assistant",
            text="Restored the Raspberry Pi conversation. You were asking to remove something.",
            source="speech",
        ),
        Message(ts="t", role="user", text="Set the beta value to point two.", source="speech"),
        Message(
            ts="t",
            role="assistant",
            text="Done. In gyro.py I added fusion.beta = 0.2.",
            source="speech",
        ),
    ]
    body = render_for_restore(msgs)
    assert "Prior conversation loaded" not in body
    assert "God, perfection" not in body
    assert "already loaded" in body
    assert "fusion.beta" in body
    assert "Can you raspberry pi conversation?" not in body


def test_find_prefers_pi_work_over_restore_shell(tmp_path):
    shell = ConversationLog(root=tmp_path)
    shell.start("shell")
    shell.append("user", "Can you raspberry pi conversation?", source="speech")
    shell.append("assistant", "I'll bring that back for you.", source="speech")
    shell.append(
        "tool",
        "Prior conversation loaded. Use it as context. Tell the user it restored.\n"
        "Restored conversation 'For the Raspberry Pi conversation.'\n"
        "Last you: God, perfection, grapefruit.\n"
        "Last grok: Happy to help.",
        name="restore_conversation",
    )
    shell.append(
        "assistant",
        "Restored the Raspberry Pi conversation. You were asking to remove something.",
        source="speech",
    )
    shell.append("user", "Go into quiet mode.", source="speech")

    work = ConversationLog(root=tmp_path)
    work.start("work")
    work.append(
        "user",
        "i have a raspberry pi controlled self-balancing robot. ssh into it.",
        source="typed",
    )
    work.append("assistant", "I'll have Grok take care of that.", source="speech")
    work.append(
        "tool",
        "SSH worked. Repo is twoWheeledRedemption. controller.py uses Madgwick.",
        name="run_grok",
    )
    work.append(
        "assistant",
        "SSH worked. The balancer uses controller.py and gyro.py on the Pi.",
        source="speech",
    )
    for i in range(6):
        work.append("user", f"check the systemd unit pass {i}", source="typed")
        work.append(
            "assistant",
            "No ExecStop line yet. robot-controller.service still starts controller.py.",
            source="speech",
        )
    work.append("user", "Set the beta value to point two.", source="speech")
    work.append(
        "assistant",
        "Done. In gyro.py I added fusion.beta = 0.2 after make_fusion.",
        source="speech",
    )

    found = find_conversation("raspberry pi", root=tmp_path)
    assert found is not None
    assert found.id == work.meta.id
    body, meta = load_restore_text("raspberry pi", root=tmp_path)
    assert meta is not None
    assert meta.id == work.meta.id
    assert "fusion.beta" in body
    assert "Last you: Set the beta value to point two." in body
    assert "God, perfection" not in body


def test_exclude_ids_skips_recently_restored(tmp_path):
    older = ConversationLog(root=tmp_path)
    older.start("older")
    older.append("user", "raspberry pi self-balancing robot first look", source="typed")
    older.append("assistant", "SSH worked on the Pi.", source="speech")

    recent = ConversationLog(root=tmp_path)
    recent.start("recent")
    recent.append("user", "raspberry pi gyro beta on the balancer", source="typed")
    recent.append("assistant", "Set fusion.beta to 0.2 in gyro.py.", source="speech")

    current = ConversationLog(root=tmp_path)
    current.start("now")
    current.append("user", "hello", source="typed")
    current.note_restored(recent.meta.id)

    found = find_conversation(
        "raspberry pi",
        root=tmp_path,
        exclude_ids=current.exclude_ids(),
    )
    assert found is not None
    assert found.id == older.meta.id
    assert recent.meta.id in current.exclude_ids()
    assert current.meta.id in current.exclude_ids()
