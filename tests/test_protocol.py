from __future__ import annotations

from grapefruit.protocol import (
    TOOLS,
    function_output_event,
    is_mute_command,
    is_quit_command,
    is_silent_command,
    is_unmute_command,
    is_unsilent_command,
    is_wake_line,
    session_update_event,
    user_text_event,
    voice_instructions,
)


def test_quit_commands():
    assert is_quit_command("/quit")
    assert is_quit_command(" EXIT ")
    assert not is_quit_command("please quit later")
    assert not is_quit_command("")


def test_mute_phrases():
    assert is_mute_command("/mute")
    assert is_mute_command("go quiet")
    assert is_mute_command("I'll be back.")
    assert is_mute_command("mute")
    assert is_mute_command("Mute now.")
    assert is_mute_command("Can you mute now?")
    assert is_mute_command("and then mute")
    assert is_mute_command(
        "Can you check that for me? And then mute."
    )
    assert is_mute_command("Mute now. A mute tool or a quiet tool that you can use.")
    assert is_mute_command("Go into quiet mode.")
    assert is_mute_command("go into mute mode")
    assert is_mute_command("quiet mode")
    assert is_mute_command("I want you to go into mute mode.")
    assert is_mute_command("Can you go into quiet mode?")
    assert not is_mute_command("mute the television please")
    assert not is_mute_command("unmute now")
    assert is_unmute_command("I'm back")
    assert is_unmute_command("/unmute")
    assert not is_unmute_command("back later maybe")
    assert is_silent_command("/silent")
    assert is_silent_command("silent")
    assert is_silent_command("Go into silent mode.")
    assert is_silent_command("Can you go silent?")
    assert not is_silent_command("silent film please")
    assert not is_silent_command("unsilent")
    assert is_unsilent_command("/unsilent")
    assert is_unsilent_command("go loud")
    assert is_unsilent_command("go into loud mode")
    assert is_wake_line("grapefruit")
    assert is_wake_line("Hey grapefruit.")
    assert not is_wake_line("grapefruit conversation")


def test_user_text_event_shape():
    event = user_text_event("play the paper")
    assert event["type"] == "conversation.item.create"
    assert event["item"]["role"] == "user"
    assert event["item"]["content"][0] == {
        "type": "input_text",
        "text": "play the paper",
    }


def test_function_output_event_shape():
    event = function_output_event("call_1", "Playing /tmp/a.wav")
    assert event["item"]["type"] == "function_call_output"
    assert event["item"]["call_id"] == "call_1"
    assert "Playing" in event["item"]["output"]


def test_tools_include_run_grok_and_native_search():
    types = {t["type"] for t in TOOLS}
    names = {t.get("name") for t in TOOLS}
    assert "web_search" in types
    assert "x_search" in types
    assert "run_grok" in names
    assert "play_file" in names
    assert "end_conversation" in names
    assert "mute_conversation" in names
    assert "silent_mode" in names
    assert "restore_conversation" in names
    nested = [t for t in TOOLS if "function" in t and isinstance(t.get("function"), dict)]
    assert nested == [], "Voice tools must use the flat xAI schema, not OpenAI nested function"


def test_session_update_uses_flat_function_tools():
    event = session_update_event(voice="eve", cwd="/tmp", project="/tmp/assistant")
    assert event["type"] == "session.update"
    assert event["session"]["voice"] == "eve"
    assert event["session"]["turn_detection"]["type"] == "server_vad"
    run = next(t for t in event["session"]["tools"] if t.get("name") == "run_grok")
    assert run["type"] == "function"
    assert "parameters" in run


def test_instructions_mention_typed_and_spoken():
    text = voice_instructions(cwd="/home/me", project="/home/me/assistant")
    assert "type" in text.lower()
    assert "run_grok" in text
    assert "/home/me/assistant" in text
    assert "never this session's title" in text.lower() or "never the current" in text.lower()
    assert "never call `end_conversation` for mute" in text.lower()
    assert "quiet mode" in text.lower()
    assert "do not call `restore_conversation` again" in text.lower()
    assert "mute_conversation" in text
    assert "silent_mode" in text
    assert "never treat that as goodbye" in text.lower() or "silent_mode" in text
    assert "edit in place" in text.lower()
    assert "do not copy" in text.lower()
    mute = next(t for t in TOOLS if t.get("name") == "mute_conversation")
    assert "quiet" in mute["description"].lower()
    assert "goodbye" in mute["description"].lower()


def test_restore_tool_rejects_current_title():
    tool = next(t for t in TOOLS if t.get("name") == "restore_conversation")
    desc = tool["description"].lower()
    assert "never" in desc
    assert "title" in desc or "current" in desc
