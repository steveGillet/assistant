from __future__ import annotations

from grapefruit.protocol import (
    TOOLS,
    function_output_event,
    is_quit_command,
    session_update_event,
    user_text_event,
    voice_instructions,
)


def test_quit_commands():
    assert is_quit_command("/quit")
    assert is_quit_command(" EXIT ")
    assert not is_quit_command("please quit later")
    assert not is_quit_command("")


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
