from __future__ import annotations

from grapefruit.tools import handle_tool, resolve_media_path


def test_resolve_media_path_finds_file(tmp_path):
    media = tmp_path / "clip.wav"
    media.write_bytes(b"RIFF")
    found = resolve_media_path("clip.wav", cwd=str(tmp_path), project=tmp_path)
    assert found == str(media.resolve())


def test_resolve_media_path_generated(tmp_path):
    gen = tmp_path / "generated"
    gen.mkdir()
    media = gen / "talk.mp3"
    media.write_bytes(b"ID3")
    found = resolve_media_path("talk.mp3", cwd=str(tmp_path), project=tmp_path)
    assert found == str(media.resolve())


def test_resolve_media_path_missing(tmp_path):
    assert resolve_media_path("nope.wav", cwd=str(tmp_path), project=tmp_path) is None


def test_handle_web_search_dispatches(monkeypatch):
    monkeypatch.setattr(
        "grapefruit.tools.web_search",
        lambda query, **kwargs: f"brief:{query}",
    )
    result, should_end = handle_tool("web_search", {"query": "Madgwick beta", "num_results": 10})
    assert should_end is False
    assert result == "brief:Madgwick beta"


def test_handle_unknown_tool():
    result, should_end = handle_tool("not_a_tool", {})
    assert "Unknown" in result
    assert should_end is False


def test_handle_restore_excludes_current(monkeypatch):
    seen = {}

    def fake_restore(query, exclude_ids=None, **kwargs):
        seen["exclude_ids"] = exclude_ids
        seen["query"] = query
        return "No match for 'resume'. Recent:\n  1  2026-08-25  robot arms", None

    monkeypatch.setattr("grapefruit.tools.load_restore_text", fake_restore)
    result, should_end = handle_tool(
        "restore_conversation",
        {"query": "resume"},
        exclude_ids=["current-id"],
    )
    assert should_end is False
    assert seen["exclude_ids"] == ["current-id"]
    assert seen["query"] == "resume"
    assert "No match" in result


def test_handle_restore_tells_voice_not_to_restore_again(monkeypatch):
    def fake_restore(query, exclude_ids=None, **kwargs):
        from grapefruit.memory import ConversationMeta

        meta = ConversationMeta(
            id="pi-1",
            path="pi-1.jsonl",
            title="pi robot",
            started="2026-08-25T00:00:00+00:00",
            updated="2026-08-25T00:00:00+00:00",
        )
        body = (
            "Restored conversation 'pi robot' from 2026-08-25 (id pi-1).\n"
            "Last you: Set the beta value to point two.\n"
            "Last grok: Done. fusion.beta = 0.2.\n"
        )
        return body, meta

    monkeypatch.setattr("grapefruit.tools.load_restore_text", fake_restore)
    result, should_end = handle_tool("restore_conversation", {"query": "raspberry pi"})
    assert should_end is False
    assert "Do not call restore_conversation again" in result
    assert "fusion.beta" in result


def test_handle_end_conversation():
    result, should_end = handle_tool("end_conversation", {})
    assert should_end is True
    assert "Goodbye" in result


def test_handle_mute_conversation_does_not_end():
    result, should_end = handle_tool("mute_conversation", {})
    assert should_end is False
    assert "parked" in result.lower()
    alias, alias_end = handle_tool("mute", {})
    assert alias_end is False
    assert "parked" in alias.lower()


def test_handle_play_file_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result, should_end = handle_tool("play_file", {"path": "missing.mp3"})
    assert should_end is False
    assert "not found" in result.lower()
