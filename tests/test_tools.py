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


def test_handle_end_conversation():
    result, should_end = handle_tool("end_conversation", {})
    assert should_end is True
    assert "Goodbye" in result


def test_handle_play_file_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result, should_end = handle_tool("play_file", {"path": "missing.mp3"})
    assert should_end is False
    assert "not found" in result.lower()
