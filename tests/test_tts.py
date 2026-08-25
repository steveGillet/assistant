from __future__ import annotations

from grapefruit.tts import MAX_CHARS, split_text


def test_split_text_empty():
    assert split_text("") == []
    assert split_text("   ") == []


def test_split_text_keeps_short_paragraph():
    text = "Hello. This is fine."
    assert split_text(text, max_chars=80) == [text]


def test_split_text_breaks_on_sentences():
    chunks = split_text("One. Two. Three.", max_chars=5)
    assert chunks == ["One.", "Two.", "Three."]


def test_split_text_splits_long_sentence_on_words():
    chunks = split_text("alpha bravo charlie delta", max_chars=12)
    assert all(len(c) <= 12 for c in chunks)
    assert " ".join(chunks) == "alpha bravo charlie delta"


def test_split_text_never_exceeds_api_cap():
    huge = ("word " * 5000).strip()
    chunks = split_text(huge, max_chars=50_000)
    assert chunks
    assert all(len(c) <= MAX_CHARS for c in chunks)
