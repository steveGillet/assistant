from __future__ import annotations

import time

from grapefruit.session import Playback


def test_speaking_while_audio_queued():
    p = Playback(hangover_ms=200)
    p._active = True
    p._done.set()
    assert p.speaking() is False
    p.push(b"\x00\x00")
    assert p.speaking() is True
    p.interrupt()
    assert p.speaking() is True  # hangover
    p._unmute_at = time.monotonic() - 0.05
    assert p.speaking() is False


def test_speaking_inactive():
    p = Playback()
    assert p.speaking() is False
