"""Official xAI Text-to-Speech helper (POST /v1/tts)."""

from __future__ import annotations

import io
import re
import time
from typing import Iterable

import requests
from pydub import AudioSegment

from grapefruit.env import get_xai_api_key

TTS_URL = "https://api.x.ai/v1/tts"
MAX_CHARS = 15000
DEFAULT_VOICE = "eve"


def split_text(text: str, max_chars: int = 8000) -> list[str]:
    """Split on sentence boundaries, then words, staying under max_chars."""
    text = (text or "").strip()
    if not text:
        return []
    max_chars = min(max_chars, MAX_CHARS)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            words = sentence.split()
            sub = ""
            for word in words:
                candidate = f"{sub} {word}".strip() if sub else word
                if len(candidate) > max_chars:
                    if sub:
                        chunks.append(sub)
                    sub = word
                else:
                    sub = candidate
            if sub:
                chunks.append(sub)
            continue
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def synthesize(
    text: str,
    *,
    voice: str = DEFAULT_VOICE,
    language: str = "en",
    codec: str = "wav",
    sample_rate: int = 24000,
    bit_rate: int | None = None,
    speed: float = 1.0,
    api_key: str | None = None,
    text_normalization: bool = True,
    max_retries: int = 3,
) -> bytes:
    if not text or not text.strip():
        return b""
    if len(text) > MAX_CHARS:
        raise ValueError(f"TTS text exceeds {MAX_CHARS} characters; split first")

    key = api_key or get_xai_api_key()
    output_format: dict = {"codec": codec, "sample_rate": sample_rate}
    if codec == "mp3" and bit_rate is not None:
        output_format["bit_rate"] = bit_rate

    payload = {
        "text": text,
        "voice_id": voice.lower(),
        "language": language,
        "output_format": output_format,
        "speed": speed,
        "text_normalization": text_normalization,
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    last_error: Exception | None = None
    for attempt in range(max_retries):
        response = requests.post(TTS_URL, json=payload, headers=headers, timeout=180)
        if response.ok:
            return response.content
        if response.status_code in (429, 500, 503):
            time.sleep(2 ** attempt)
            last_error = RuntimeError(
                f"TTS {response.status_code}: {response.text[:300]}"
            )
            continue
        raise RuntimeError(f"TTS {response.status_code}: {response.text[:500]}")
    raise last_error or RuntimeError("TTS failed")


def _bytes_to_segment(audio: bytes, codec: str) -> AudioSegment:
    if not audio:
        return AudioSegment.empty()
    if codec == "wav":
        return AudioSegment.from_file(io.BytesIO(audio), format="wav")
    if codec == "mp3":
        return AudioSegment.from_file(io.BytesIO(audio), format="mp3")
    if codec == "pcm":
        return AudioSegment.from_raw(
            io.BytesIO(audio), sample_width=2, frame_rate=24000, channels=1
        )
    return AudioSegment.from_file(io.BytesIO(audio))


def synthesize_segments(
    texts: Iterable[str],
    *,
    voice: str = DEFAULT_VOICE,
    language: str = "en",
    pause_ms: int = 250,
    **kwargs,
) -> AudioSegment:
    combined = AudioSegment.empty()
    silence = AudioSegment.silent(duration=pause_ms) if pause_ms else None
    first = True
    for text in texts:
        for chunk in split_text(text):
            audio = synthesize(
                chunk, voice=voice, language=language, codec="wav", **kwargs
            )
            segment = _bytes_to_segment(audio, "wav")
            if not first and silence is not None:
                combined += silence
            combined += segment
            first = False
    return combined


def synthesize_text(
    text: str,
    *,
    voice: str = DEFAULT_VOICE,
    language: str = "en",
    pause_ms: int = 200,
    **kwargs,
) -> AudioSegment:
    return synthesize_segments(
        [text], voice=voice, language=language, pause_ms=pause_ms, **kwargs
    )
