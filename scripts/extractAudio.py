#!/usr/bin/env python3
"""Read paper.txt and synthesize extracted_audio.wav with official xAI TTS."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from grapefruit.env import get_xai_api_key
from grapefruit.paths import GENERATED, ensure_dirs
from grapefruit.tts import synthesize_text


def main() -> None:
    ensure_dirs()
    parser = argparse.ArgumentParser(
        description="Convert paper.txt to extracted_audio.wav using xAI TTS."
    )
    parser.add_argument("--input", default=str(GENERATED / "paper.txt"))
    parser.add_argument("--output", default=str(GENERATED / "extracted_audio.wav"))
    parser.add_argument("--voice", default=os.getenv("GROK_TTS_VOICE", "mara"))
    args = parser.parse_args()

    get_xai_api_key()  # fail fast with a clear message
    if not os.path.isfile(args.input):
        raise SystemExit(f"Input not found: {args.input}")

    with open(args.input, encoding="utf-8") as f:
        text = f.read()
    if not text.strip():
        raise SystemExit(f"{args.input} is empty")

    print(f"Synthesizing {len(text)} characters with voice '{args.voice}'...")
    audio = synthesize_text(text, voice=args.voice, language="en")
    audio.export(args.output, format="wav")
    print(f"Saved {args.output} ({len(audio)} ms)")


if __name__ == "__main__":
    main()
