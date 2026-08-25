#!/usr/bin/env python3
"""Smoke-test the official xAI Text-to-Speech API (POST /v1/tts)."""

from __future__ import annotations

import argparse

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from grapefruit.env import get_xai_api_key
from grapefruit.tts import synthesize


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a short TTS clip.")
    parser.add_argument(
        "--text",
        default="Hello from the official Grok text to speech API.",
    )
    parser.add_argument("--voice", default="eve")
    parser.add_argument("--output", default="output.mp3")
    args = parser.parse_args()

    audio = synthesize(
        args.text,
        voice=args.voice,
        language="en",
        codec="mp3",
        api_key=get_xai_api_key(),
    )
    with open(args.output, "wb") as f:
        f.write(audio)
    print(f"Saved {len(audio):,} bytes to {args.output}")


if __name__ == "__main__":
    main()
