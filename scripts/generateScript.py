#!/usr/bin/env python3
"""Generate a long-form document with Grok 4.6 and write it to paper.txt."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI

from grapefruit.env import get_xai_api_key
from grapefruit.paths import GENERATED, ensure_dirs

MODEL = os.getenv("GROK_TEXT_MODEL", "grok-4.6")
ensure_dirs()
OUTPUT_FILENAME = str(GENERATED / "paper.txt")


def grok_json(client: OpenAI, prompt: str, max_tokens: int) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.7,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content or "{}"


def grok_text(client: OpenAI, prompt: str, max_tokens: int) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.8,
    )
    return (response.choices[0].message.content or "").strip()


def parse_outline(raw: str) -> tuple[str, list[str]]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"Failed to parse outline JSON: {raw[:400]}")
        data = json.loads(raw[start : end + 1])
    title = data.get("title") or "Untitled"
    sections = data.get("sections") or []
    if not isinstance(sections, list) or not sections:
        raise ValueError("Outline JSON had no sections")
    return title, [str(s) for s in sections]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a long-form document using Grok 4.6."
    )
    parser.add_argument("--input", required=True, help="Prompt for the document.")
    parser.add_argument("--output", default=OUTPUT_FILENAME)
    args = parser.parse_args()

    client = OpenAI(base_url="https://api.x.ai/v1", api_key=get_xai_api_key())

    outline_prompt = f"""
Based on the following user prompt: '{args.input}',
Generate a detailed outline for a long-form document (report, short book, story, or script).
Choose the format that fits the prompt.
Output strictly as JSON:
{{
    "title": "Main Title of the Document",
    "sections": [
        "Section 1 Title: Brief description of what this section covers",
        "Section 2 Title: Brief description"
    ]
}}
Aim for 5-15 sections.
"""
    title, sections = parse_outline(grok_json(client, outline_prompt, max_tokens=2000))
    print(f"Outline: {title} ({len(sections)} sections)")

    full_content = [f"# {title}\n"]
    for section in sections:
        if ":" in section:
            section_title, section_desc = section.split(":", 1)
            section_title = section_title.strip()
            section_desc = section_desc.strip()
        else:
            section_title = section.strip()
            section_desc = ""

        section_prompt = f"""
You are writing a section for a long-form document titled '{title}'.
The overall document is based on this user prompt: '{args.input}'.
This specific section is titled '{section_title}' and should cover: {section_desc}.
Write a detailed section. For stories or scripts, use appropriate formatting.
"""
        print(f"Writing: {section_title}")
        section_text = grok_text(client, section_prompt, max_tokens=4096)
        full_content.append(f"## {section_title}\n\n{section_text}\n")

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(full_content))
    print(f"Document saved to '{args.output}'.")


if __name__ == "__main__":
    main()
