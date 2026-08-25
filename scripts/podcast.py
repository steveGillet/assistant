import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Literal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
from pydantic import BaseModel
from pydub import AudioSegment
from pypdf import PdfReader

from grapefruit.env import get_xai_api_key
from grapefruit.paths import GENERATED, ensure_dirs
from grapefruit.tts import synthesize_text

TEXT_MODEL = os.getenv("GROK_TEXT_MODEL", "grok-4.6")

class LineItem(BaseModel):
    speaker: Literal["Rachel", "Roger"]
    text: str

class Script(BaseModel):
    script: List[LineItem]

# Step 1: Extract text and split into sections
def extract_sections(input_path: str):
    path = Path(input_path)
    if path.suffix not in ['.pdf', '.txt']:
        raise ValueError("Currently supports PDF or TXT only for section splitting")
    
    if path.suffix == '.pdf':
        reader = PdfReader(path)
        full_text = "\n\n".join(page.extract_text() or '' for page in reader.pages)
        pages = [page.extract_text() or '' for page in reader.pages]
    elif path.suffix == '.txt':
        full_text = path.read_text()
        pages = None  # No pages for TXT
    
    # Detect sections via regex: e.g., "1. Introduction", "2. Methods" (adjust regex as needed)
    section_pattern = re.compile(r'^(\d+\.?\s?[A-Z][A-Za-z\s]+)$', re.MULTILINE)
    matches = list(section_pattern.finditer(full_text))
    
    sections = []
    if matches:
        for i in range(len(matches)):
            start = matches[i].start()
            end = matches[i+1].start() if i+1 < len(matches) else len(full_text)
            title = matches[i].group(1).strip()
            text = full_text[start:end].strip()
            sections.append({'title': title, 'text': text})
    else:
        # Fallback for PDF: Chunk by pages (e.g., 5 pages per section)
        if path.suffix == '.pdf':
            chunk_size = 5
            for i in range(0, len(pages), chunk_size):
                chunk_text = "\n\n".join(pages[i:i+chunk_size])
                sections.append({'title': f"Section {i//chunk_size + 1}", 'text': chunk_text})
        # Fallback for TXT: Chunk by character length (e.g., 5000 chars per section)
        elif path.suffix == '.txt':
            chunk_size = 5000
            for i in range(0, len(full_text), chunk_size):
                chunk_text = full_text[i:i+chunk_size]
                sections.append({'title': f"Section {i//chunk_size + 1}", 'text': chunk_text})
    
    return sections, full_text

# Step 2: Generate podcast script segment using xAI Grok API
def generate_script_segment(content: str, xai_api_key: str, segment_type: str = "discussion", prev_summary: str = "", next_title: str = "") -> Script:
    url = "https://api.x.ai/v1/chat/completions"
    base_prompt = """
You are a podcast producer creating a long, detailed discussion segment.
Hosts: Rachel (female, enthusiastic expert) and Roger (male, analytical co-host). They address each other by name naturally (e.g., 'Roger, what strikes you about this?').
- Alternate speakers frequently.
- Make lines expansive and detailed (300-600+ words each): Explain concepts deeply, use analogies, examples, pros/cons, real-world apps, debates, hypotheticals.
- Aim for 5-10+ minutes of spoken content per segment (many exchanges).
- Do not mention word counts, line lengths, or any meta information about the script in the dialogue.
- Do not use symbols in the dialogue. Instead, describe them in plain words, like 'r dot' for ṙ or 'integral from a to b' for ∫_a^b. This is for better audio quality.
- Output only valid JSON with 'script' key: list of {'speaker': 'Rachel' or 'Roger', 'text': ...}.
"""
    if segment_type == "intro":
        system_prompt = base_prompt + """
- Start with Rachel welcoming listeners and introducing the overall topic from the summary.
- Tease sections ahead.
- End with transition to the background and key concepts section.
- Ground in the full document summary.
"""
        user_content = f"Document summary: {content[:20000]}"
    elif segment_type == "primer":
        system_prompt = base_prompt + f"""
- Discuss and explain the foundational background, key terms, concepts, mathematics, and any necessary prerequisites extracted from the document.
- Assume the listener has general knowledge but not specialized expertise; bring them up to speed on the specifics.
- Use analogies, simple explanations, real-world examples, and break down any math step-by-step.
- Start with a smooth transition from the previous summary (introduction).
- End with teaser/transition to the next section (first content section: '{next_title}').
"""
        user_content = content[:100000]
    elif segment_type == "outro":
        system_prompt = base_prompt + """
- Start with Roger recapping key insights from all sections.
- Discuss implications, future directions.
- End with Rachel thanking listeners and signing off.
"""
        user_content = f"Summary of discussions: {content}"
    else:  # discussion
        system_prompt = base_prompt + f"""
- Discuss this section in depth: {content[:100000]}.
- Start with a smooth transition from previous (if any: '{prev_summary}').
- End with teaser/transition to next section (if any: '{next_title}').
"""
        user_content = content[:100000]
    
    payload = {
        "model": TEXT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {xai_api_key}"}
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"API error: {e}\nResponse: {response.text if 'response' in locals() else ''}")
        raise
    
    content = json.loads(response.json()["choices"][0]["message"]["content"])
    return Script(**content)

def script_to_audio(script: Script, api_key: str) -> AudioSegment:
    audio_segments = []
    pause = AudioSegment.silent(duration=250)

    for line in script.script:
        voice = "ara" if line.speaker == "Rachel" else "rex"
        print(f"TTS {line.speaker} ({voice}): {len(line.text)} chars")
        segment = synthesize_text(
            line.text, voice=voice, language="en", api_key=api_key
        )
        if len(segment) > 0:
            audio_segments.append(segment)
            audio_segments.append(pause)

    if audio_segments:
        audio_segments.pop()
    return sum(audio_segments) if audio_segments else AudioSegment.empty()

# Main CLI entrypoint
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a section-by-section podcast from PDF or TXT using Grok 4.6 and xAI TTS"
    )
    parser.add_argument("--input", required=True, help="Path to input file (PDF or TXT)")
    parser.add_argument(
        "--output",
        default=str(GENERATED / "podcast.mp3"),
        help="Output audio file",
    )
    args = parser.parse_args()
    ensure_dirs()

    xai_key = get_xai_api_key()
    
    sections, full_text = extract_sections(args.input)
    print(f"Detected {len(sections)} sections")
    
    # Generate intro
    intro_script = generate_script_segment(full_text, xai_key, segment_type="intro")
    
    # Generate primer (background/key concepts)
    prev_summary = "The introduction to the topic."
    primer_script = generate_script_segment(full_text, xai_key, segment_type="primer", prev_summary=prev_summary, next_title=sections[0]['title'] if sections else "")
    
    # Generate discussion segments iteratively
    prev_summary = "The background explanation of key terms, concepts, and mathematics."
    discussion_scripts = []
    for i, sec in enumerate(sections):
        next_title = sections[i+1]['title'] if i+1 < len(sections) else ""
        seg_script = generate_script_segment(sec['text'], xai_key, prev_summary=prev_summary, next_title=next_title)
        discussion_scripts.append(seg_script)
        prev_summary = f"Discussion of {sec['title']}: Key points included [briefly summarize in prompt if needed, but keep simple]."
    
    # Generate outro
    outro_summary = " ".join([f"{sec['title']}: [discussed in detail]." for sec in sections])
    outro_script = generate_script_segment(outro_summary, xai_key, segment_type="outro")
    
    # Load jingle
    jingle = AudioSegment.from_wav("jingle.wav")

    # Determine output format
    out_format = "wav" if args.output.endswith(".wav") else "mp3"

    # Build list of temp files
    temp_files = []
    index = 0

    # Function to add part
    def add_part(script):
        global index
        part_audio = jingle + script_to_audio(script, xai_key)
        temp_path = f"temp_part_{index}.{out_format}"
        part_audio.export(temp_path, format=out_format)
        temp_files.append(temp_path)
        index += 1

    # Add parts
    add_part(intro_script)
    add_part(primer_script)
    for disc_script in discussion_scripts:
        add_part(disc_script)
    add_part(outro_script)

    # Create concat list file
    concat_list = "concat_list.txt"
    with open(concat_list, "w") as f:
        for tf in temp_files:
            f.write(f"file '{tf}'\n")

    # Use ffmpeg to concatenate
    subprocess.run([
        "ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_list, "-c", "copy", args.output
    ], check=True)

    # Clean up
    for tf in temp_files:
        os.remove(tf)
    os.remove(concat_list)

    # TODO: Add normalization if needed
    print(f"Podcast generated: {args.output}")