import argparse
import json
import requests
from pathlib import Path
from pypdf import PdfReader
from pydantic import BaseModel
from typing import List, Literal
import io
from pydub import AudioSegment
import os
import subprocess
import re
import asyncio
import websockets
import base64
import time  # For delays

# Pydantic models for structured script output
def verbalize_math(text: str) -> str:
    replacements = {
        r'\+': ' plus ',
        r'\*': ' times ',
        r'/': ' divided by ',
        r'=': ' equals ',
        r'\^2': ' squared',
        r'\^3': ' cubed',
        r'\^': ' to the power of ',
        r'\(': ' open parenthesis ',
        r'\)': ' close parenthesis ',
    }
    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text)
    return text

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
        "model": "grok-4",
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

# Helper to split long text into chunks at sentence boundaries, and further split long sentences at word boundaries
def split_long_text(text: str, max_chars: int = 4000):  # Increased for Grok
    if not text.strip():
        return []
    
    # Split into sentences using lookbehind for punctuation followed by space
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Filter out empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If the sentence itself is longer than max_chars, split it into word-based subchunks
        if len(sentence) > max_chars:
            words = sentence.split()
            sub_chunk = ""
            for word in words:
                if len(sub_chunk) + len(word) + (1 if sub_chunk else 0) > max_chars:
                    if sub_chunk:
                        chunks.append(sub_chunk.strip())
                    sub_chunk = word
                else:
                    sub_chunk += (" " + word) if sub_chunk else word
            if sub_chunk:
                chunks.append(sub_chunk.strip())
        else:
            # Otherwise, add to current_chunk as before
            if len(current_chunk) + len(sentence) + (1 if current_chunk else 0) > max_chars:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += (" " + sentence) if current_chunk else sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

# Helper function to generate audio from a single script segment using Grok Voice API
async def script_to_audio_async(script: Script, api_key: str) -> AudioSegment:
    uri = "wss://api.x.ai/v1/realtime"
    audio_segments = []
    pause = AudioSegment.silent(duration=250)
    
    for line in script.script:
        voice = "ara" if line.speaker == "Rachel" else "Rex"
        
        text_chunks = split_long_text(line.text)
        print(f"Split {len(line.text)} chars into {len(text_chunks)} chunks for {line.speaker} ({voice})")
        
        line_wav_segments = []
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            try:
                async with websockets.connect(uri, additional_headers={"Authorization": f"Bearer {api_key}"}) as websocket:
                    await websocket.recv()
                    
                    session_message = {
                        "type": "session.update",
                        "session": {
                            "instructions": (
                                "You are a verbatim TTS reader. Output ONLY the exact input text as speech. No paraphrase, improv, summary, explanation, or changes. Word-for-word exact read. "
                                "without adding, removing, changing, or commenting on any content. Do not add introductions, "
                                "summaries, explanations, or any extra words whatsoever. Output only the spoken audio of the text."
                            ),
                            "turn_detection": {"type": None},
                            "audio": {
                                "output": {
                                    "format": {"type": "audio/pcm", "rate": 24000}
                                }
                            },
                            "voice": voice
                        }
                    }
                    await websocket.send(json.dumps(session_message))
                    
                    while True:
                        msg = await websocket.recv()
                        data = json.loads(msg)
                        if data["type"] == "session.updated":
                            break
                        elif data["type"] == "error":
                            print("Error updating session:", data)
                            raise Exception("Session update error")
                    
                    for chunk in text_chunks:
                        text_input = {
                            "type": "conversation.item.create",
                            "item": {
                                "type": "message",
                                "role": "user",
                                "content": [{"type": "input_text", "text": chunk}]
                            }
                        }
                        await websocket.send(json.dumps(text_input))
                        
                        generate_message = {"type": "response.create", "response": {}}
                        await websocket.send(json.dumps(generate_message))
                        
                        audio_data = b""
                        while True:
                            msg = await websocket.recv()
                            data = json.loads(msg)
                            if data["type"] == "response.output_audio.delta":
                                audio_data += base64.b64decode(data["delta"])
                            elif data["type"] == "response.output_audio.done":
                                break
                            elif data["type"] == "error":
                                print("Error:", data)
                                raise Exception("Audio generation error")
                        
                        if audio_data:
                            segment = AudioSegment.from_raw(
                                io.BytesIO(audio_data),
                                sample_width=2,
                                frame_rate=24000,
                                channels=1
                            )
                            line_wav_segments.append(segment)
                        
                        await asyncio.sleep(0.5)
                    
                    break  # Success, exit retry loop
            
            except Exception as e:
                retry_count += 1
                print(f"Connection error for line: {e}. Retrying {retry_count}/{max_retries}...")
                await asyncio.sleep(2 ** retry_count)
        
        if retry_count >= max_retries:
            print("Max retries exceeded for this line. Skipping.")
            continue
        
        if line_wav_segments:
            full_line_segment = line_wav_segments[0]
            for seg in line_wav_segments[1:]:
                full_line_segment = full_line_segment.append(seg, crossfade=50)
            audio_segments.append(full_line_segment)
            audio_segments.append(pause)
    
    if audio_segments:
        audio_segments.pop()
    
    full_segment_audio = sum(audio_segments) if audio_segments else AudioSegment.empty()
    
    return full_segment_audio

def script_to_audio(script: Script, api_key: str) -> AudioSegment:
    return asyncio.run(script_to_audio_async(script, api_key))

# Main CLI entrypoint
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate long, section-by-section podcast from PDF or TXT using Grok API and Grok Voice API")
    parser.add_argument("--input", required=True, help="Path to input file (PDF or TXT)")
    parser.add_argument("--output", default="podcast.mp3", help="Output audio file")
    args = parser.parse_args()
    
    xai_key = os.getenv("GROK_API_KEY")
    if not xai_key:
        raise ValueError("GROK_API_KEY not set")
    
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