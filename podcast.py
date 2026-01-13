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
import re
import numpy as np
import torch
from TTS.api import TTS

# Pydantic models for structured script output
class LineItem(BaseModel):
    speaker: Literal["Rachel", "Roger"]
    text: str

class Script(BaseModel):
    script: List[LineItem]

# Step 1: Extract text and split into sections
def extract_sections(input_path: str):
    path = Path(input_path)
    if path.suffix != '.pdf':
        raise ValueError("Currently supports PDF only for section splitting")
    
    reader = PdfReader(path)
    full_text = "\n\n".join(page.extract_text() or '' for page in reader.pages)
    
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
        # Fallback: Chunk by pages (e.g., 5 pages per section)
        pages = [page.extract_text() or '' for page in reader.pages]
        chunk_size = 5
        for i in range(0, len(pages), chunk_size):
            chunk_text = "\n\n".join(pages[i:i+chunk_size])
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
- Output only valid JSON with 'script' key: list of {'speaker': 'Rachel' or 'Roger', 'text': ...}.
"""
    if segment_type == "intro":
        system_prompt = base_prompt + """
- Start with Rachel welcoming listeners and introducing the overall topic from the summary.
- Tease sections ahead.
- End with transition to first section.
- Ground in the full document summary.
"""
        user_content = f"Document summary: {content[:20000]}"
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

# Helper to split long text into chunks at sentence boundaries (avoid truncation and mid-sentence breaks)
def split_long_text(text: str, max_chars: int = 250):
    # Split into sentences using regex (preserves punctuation)
    sentences = re.findall(r'[^.!?]*[.!?]', text) + [text] if not re.findall(r'[^.!?]*[.!?]', text) else []
    
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(current_chunk) + len(sentence) > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Step 3: Generate audio from script using Coqui TTS
def generate_audio(full_script: Script, output_file: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    
    audio_segments = []
    pause = AudioSegment.silent(duration=500)  # Pause between lines
    
    for line in full_script.script:
        speaker_wav = "jessica.mp3" if line.speaker == "Rachel" else "roger.mp3"
        
        # Split long line into chunks
        text_chunks = split_long_text(line.text)
        print(f"Split {len(line.text)} chars into {len(text_chunks)} chunks for {line.speaker}")
        
        line_wav_segments = []  # Collect chunks for this line
        for chunk in text_chunks:
            wav = tts.tts(
                text=chunk,
                speaker_wav=speaker_wav,
                language="en",
                split_sentences=True
            )
            
            # Convert to numpy array if not already
            wav_np = np.array(wav)
            
            # Convert to 16-bit PCM bytes
            pcm_data = (wav_np * 32767).astype(np.int16).tobytes()
            
            # Create AudioSegment from PCM
            segment = AudioSegment.from_raw(
                io.BytesIO(pcm_data),
                sample_width=2,
                frame_rate=24000,  # XTTS-v2 sample rate
                channels=1
            )
            line_wav_segments.append(segment)
        
        # Concat chunks for this full line with small crossfade for smoothness
        if line_wav_segments:
            full_line_segment = line_wav_segments[0]
            for seg in line_wav_segments[1:]:
                full_line_segment = full_line_segment.append(seg, crossfade=50)  # 50ms overlap to smooth joins
            audio_segments.append(full_line_segment)
            audio_segments.append(pause)
    
    if audio_segments:
        audio_segments.pop()  # Remove last pause
    
    full_audio = sum(audio_segments) if audio_segments else AudioSegment.empty()
    
    # Normalize volume across the entire audio
    full_audio = full_audio.normalize(headroom=-3.0)  # Boosts to max without clipping; adjust headroom if needed
    
    full_audio.export(output_file, format="wav" if output_file.endswith(".wav") else "mp3")
    print(f"Podcast generated: {output_file} (Estimated length: ~{(len(full_audio) / 1000 / 60):.1f} minutes)")
    
# Main CLI entrypoint
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate long, section-by-section podcast from PDF using Grok API and Coqui TTS")
    parser.add_argument("--input", required=True, help="Path to input PDF file")
    parser.add_argument("--output", default="podcast.mp3", help="Output audio file")
    args = parser.parse_args()
    
    xai_key = os.getenv("GROK_API_KEY")
    if not xai_key:
        raise ValueError("GROK_API_KEY not set")
    
    sections, full_text = extract_sections(args.input)
    print(f"Detected {len(sections)} sections")
    
    # Generate intro
    intro_script = generate_script_segment(full_text, xai_key, segment_type="intro")
    
    # Generate discussion segments iteratively
    discussion_scripts = []
    prev_summary = "The introduction to the topic."
    for i, sec in enumerate(sections):
        next_title = sections[i+1]['title'] if i+1 < len(sections) else ""
        seg_script = generate_script_segment(sec['text'], xai_key, prev_summary=prev_summary, next_title=next_title)
        discussion_scripts.append(seg_script)
        prev_summary = f"Discussion of {sec['title']}: Key points included [briefly summarize in prompt if needed, but keep simple]."
    
    # Generate outro
    outro_summary = " ".join([f"{sec['title']}: [discussed in detail]." for sec in sections])
    outro_script = generate_script_segment(outro_summary, xai_key, segment_type="outro")
    
    # Combine all scripts
    full_script = Script(script=intro_script.script + sum((s.script for s in discussion_scripts), []) + outro_script.script)
    
    generate_audio(full_script, args.output)