import argparse
import json
import requests
from pathlib import Path
from pypdf import PdfReader
from pydantic import BaseModel
from typing import List, Literal
from elevenlabs.client import ElevenLabs
import io
from pydub import AudioSegment
import os

# Pydantic models for structured script output
class LineItem(BaseModel):
    speaker: Literal["Rachel", "Roger"]  # Updated host names
    text: str

class Script(BaseModel):
    script: List[LineItem]

# Step 1: Extract text from PDF or text file
def extract_text(input_path: str) -> str:
    path = Path(input_path)
    if path.suffix == '.pdf':
        reader = PdfReader(path)
        return "\n\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    elif path.suffix == '.txt':
        return path.read_text()
    else:
        raise ValueError("Input must be .pdf or .txt")

# Step 2: Generate podcast script using xAI Grok API
def generate_script(text: str, xai_api_key: str, max_lines: int = None) -> Script:
    url = "https://api.x.ai/v1/chat/completions"
    system_prompt = """
You are a podcast producer creating a long, in-depth discussion episode from the input text.
Generate a detailed script with two hosts: Rachel (female, enthusiastic expert) and Roger (male, analytical co-host). They should address each other by name naturally (e.g., 'What do you think about that, Roger?' or 'Rachel, let's dive deeper into this.').
- Structure the episode for maximum length: Start with Rachel introducing the topic and overview.
- Break down the content section-by-section with extensive discussion: Key points, technical details, examples, pros/cons, real-world applications, historical context, future implications, and debates.
- Include back-and-forth Q&A, hypotheticals, analogies, and listener question simulations to extend the conversation.
- Alternate speakers frequently (aim for 50-100+ exchanges total).
- Make lines detailed and expansive (200-500+ words each where appropriate) to create a long episode (target 15-30+ minutes when spoken).
- End with Roger summarizing key takeaways and wrapping up.
- Ground everything in the input text, but elaborate extensively.
- Output only valid JSON with the 'script' key containing a list of objects with 'speaker' ('Rachel' or 'Roger') and 'text'.
"""
    payload = {
        "model": "grok-4",  # Current model for best reasoning
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Input text: {text[:128000]}"},  # Increased truncate for longer context
        ],
        "response_format": {"type": "json_object"},  # Enforce JSON
    }
    headers = {"Authorization": f"Bearer {xai_api_key}"}
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"API error: {e}\nResponse: {response.text if 'response' in locals() else 'No response'}")
        raise
    
    content = json.loads(response.json()["choices"][0]["message"]["content"])
    script = Script(**content)
    
    # Optional: Cap lines if specified
    if max_lines:
        script.script = script.script[:max_lines]
    
    return script

# Step 3: Generate audio from script using ElevenLabs TTS
def generate_audio(script: Script, elevenlabs_key: str, output_file: str):
    client = ElevenLabs(api_key=elevenlabs_key)
    rachel_voice = "21m00Tcm4TlvDq8ikWAM"  # Rachel (female)
    roger_voice = "CwhRBWXzGAHq8TQ4Fs17"  # Roger (male)
    
    audio_segments = []
    pause = AudioSegment.silent(duration=500)  # 0.5s pause between lines for natural flow
    
    for line in script.script:
        voice_id = rachel_voice if line.speaker == "Rachel" else roger_voice
        tts_stream = client.text_to_speech.convert(
            text=line.text,
            voice_id=voice_id,
            model_id="eleven_turbo_v2",  # Fast, matches your setup
            output_format="pcm_22050"  # 22kHz for quality
        )
        pcm_data = b''.join(tts_stream)
        
        # Create AudioSegment from PCM
        segment = AudioSegment.from_raw(
            io.BytesIO(pcm_data),
            sample_width=2,  # 16-bit
            frame_rate=22050,
            channels=1  # Mono
        )
        audio_segments.append(segment)
        audio_segments.append(pause)  # Add pause after each line
    
    # Remove last pause
    if audio_segments:
        audio_segments.pop()
    
    # Concatenate all segments
    full_audio = sum(audio_segments) if audio_segments else AudioSegment.empty()
    
    # Export to file (WAV or MP3 based on extension)
    full_audio.export(output_file, format="wav" if output_file.endswith(".wav") else "mp3")
    print(f"Podcast generated: {output_file} (Estimated length: ~{(len(full_audio) / 1000 / 60):.1f} minutes)")

# Main CLI entrypoint
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate long podcast from text/PDF using Grok API and ElevenLabs")
    parser.add_argument("--input", required=True, help="Path to input PDF or text file")
    parser.add_argument("--output", default="podcast.mp3", help="Output audio file (e.g., .mp3 or .wav)")
    parser.add_argument("--max_lines", type=int, default=None, help="Optional: Max script lines to cap length")
    args = parser.parse_args()
    
    xai_key = os.getenv("GROK_API_KEY")
    if not xai_key:
        raise ValueError("GROK_API_KEY environment variable not set")
    
    elevenlabs_key = os.getenv("ELEVENLABS_KEY")
    if not elevenlabs_key:
        raise ValueError("ELEVENLABS_KEY environment variable not set")
    
    text = extract_text(args.input)
    script = generate_script(text, xai_key, args.max_lines)
    generate_audio(script, elevenlabs_key, args.output)