import argparse
import json
import os
from typing import List, Literal, Union

import torch
import torchaudio
from diffusers import AudioLDMPipeline
from pydantic import BaseModel
from pydub import AudioSegment
from pydub.effects import normalize

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from grapefruit.env import get_xai_api_key
from grapefruit.paths import GENERATED, ensure_dirs
from grapefruit.tts import synthesize_text

# Pydantic models for the audio drama script
class DialogueItem(BaseModel):
    type: Literal["dialogue"]
    speaker: str
    text: str

class SFXItem(BaseModel):
    type: Literal["sfx"]
    prompt: str
    duration: float = 3.0  # Default duration in seconds

ScriptItem = Union[DialogueItem, SFXItem]

class DramaScript(BaseModel):
    script: List[ScriptItem]

# Function to generate sound effect using AudioLDM
def generate_sfx(prompt: str, duration: float = 3.0, steps: int = 20) -> AudioSegment:
    pipe = AudioLDMPipeline.from_pretrained('cvssp/audioldm-s-full-v2', torch_dtype=torch.float32)
    pipe = pipe.to('cpu')  # Change to 'cuda' if GPU available
    audio = pipe(prompt, num_inference_steps=steps, audio_length_in_s=duration).audios[0]
    audio_tensor = torch.tensor([audio])
    temp_path = 'temp_sfx.wav'
    torchaudio.save(temp_path, audio_tensor, 16000)
    sfx_segment = AudioSegment.from_wav(temp_path)
    os.remove(temp_path)
    return sfx_segment

def text_to_voice(text: str, voice: str, api_key: str) -> AudioSegment:
    return synthesize_text(text, voice=voice.lower(), language="en", api_key=api_key)

# Parse a text file into DramaScript (assumes format: "SPEAKER: text" or "SFX: prompt ; duration")
def parse_drama_script(file_path: str) -> DramaScript:
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    script_items = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("SFX:"):
            parts = line[4:].strip().split(';')
            prompt = parts[0].strip()
            # duration = float(parts[1].strip()) if len(parts) > 1 else 3.0
            script_items.append(SFXItem(type="sfx", prompt=prompt, duration=3.0))
        elif ':' in line:
            speaker, text = line.split(':', 1)
            speaker = speaker.strip()
            text = text.strip()
            script_items.append(DialogueItem(type="dialogue", speaker=speaker, text=text))
    
    return DramaScript(script=script_items)

# Main function to generate audio drama
def generate_audio_drama(script: DramaScript, api_key: str, output_path: str, voice_map: dict):
    audio_segments = []
    pause = AudioSegment.silent(duration=250)  # Short pause between lines
    
    for item in script.script:
        if item.type == "dialogue":
            voice = voice_map.get(item.speaker, "Ara")  # Default to Ara
            print(f"Generating voice for {item.speaker} ({voice}): {item.text[:50]}...")
            voice_segment = text_to_voice(item.text, voice, api_key)
            audio_segments.append(voice_segment)
        elif item.type == "sfx":
            print(f"Generating SFX: {item.prompt} ({item.duration}s)")
            sfx_segment = generate_sfx(item.prompt, item.duration)
            audio_segments.append(sfx_segment)
        
        audio_segments.append(pause)
    
    if audio_segments:
        audio_segments.pop()  # Remove last pause
    
    full_audio = AudioSegment.empty()
    for seg in audio_segments:
        full_audio += seg
    
    # Normalize audio
    full_audio = normalize(full_audio)
    
    full_audio.export(output_path, format="mp3")
    print(f"Audio drama generated: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Turn audio drama script into audio with voices and SFX")
    parser.add_argument("--input", required=True, help="Path to input script file (TXT with format SPEAKER: text or SFX: prompt ; duration)")
    parser.add_argument(
        "--output",
        default=str(GENERATED / "audio_drama.mp3"),
        help="Output audio file",
    )
    parser.add_argument("--voice-map", default='{"Narrator": "Ara", "Female": "Ara", "Male": "Sal", "Alt Female": "Eve", "Alt Male": "Rex"}', help="JSON dict mapping speakers to voices (ara or rex)")
    args = parser.parse_args()
    ensure_dirs()

    api_key = get_xai_api_key()
    
    script = parse_drama_script(args.input)
    voice_map = json.loads(args.voice_map)
    generate_audio_drama(script, api_key, args.output, voice_map)