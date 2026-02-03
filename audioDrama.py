import torch
from diffusers import AudioLDMPipeline
import torchaudio
import argparse
import json
import re
import asyncio
import websockets
import base64
import time
import io
from pydub import AudioSegment
from pydub.effects import normalize
import os
from typing import List, Literal, Union
from pydantic import BaseModel

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

# Helper to split long text into chunks
def split_long_text(text: str, max_chars: int = 4000):
    if not text.strip():
        return []
    
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
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
            if len(current_chunk) + len(sentence) + (1 if current_chunk else 0) > max_chars:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += (" " + sentence) if current_chunk else sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

# Async function to generate voice audio using Grok Voice API
async def text_to_voice_async(text: str, voice: str, api_key: str) -> AudioSegment:
    uri = "wss://api.x.ai/v1/realtime"
    text_chunks = split_long_text(text)
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
                
                break  # Success
            
        except Exception as e:
            retry_count += 1
            print(f"Connection error: {e}. Retrying {retry_count}/{max_retries}...")
            await asyncio.sleep(2 ** retry_count)
    
    if retry_count >= max_retries:
        raise Exception("Max retries exceeded for voice generation.")
    
    if line_wav_segments:
        full_segment = line_wav_segments[0]
        for seg in line_wav_segments[1:]:
            full_segment = full_segment.append(seg, crossfade=50)
        return full_segment
    return AudioSegment.empty()

def text_to_voice(text: str, voice: str, api_key: str) -> AudioSegment:
    return asyncio.run(text_to_voice_async(text, voice, api_key))

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
    parser.add_argument("--output", default="audio_drama.mp3", help="Output audio file")
    parser.add_argument("--voice-map", default='{"Narrator": "Ara", "Female": "Ara", "Male": "Sal", "Alt Female": "Eve", "Alt Male": "Rex"}', help="JSON dict mapping speakers to voices (ara or rex)")
    args = parser.parse_args()
    
    api_key = os.getenv("GROK_API_KEY")
    if not api_key:
        raise ValueError("GROK_API_KEY not set")
    
    script = parse_drama_script(args.input)
    voice_map = json.loads(args.voice_map)
    generate_audio_drama(script, api_key, args.output, voice_map)