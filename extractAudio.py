import os
import io
from pydub import AudioSegment
import asyncio
import websockets
import json
import base64
import re
import time

# Read the text from file
with open("paper.txt", "r") as f:
    text = f.read()

# Get API key
api_key = os.environ.get("XAI_API_KEY")
if not api_key:
    raise ValueError("XAI_API_KEY environment variable is required")

# Default voice
default_voice = "mara"  # Default voice Mara

# Verbalize math
def verbalize_math(text: str) -> str:
    replacements = {
        r'\+': ' plus ',
        r'\*': ' times ',
        r'/': ' divided by ',
        r'=': ' equals ',
        r'\^2': ' squared',
        r'\^3': ' cubed ',
        r'\^': ' to the power of ',
        r'\(': ' open parenthesis ',
        r'\)': ' close parenthesis ',
    }
    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text)
    return text

# Split long text
def split_long_text(text: str, max_chars: int = 1000):
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

# Function to generate audio
async def generate_audio(text: str, api_key: str, voice: str) -> AudioSegment:
    uri = "wss://api.x.ai/v1/realtime"
    text = verbalize_math(text)
    text_chunks = split_long_text(text, max_chars=1000)
    print(f"Split into {len(text_chunks)} chunks")
    audio_data = b""
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
                            "You are a voice actor reading a research paper. Speak only the exact text provided in the user message, "
                            "without adding, removing, changing, or commenting on any content. Output only the spoken audio of the text."
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
                        print("Error:", data)
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
                    await asyncio.sleep(0.5)
                break  # Success
        except Exception as e:
            retry_count += 1
            print(f"Error: {e}. Retrying {retry_count}/{max_retries}...")
            await asyncio.sleep(2 ** retry_count)
    if retry_count >= max_retries:
        raise Exception("Max retries exceeded")
    if not audio_data:
        return AudioSegment.empty()
    segment = AudioSegment.from_raw(
        io.BytesIO(audio_data),
        sample_width=2,
        frame_rate=24000,
        channels=1
    )
    return segment

# Generate
audio_segment = asyncio.run(generate_audio(text, api_key, default_voice))

# Export
audio_segment.export("extracted_audio.wav", format="wav")

print("Audio generated to extracted_audio.wav")