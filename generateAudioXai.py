import asyncio
import websockets
import json
import base64
import os
import io
from pydub import AudioSegment  # pip install pydub (requires ffmpeg: sudo apt install ffmpeg)

# Replace with your actual xAI API key
API_KEY = os.getenv("GROK_API_KEY")  # Set this in your environment: export GROK_API_KEY=your_key_here

# Configuration
VOICE = "Ara"  # Options: Ara (female), Rex (male), Sal (neutral), Eve (female), Leo (male)
SAMPLE_RATE = 24000
TEXT_FILE = "paper.txt"
OUTPUT_FILE = "extracted_audio.wav"

# Read the text from file
with open(TEXT_FILE, "r") as f:
    text = f.read()

# Function to split long text into chunks (safe limit: ~4000 chars)
def split_text(text, max_chars=4000):
    chunks = []
    current_chunk = ""
    for paragraph in text.split("\n\n"):
        if len(current_chunk) + len(paragraph) + 2 > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph
        else:
            if current_chunk:
                current_chunk += "\n\n"
            current_chunk += paragraph
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

text_chunks = split_text(text)
print(f"Split text into {len(text_chunks)} chunks.")

async def generate_tts():
    uri = "wss://api.x.ai/v1/realtime"
    
    async with websockets.connect(uri, additional_headers={"Authorization": f"Bearer {API_KEY}"}) as websocket:
        # Optional: Receive initial connection message
        initial_msg = await websocket.recv()
        print("Connection established:", initial_msg)
        
        # Configure session for TTS (text-only, no turn detection)
        session_message = {
            "type": "session.update",
            "session": {
                "instructions": "You are a text-to-speech reader. Read the provided text aloud exactly as written, without adding, removing, or modifying any content. Do not add commentary.",
                "voice": VOICE,
                "turn_detection": None,  # null for manual/text-only mode
                "audio": {
                    "input": {
                        "format": {
                            "type": "audio/pcm",
                            "rate": SAMPLE_RATE
                        }
                    },
                    "output": {
                        "format": {
                            "type": "audio/pcm",
                            "rate": SAMPLE_RATE
                        }
                    }
                }
            }
        }
        await websocket.send(json.dumps(session_message))
        
        # Receive session update confirmation (optional)
        try:
            conf_msg = await asyncio.wait_for(websocket.recv(), timeout=5)
            print("Session configured:", conf_msg)
        except asyncio.TimeoutError:
            pass  # No confirmation expected
        
        all_audio_bytes = b""
        
        for i, chunk in enumerate(text_chunks):
            print(f"Processing chunk {i+1}/{len(text_chunks)}...")
            
            # Send text input
            text_input = {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": chunk}
                    ]
                }
            }
            await websocket.send(json.dumps(text_input))
            
            # Trigger response generation
            generate_message = {
                "type": "response.create",
                "response": {}
            }
            await websocket.send(json.dumps(generate_message))
            
            # Receive audio chunks
            chunk_audio = b""
            while True:
                msg = await websocket.recv()
                data = json.loads(msg)
                
                if data["type"] == "ping":
                    await websocket.send(json.dumps({"type": "pong"}))
                elif data["type"] == "response.output_audio.delta":
                    chunk_audio += base64.b64decode(data["delta"])
                elif data["type"] == "response.output_audio.done":
                    print(f"Chunk {i+1} audio complete.")
                    break
                elif data["type"] == "error":
                    print("API Error:", data)
                    return
                # Ignore other events (e.g., response.done)
            
            all_audio_bytes += chunk_audio
        
        # Save raw PCM to WAV using pydub
        if all_audio_bytes:
            audio_segment = AudioSegment.from_raw(
                io.BytesIO(all_audio_bytes),
                sample_width=2,  # 16-bit
                frame_rate=SAMPLE_RATE,
                channels=1  # Mono
            )
            audio_segment.export(OUTPUT_FILE, format="wav")
            print(f"Audio generated and saved to {OUTPUT_FILE}")
        else:
            print("No audio generated.")

# Run the async function
asyncio.run(generate_tts())