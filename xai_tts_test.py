import asyncio
import websockets
import json
import base64
import os

# Replace with your actual xAI API key
API_KEY = os.getenv("GROK_API_KEY")

async def tts_via_voice_agent():
    uri = "wss://api.x.ai/v1/realtime"
    
    async with websockets.connect(uri, additional_headers={"Authorization": f"Bearer {API_KEY}"}) as websocket:
        # Wait for initial connection response
        response = await websocket.recv()
        print("Connection response:", response)
        
        # Send session config with proper nested structure
        session_message = {
            "type": "session.update",
            "session": {
                "instructions": "You are a text-to-speech system. Repeat the user's message verbatim without adding, removing, or modifying any content.",
                "voice": "Ara",  # Options: Ara, Rex, Sal, Eve, Leo
                "turn_detection": {
                    "type": None  # null for manual control (no audio input)
                },
                "audio": {
                    "input": {
                        "format": {
                            "type": "audio/pcm",
                            "rate": 24000
                        }
                    },
                    "output": {
                        "format": {
                            "type": "audio/pcm",
                            "rate": 24000
                        }
                    }
                }
            }
        }
        await websocket.send(json.dumps(session_message))
        
        # Send text input as a conversation item
        text_input = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Hello, this is a test of the xAI text-to-speech API."}
                ]
            }
        }
        await websocket.send(json.dumps(text_input))
        
        # Trigger response creation (manual, since turn_detection is null)
        generate_message = {
            "type": "response.create",
            "response": {}  # Can add options like max_tokens
        }
        await websocket.send(json.dumps(generate_message))
        
        # Receive and handle responses (loop until audio is received)
        audio_data = b""
        while True:
            msg = await websocket.recv()
            data = json.loads(msg)
            
            if data["type"] == "response.output_audio.delta":
                # Append base64-decoded audio chunk
                audio_data += base64.b64decode(data["delta"])
            elif data["type"] == "response.output_audio.done":
                # Audio complete
                break
            elif data["type"] == "error":
                print("Error:", data)
                return
            
            print("Received:", data["type"])
        
        # Save audio (PCM raw; convert to MP3 if needed via pydub or ffmpeg)
        with open("output.raw", "wb") as f:
            f.write(audio_data)
        print("Raw audio saved as output.raw. Convert to MP3 with: ffmpeg -f s16le -ar 24000 -ac 1 -i output.raw output.mp3")

asyncio.run(tts_via_voice_agent())