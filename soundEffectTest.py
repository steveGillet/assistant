import os
import io
import asyncio
import websockets
import json
import base64
from pydub import AudioSegment

async def generate_sound_effect(prompt: str, api_key: str, voice: str = "mara") -> AudioSegment:
    uri = "wss://api.x.ai/v1/realtime"
    audio_data = b""
    retry_count = 0
    max_retries = 3
    while retry_count < max_retries:
        try:
            async with websockets.connect(uri, additional_headers={"Authorization": f"Bearer {api_key}"}) as websocket:
                # Initial recv to establish connection
                await websocket.recv()

                # Set up session for sound effects
                session_message = {
                    "type": "session.update",
                    "session": {
                        "instructions": (
                            "You are a sound effect generator. Interpret the input text as a description of a sound effect "
                            "and produce the corresponding audio. Use vocalizations, onomatopoeia, or imitations to create "
                            "the sound. Do not add any spoken words or explanations unless part of the effect. Output only "
                            "the audio of the sound effect."
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

                # Wait for session updated
                while True:
                    msg = await websocket.recv()
                    data = json.loads(msg)
                    if data["type"] == "session.updated":
                        break
                    elif data["type"] == "error":
                        print("Session error:", data)
                        raise Exception("Session update error")

                # Send the prompt as input text
                text_input = {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": prompt}]
                    }
                }
                await websocket.send(json.dumps(text_input))

                # Trigger response generation
                generate_message = {"type": "response.create", "response": {}}
                await websocket.send(json.dumps(generate_message))

                # Collect audio deltas
                while True:
                    msg = await websocket.recv()
                    data = json.loads(msg)
                    if data["type"] == "response.output_audio.delta":
                        audio_data += base64.b64decode(data["delta"])
                    elif data["type"] == "response.output_audio.done":
                        break
                    elif data["type"] == "error":
                        print("Audio error:", data)
                        raise Exception("Audio generation error")

                print("Sound effect generated successfully.")
                break
        except Exception as e:
            retry_count += 1
            print(f"Generation failed: {e}. Retrying {retry_count}/{max_retries}")
            await asyncio.sleep(2 ** retry_count)

    if retry_count >= max_retries or not audio_data:
        print("Failed to generate sound effect after retries.")
        return AudioSegment.empty()

    # Convert raw audio to AudioSegment
    segment = AudioSegment.from_raw(
        io.BytesIO(audio_data),
        sample_width=2,
        frame_rate=24000,
        channels=1
    )
    return segment

async def main():
    # Get API key
    api_key = os.environ.get("GROK_API_KEY")
    if not api_key:
        raise ValueError("GROK_API_KEY environment variable is required")

    # Test prompt for sound effect (change this to test different effects)
    test_prompt = "Make a realistic explosion sound: Boom! Kaboom!"

    # Generate and save
    audio = await generate_sound_effect(test_prompt, api_key)
    if len(audio) > 0:
        audio.export("test_sound_effect.wav", format="wav")
        print("Saved to test_sound_effect.wav")
    else:
        print("No audio generated.")

if __name__ == "__main__":
    asyncio.run(main())