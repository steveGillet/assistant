import os
import io
from pydub import AudioSegment
import asyncio
import websockets
import json
import base64
import re
import time

async def main():
    # Read the text from file
    with open("paper.txt", "r") as f:
        text = f.read()

    # Get API key
    api_key = os.environ.get("GROK_API_KEY")
    if not api_key:
        raise ValueError("GROK_API_KEY environment variable is required")

    default_voice = "mara"

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

    async def generate_one_chunk(ttext: str, api_key: str, voice: str) -> AudioSegment:
        uri = "wss://api.x.ai/v1/realtime"
        print(f"Chunk preview: {ttext[:100]}...")
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
                                "You are a verbatim TTS reader for papers. Output ONLY the exact input text as speech. No paraphrase, improv, summary, explanation, or changes. Word-for-word exact read. "
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
                            print("Session error:", data)
                            raise Exception("Session update error")

                    text_input = {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": ttext}]
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
                            print("Audio error:", data)
                            raise Exception("Audio generation error")
                    print("Chunk success.")
                    break
            except Exception as e:
                retry_count += 1
                print(f"Chunk fail: {e}. Retry {retry_count}/{max_retries}")
                await asyncio.sleep(2 ** retry_count)
        if retry_count >= max_retries or not audio_data:
            print("Chunk skipped.")
            return AudioSegment.empty()
        segment = AudioSegment.from_raw(
            io.BytesIO(audio_data),
            sample_width=2,
            frame_rate=24000,
            channels=1
        )
        return segment

    # Run
    text_chunks = split_long_text(text, max_chars=1000)
    print(f"Split paper into {len(text_chunks)} chunks.")

    audio_segments = []
    for i, chunk in enumerate(text_chunks):
        print(f"Doing chunk {i+1}/{len(text_chunks)}")
        seg = await generate_one_chunk(chunk, api_key, default_voice)
        audio_segments.append(seg)

    # Assemble
    full_audio = AudioSegment.empty()
    silence_ms = 300
    for idx, seg in enumerate(audio_segments):
        if len(seg) > 0:
            full_audio += seg
            if idx < len(audio_segments) - 1:
                full_audio += AudioSegment.silent(duration=silence_ms)

    full_audio.export("extracted_audio.wav", format="wav")
    print("Saved extracted_audio.wav")

if __name__ == "__main__":
    asyncio.run(main())
