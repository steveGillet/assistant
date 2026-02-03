import pvporcupine
import pyaudio
import struct
import subprocess
import os
import wave
import json
import io
import time
import torch
import whisper
import numpy as np
from pydub import AudioSegment
from openai import OpenAI
from ddgs import DDGS
import asyncio
import websockets
import base64
import re
# --- DIRECTORY CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CURRENT_WDR = os.getcwd()
def get_script_file(filename):
    return os.path.join(SCRIPT_DIR, filename)
# --- CUSTOM INSTRUCTIONS ---
CUSTOM_INSTRUCTIONS = """
## General Instructions
1. OPTIMIZE FOR TTS: Use short sentences. No markdown tables. Use spoken language.
2. ASSUME SUCCESS: Never report errors for background tasks. If you run a command, assume it worked.
3. CONTEXT: If I mention a file without a path, look in the current directory.
4. BACKGROUND TASKS: ALWAYS use 'run_in_background=True' for:
   - Playing audio/video (VLC)
   - generating scripts (generateScript.py)
   - creating podcasts (podcast.py)
   - downloading large files
## specific Workflows
- **Download/Read Papers**:
  1. Download PDF (wget).
  2. Extract text (pdftotext).
  3. Convert to audio: `python extractAudio.py` (Background=True).
- **Play Research Paper**:
  1. Rename `extracted_audio.wav` to a descriptive name.
  2. Play with `vlc --play-and-exit`. (Background=True).
- **Podcast Creation**:
  - Use: `python podcast.py --input mpc.pdf --output mpcPod.mp3` (Background=True).
- **Script Generation**:
  - Use: `python generateScript.py --input 'Topic'` (Background=True).
  - Output is usually paper.txt.
"""
# --- API KEYS ---
PORCUPINE_ACCESS_KEY = os.getenv("PORCUPINE_ACCESS_KEY")
XAI_API_KEY = os.getenv("GROK_API_KEY")
WAKE_WORD = "grapefruit"
SAMPLE_RATE = 16000
RECORD_SECONDS = 10
# --- MODELS ---
client = OpenAI(api_key=XAI_API_KEY, base_url="https://api.x.ai/v1")
whisper_model = whisper.load_model("large-v3-turbo", device="cpu")
# --- TOOLS ---
tools = [
    {
        "type": "function",
        "function": {
            "name": "execute_bash",
            "description": "Executes bash commands in the current venv.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "run_in_background": {"type": "boolean", "default": False}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Searches the web for live information.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        }
    }
]
# --- HANDLERS ---
def run_bash(command, run_in_background=False):
    print(f"Terminal >> [Background: {run_in_background}] {command}")
    if run_in_background:
        try:
            subprocess.Popen(command, shell=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            return "Background process started."
        except:
            return "Failed to start."
    else:
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=15)
            output = result.stdout or result.stderr
            return output or "Done."
        except:
            return "Error."
def run_web_search(query):
    print(f"Web >> {query}")
    try:
        return json.dumps(DDGS().text(query, max_results=3))
    except Exception as e:
        return f"Error: {str(e)}"
# --- SYSTEM PROMPT ---
SYSTEM_MSG = f"""You are Grapefruit, an automated assistant.
CURRENT DIRECTORY: {CURRENT_WDR}
{CUSTOM_INSTRUCTIONS}
"""
# --- Grok Realtime TTS Functions ---
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
async def generate_realtime_audio(text: str, api_key: str, voice: str = "ara") -> AudioSegment:
    if not text or not text.strip():
        return AudioSegment.empty()
    uri = "wss://api.x.ai/v1/realtime"
    text_chunks = split_long_text(text)
    print(f"Split response into {len(text_chunks)} TTS chunks")
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
                        "instructions": "You are a text repeater for TTS. Your only job is to output the exact text from the user message as speech. Do not add, remove, or change any words. Do not introduce, comment, or respond. Repeat verbatim only.",
                        "turn_detection": {"type": None},
                        "audio": {"output": {"format": {"type": "audio/pcm", "rate": 24000}}},
                        "voice": voice
                    }
                }
                await websocket.send(json.dumps(session_message))
                while True:
                    msg = await websocket.recv()
                    data = json.loads(msg)
                    if data["type"] == "session.updated":
                        break
                for chunk in text_chunks:
                    text_input = {
                        "type": "conversation.item.create",
                        "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": chunk}]}
                    }
                    await websocket.send(json.dumps(text_input))
                    await websocket.send(json.dumps({"type": "response.create", "response": {}}))
                    while True:
                        msg = await websocket.recv()
                        data = json.loads(msg)
                        if data["type"] == "response.output_audio.delta":
                            audio_data += base64.b64decode(data["delta"])
                        elif data["type"] == "response.output_audio.done":
                            break
                        elif data["type"] == "error":
                            raise Exception("Audio error")
                    await asyncio.sleep(0.5)
                break
        except Exception as e:
            retry_count += 1
            print(f"TTS retry {retry_count}: {e}")
            await asyncio.sleep(2 ** retry_count)
    if not audio_data:
        return AudioSegment.empty()
    segment = AudioSegment.from_raw(io.BytesIO(audio_data), sample_width=2, frame_rate=24000, channels=1)
    return segment.normalize()
# --- MAIN LOOP ---
messages = [{"role": "system", "content": SYSTEM_MSG}]
porcupine = pvporcupine.create(access_key=PORCUPINE_ACCESS_KEY, keywords=[WAKE_WORD])
pa = pyaudio.PyAudio()
input_stream = pa.open(rate=porcupine.sample_rate, channels=1, format=pyaudio.paInt16, input=True, frames_per_buffer=porcupine.frame_length)
print(f"Grapefruit Listening in {CURRENT_WDR}")
try:
    while True:
        pcm = input_stream.read(porcupine.frame_length)
        if porcupine.process(struct.unpack_from("h" * porcupine.frame_length, pcm)) >= 0:
            print("\n[Wake Word]")
            subprocess.run("pactl set-source-mute @DEFAULT_SOURCE@ 1", shell=True)
            subprocess.run(["paplay", get_script_file("ack.wav")])
            subprocess.run("pactl set-source-mute @DEFAULT_SOURCE@ 0", shell=True)
            time.sleep(0.5)
           
            # Record
            frames = []
            for _ in range(0, int(SAMPLE_RATE / porcupine.frame_length * RECORD_SECONDS)):
                frames.append(input_stream.read(porcupine.frame_length))
            with wave.open(get_script_file("temp_query.wav"), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(b''.join(frames))
           
            # STT
            user_query = whisper_model.transcribe(get_script_file("temp_query.wav"))["text"].strip()
            if not user_query: continue
            print(f"User: {user_query}")
            messages.append({"role": "user", "content": user_query})
            # Agent loop
            while True:
                response = client.chat.completions.create(model="grok-4-1-fast", messages=messages, tools=tools)
                msg = response.choices[0].message
                if not msg.tool_calls:
                    break
                messages.append(msg)
                for tool in msg.tool_calls:
                    args = json.loads(tool.function.arguments)
                    if tool.function.name == "execute_bash":
                        res = run_bash(args["command"], args.get("run_in_background", False))
                    elif tool.function.name == "web_search":
                        res = run_web_search(args["query"])
                    messages.append({"role": "tool", "tool_call_id": tool.id, "name": tool.function.name, "content": res})
            final_text = response.choices[0].message.content
            print(f"Grok: {final_text}")
            messages.append({"role": "assistant", "content": final_text})
            # TTS with Grok Realtime
            print("Generating speech...")
            audio_segment = asyncio.run(generate_realtime_audio(final_text, XAI_API_KEY, voice="ara"))
            audio_segment.export(get_script_file("temp_res.wav"), format="wav")
            subprocess.run("pactl set-source-mute @DEFAULT_SOURCE@ 1", shell=True)
            subprocess.run(["paplay", get_script_file("temp_res.wav")])
            time.sleep(2)
            subprocess.run("pactl set-source-mute @DEFAULT_SOURCE@ 0", shell=True)
except KeyboardInterrupt:
    pass
finally:
    input_stream.close()
    pa.terminate()
    porcupine.delete()