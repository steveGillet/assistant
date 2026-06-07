import pyaudio
import subprocess
import os
import json
import time
import asyncio
import websockets
import base64
import io
from pydub import AudioSegment
from pydub.playback import play
import queue
import threading
from vosk import Model, KaldiRecognizer
import shlex

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CURRENT_WDR = os.getcwd()

def get_script_file(filename):
    return os.path.join(SCRIPT_DIR, filename)

# === Context File Loading ===
CONTEXT_PATHS = [
    os.path.expanduser("~/.grok/GROK.md"),           # Most common location
    get_script_file(".grok/GROK.md"),                # Local to script
    get_script_file("grok.md"),
    get_script_file("GROK.md"),
]

def load_grok_context():
    for path in CONTEXT_PATHS:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        print(f"[Context] Loaded from {path}")
                        return content
            except Exception as e:
                print(f"[Context] Failed to read {path}: {e}")
    return ""

# Load context once at startup
GROK_CONTEXT = load_grok_context()

CUSTOM_INSTRUCTIONS = """
OPTIMIZE FOR TTS: Use short sentences. Speak naturally.
If the user says "stop", "goodbye", "end session", or similar, call end_conversation immediately.
"""

XAI_API_KEY = os.getenv("GROK_API_KEY")
WAKE_WORD = "grapefruit"

# Path to your downloaded Vosk model
VOSK_MODEL_PATH = get_script_file("vosk-model-small-en-us-0.15")

tools = [
    {
        "type": "function",
        "function": {
            "name": "execute_bash",
            "description": "Executes bash commands.",
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
            "description": "Searches the web.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "end_conversation",
            "description": "Ends the voice session when user says stop/goodbye/etc.",
            "parameters": {"type": "object", "properties": {}}
        }
    }
]

def run_web_search(query):
    print(f"[Web] {query}")
    try:
        from ddgs import DDGS
        return json.dumps(DDGS().text(query, max_results=5))
    except Exception as e:
        return str(e)

SYSTEM_MSG = f"""You are Grapefruit, a helpful local voice assistant running on the user's computer.
Current working directory: {CURRENT_WDR}

{GROK_CONTEXT}

{CUSTOM_INSTRUCTIONS}
"""

def run_bash(command, run_in_background=False):
    print(f"[Bash] {command}")

    # Detect media playback commands and force background mode
    media_commands = ["aplay", "paplay", "vlc", "cvlc", "ffplay", "mpv", "mplayer"]
    first_word = shlex.split(command)[0] if command else ""

    if any(first_word.endswith(player) for player in media_commands):
        run_in_background = True

    if run_in_background:
        subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return f"Playback started in background: {command}"

    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return "Command timed out."

    output_parts = []
    if result.stdout.strip():
        output_parts.append(f"stdout:\n{result.stdout.strip()}")
    if result.stderr.strip():
        output_parts.append(f"stderr:\n{result.stderr.strip()}")
    output_parts.append(f"return_code: {result.returncode}")

    return "\n".join(output_parts)

async def voice_agent_session(api_key):
    uri = "wss://api.x.ai/v1/realtime?model=grok-voice-latest"
    headers = {"Authorization": f"Bearer {api_key}"}

    async with websockets.connect(uri, additional_headers=headers) as ws:
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "instructions": SYSTEM_MSG,
                "voice": "eve",
                "turn_detection": {"type": "server_vad"},
                "tools": tools
            }
        }))

        audio_buffer = b""
        audio_queue = queue.Queue()
        session_active = True
        end_requested = False

        def mic_streamer():
            pa = pyaudio.PyAudio()
            stream = pa.open(rate=24000, channels=1, format=pyaudio.paInt16,
                             input=True, frames_per_buffer=2400)
            print("[Mic] Streaming to Grok...")
            while session_active:
                try:
                    data = stream.read(2400, exception_on_overflow=False)
                    audio_queue.put(base64.b64encode(data).decode("utf-8"))
                except:
                    break
            stream.close()
            pa.terminate()

        threading.Thread(target=mic_streamer, daemon=True).start()
        await asyncio.sleep(0.5)

        # Kick off the first response
        await ws.send(json.dumps({"type": "response.create"}))

        async for message in ws:
            if not session_active:
                break

            event = json.loads(message)

            while not audio_queue.empty():
                await ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": audio_queue.get()
                }))

            if event.get("type") == "response.output_audio.delta":
                audio_buffer += base64.b64decode(event["delta"])

            elif event.get("type") == "response.output_audio.done":
                if audio_buffer:
                    try:
                        seg = AudioSegment.from_raw(io.BytesIO(audio_buffer),
                                                    sample_width=2, frame_rate=24000, channels=1)
                        play(seg)
                    except Exception as e:
                        print(f"[Playback] {e}")
                    audio_buffer = b""

            elif event.get("type") == "response.function_call_arguments.done":
                name = event.get("name")
                args = json.loads(event.get("arguments", "{}"))
                call_id = event.get("call_id")

                if name == "execute_bash":
                    result = run_bash(**args)
                elif name == "web_search":
                    result = run_web_search(**args)
                elif name == "end_conversation":
                    result = "Goodbye!"
                    end_requested = True
                else:
                    result = "Unknown tool"

                await ws.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps({"result": result})
                    }
                }))

                if end_requested:
                    break   # <-- Important: actually exit the loop

                await ws.send(json.dumps({"type": "response.create"}))

            elif event.get("type") == "response.done":
                if end_requested:
                    break

        session_active = False
        print("[Session] Ended cleanly")

        # Cooldown so Vosk doesn't immediately re-trigger
        await asyncio.sleep(1.5)

# ====================== MAIN LOOP (Vosk Wake Word) ======================
print("Loading Vosk model...")
vosk_model = Model(VOSK_MODEL_PATH)
recognizer = KaldiRecognizer(vosk_model, 16000)

pa = pyaudio.PyAudio()
stream = pa.open(rate=16000, channels=1, format=pyaudio.paInt16,
                 input=True, frames_per_buffer=4096)
stream.start_stream()

print("Vosk is listening for 'grapefruit'...")

try:
    while True:
        data = stream.read(4096, exception_on_overflow=False)

        if recognizer.AcceptWaveform(data):
            # Only check FINAL results, not partials
            result = json.loads(recognizer.Result())
            text = result.get("text", "").lower()

            if WAKE_WORD in text:
                print(f"\n[Wake Word Detected: {WAKE_WORD}]")
                subprocess.run(["paplay", get_script_file("ack.wav")], check=False)
                time.sleep(0.2)

                try:
                    asyncio.run(voice_agent_session(XAI_API_KEY))
                except Exception as e:
                    print(f"[Session Error] {e}")

                # Longer cooldown + reset recognizer state
                time.sleep(3.0)
                recognizer = KaldiRecognizer(vosk_model, 16000)  # reset state

        # We are deliberately NOT checking partial results anymore
        # because they were causing constant false triggers

except KeyboardInterrupt:
    print("\nExiting...")
finally:
    stream.stop_stream()
    stream.close()
    pa.terminate()