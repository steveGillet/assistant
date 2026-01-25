import pvporcupine
import pyaudio
import struct
import subprocess
import os
import wave
import json
import io
from pydub import AudioSegment  # Install with: pip install pydub (requires ffmpeg: sudo apt install ffmpeg)
import time
import torch
import whisper
import numpy as np
from TTS.api import TTS

# Config
PORCUPINE_ACCESS_KEY = os.getenv("PORCUPINE_ACCESS_KEY")
if not PORCUPINE_ACCESS_KEY:
    raise ValueError("PORCUPINE_ACCESS_KEY environment variable not set")

GROK_CLI_PATH = "/home/steve0gillet/.nvm/versions/node/v24.6.0/bin/grok"
XAI_API_KEY = os.getenv("GROK_API_KEY")
prev_question = None
prev_answer = None

WAKE_WORD = "grapefruit"

# Audio parameters
SAMPLE_RATE = 16000  # Matches porcupine.sample_rate
RECORD_SECONDS = 15

# Init Porcupine
porcupine = pvporcupine.create(
    access_key=PORCUPINE_ACCESS_KEY,
    keywords=[WAKE_WORD]
)

# Init Whisper for STT
whisper_model = whisper.load_model("large-v3-turbo", device="cpu")  # Or "base" for lighter

# Init Coqui TTS
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
speaker_wav = "roger.mp3"  # Reference for cloning (male voice)

pa = pyaudio.PyAudio()
input_stream = pa.open(
    rate=porcupine.sample_rate,
    channels=1,
    format=pyaudio.paInt16,
    input=True,
    frames_per_buffer=porcupine.frame_length
)

print("Listening for 'Hey Grapefruit'...")

try:
    while True:
        pcm = input_stream.read(porcupine.frame_length)
        pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
        
        keyword_index = porcupine.process(pcm)
        if keyword_index >= 0:
            print("Wake word detected! Recording query...")
            subprocess.run(["paplay", "ack.wav"])
            
            query_audio = []
            start_time = time.time()
            
            while time.time() - start_time < RECORD_SECONDS:
                data = input_stream.read(porcupine.frame_length)
                chunk_samples = struct.unpack_from("h" * porcupine.frame_length, data)
                query_audio.extend(chunk_samples)
            
            # Save query WAV
            with wave.open("temp_query.wav", "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(struct.pack("h" * len(query_audio), *query_audio))
            
            # Transcribe with Whisper
            print("Transcribing query...")
            transcription = whisper_model.transcribe("temp_query.wav", language="en")
            query = transcription["text"].strip()
            print(f"Query: {query}")
            
            if query:
                if prev_question is not None:
                    prompt = f"Previous conversation:\nUser: {prev_question}\nGrok: {prev_answer}\n\nUser: {query}"
                else:
                    prompt = query
                print(f"Executing CLI: {GROK_CLI_PATH} -p '{prompt}'")
                cmd = [GROK_CLI_PATH, "-p", prompt]
                try:
                    cli_result = subprocess.run(cmd, capture_output=True, text=True)
                    raw_output = cli_result.stdout.strip() if cli_result.stdout else cli_result.stderr.strip()
                    
                    # Parse multi-line JSON: split by lines and load each as dict
                    messages = []
                    for line in raw_output.splitlines():
                        if line.strip():  # Skip empty lines
                            try:
                                messages.append(json.loads(line))
                            except json.JSONDecodeError:
                                print(f"Warning: Invalid JSON line: {line}")
                    
                    # Find the last assistant message's content
                    response = ""
                    for msg in reversed(messages):  # Search from end
                        if msg.get("role") == "assistant" and "content" in msg:
                            response = msg["content"]
                            break
                    
                    print(f"Grok: {response}")
                    prev_question = query
                    prev_answer = response
                except Exception as e:
                    print(f"CLI error: {str(e)}")
                    response = "CLI failed"
                
                # TTS with Coqui
                if response:
                    print("Synthesizing response...")
                    
                    # Generate wav as list of floats
                    wav = tts.tts(
                        text=response,
                        speaker_wav=speaker_wav,
                        language="en",
                        split_sentences=True
                    )
                    
                    # Convert to numpy array if not already
                    wav_np = np.array(wav)
                    
                    # Convert to 16-bit PCM bytes
                    pcm_data = (wav_np * 32767).astype(np.int16).tobytes()
                    
                    # Normalize volume with pydub
                    audio = AudioSegment.from_raw(
                        io.BytesIO(pcm_data),
                        sample_width=2,
                        frame_rate=24000,  # XTTS-v2 sample rate
                        channels=1
                    )
                    audio = audio.normalize()  # Boosts volume to max without clipping
                    
                    # Export to WAV
                    audio.export("temp_response.wav", format="wav")

                    print("Playing response...")
                    subprocess.run(["paplay", "temp_response.wav"])
                    os.remove("temp_response.wav")
            
            os.remove("temp_query.wav")

except KeyboardInterrupt:
    pass
finally:
    input_stream.close()
    pa.terminate()
    porcupine.delete()