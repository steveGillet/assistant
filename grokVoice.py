import pvporcupine
import pyaudio
import struct
import subprocess
import os
import wave
import json
from elevenlabs.client import ElevenLabs
import io
from pydub import AudioSegment  # Install with: pip install pydub (requires ffmpeg: sudo apt install ffmpeg)
import numpy as np  # For VAD

# Config
PORCUPINE_ACCESS_KEY = os.getenv("PORCUPINE_ACCESS_KEY")
if not PORCUPINE_ACCESS_KEY:
    raise ValueError("PORCUPINE_ACCESS_KEY environment variable not set")

ELEVENLABS_KEY = os.getenv("ELEVENLABS_KEY")
if not ELEVENLABS_KEY:
    raise ValueError("ELEVENLABS_KEY environment variable not set")

WAKE_WORD = "grapefruit"
GROK_CLI_PATH = "/home/steve0gillet/.nvm/versions/node/v24.6.0/bin/grok"
XAI_API_KEY = os.getenv("GROK_API_KEY")
prev_question = None
prev_answer = None

# VAD parameters
SAMPLE_RATE = 16000  # Matches porcupine.sample_rate
CHUNK_SIZE = 512  # Bytes: 256 int16 samples (~16ms at 16000Hz)
ENERGY_THRESHOLD = 1000  # Lowered as per your change; tune based on debug prints
MIN_SPEECH_CHUNKS = 10  # Min chunks with speech to consider valid (~160ms)
MAX_SILENCE_CHUNKS = 50  # Increased to ~0.8s for more tolerance
MAX_RECORD_CHUNKS = 3000  # Max recording time: ~48s to prevent infinite loop

# Init Porcupine
porcupine = pvporcupine.create(
    access_key=PORCUPINE_ACCESS_KEY,
    keywords=[WAKE_WORD]
)

# Init ElevenLabs client (for both STT and TTS)
client = ElevenLabs(api_key=ELEVENLABS_KEY)

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
            speech_chunks = 0
            silence_chunks = 0
            total_chunks = 0
            speech_detected = False
            
            while total_chunks < MAX_RECORD_CHUNKS:
                data = input_stream.read(CHUNK_SIZE)
                chunk_samples = struct.unpack_from("h" * (CHUNK_SIZE // 2), data)
                query_audio.extend(chunk_samples)
                
                # VAD: Compute RMS energy
                chunk_np = np.array(chunk_samples, dtype=np.int16)
                if len(chunk_np) > 0:
                    energy = np.sqrt(np.mean(np.square(chunk_np.astype(np.float64))))
                else:
                    energy = 0.0
                
                # print(f"Chunk {total_chunks}: energy={energy:.2f}")  # Debug print to tune threshold
                
                if energy > ENERGY_THRESHOLD:
                    speech_chunks += 1
                    silence_chunks = 0
                    speech_detected = True
                else:
                    silence_chunks += 1
                
                total_chunks += 1
                
                # Stop if enough silence after speech
                if speech_detected and silence_chunks >= MAX_SILENCE_CHUNKS:
                    print(f"Stopping recording: silence_chunks={silence_chunks}")
                    break
            
            if total_chunks >= MAX_RECORD_CHUNKS:
                print("Warning: Reached max recording length")
            
            # Trim trailing silence
            trim_samples = silence_chunks * (CHUNK_SIZE // 2)
            query_audio = query_audio[:-trim_samples] if len(query_audio) > trim_samples else query_audio
            
            # Silence check (whole recording)
            if speech_chunks < MIN_SPEECH_CHUNKS:
                print("Warning: Insufficient speech detected—skipping.")
                continue
            
            # Save query WAV
            with wave.open("temp_query.wav", "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(struct.pack("h" * len(query_audio), *query_audio))
            
            # Transcribe with ElevenLabs Scribe
            print("Transcribing query...")
            with open("temp_query.wav", "rb") as audio_file:
                transcription = client.speech_to_text.convert(
                    file=audio_file,
                    model_id="scribe_v1",  # Model to use
                    language_code="en"  # 'en' for English; set to None for auto-detection
                )
            query = transcription.text.strip()
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
                
                # TTS with ElevenLabs
                if response:
                    print("Synthesizing response...")
                    
                    tts_stream = client.text_to_speech.convert(
                        text=response,
                        voice_id="CwhRBWXzGAHq8TQ4Fs17",  # Voice ID for 'Rachel'; change via your ElevenLabs dashboard if needed
                        model_id="eleven_turbo_v2",  # Or 'eleven_turbo_v2' for faster synthesis
                        output_format="pcm_16000"  # Raw PCM at 16kHz for WAV compatibility
                    )
                    pcm_data = b''.join(tts_stream)  # Consume generator into bytes
                    
                    # Normalize volume with pydub
                    audio = AudioSegment.from_raw(
                        io.BytesIO(pcm_data),
                        sample_width=2,
                        frame_rate=16000,
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