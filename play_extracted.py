import os
import io
import numpy as np
import torch
from TTS.api import TTS
from pydub import AudioSegment  # pip install pydub (requires ffmpeg: sudo apt install ffmpeg)

# Read the text from file
with open("paper.txt", "r") as f:
    text = f.read()

# Init Coqui TTS
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Function to split long text into chunks (safe limit: ~4000 chars per request to avoid context issues)
def split_text(text, max_chars=4000):
    chunks = []
    current_chunk = ""
    for paragraph in text.split("\n\n"):  # Split by paragraphs
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

# Synthesize audio in chunks and concatenate
text_chunks = split_text(text)
audio_segments = []

for chunk in text_chunks:
    # Generate wav as list of floats
    wav = tts.tts(
        text=chunk,
        speaker_wav="roger.mp3",  # Reference audio for cloning
        language="en",
        split_sentences=True  # Handle sentence splitting internally
    )
    
    # Convert to numpy array if not already
    wav_np = np.array(wav)
    
    # Convert to 16-bit PCM bytes
    pcm_data = (wav_np * 32767).astype(np.int16).tobytes()
    
    # Create AudioSegment from PCM
    segment = AudioSegment.from_raw(
        io.BytesIO(pcm_data),
        sample_width=2,  # 16-bit
        frame_rate=24000,  # XTTS-v2 sample rate
        channels=1  # Mono
    )
    audio_segments.append(segment)

# Concatenate all segments
full_audio = sum(audio_segments) if audio_segments else AudioSegment.empty()

# Export to WAV
full_audio.export("extracted_audio.wav", format="wav")

print("Audio generated to extracted_audio.wav")