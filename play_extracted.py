import os
import wave
from elevenlabs.client import ElevenLabs
import io
from pydub import AudioSegment  # pip install pydub (requires ffmpeg: sudo apt install ffmpeg)

# Config
ELEVENLABS_KEY = os.getenv("ELEVENLABS_KEY")
if not ELEVENLABS_KEY:
    raise ValueError("ELEVENLABS_KEY environment variable not set")

# Read the text from file
with open("paper.txt", "r") as f:
    text = f.read()

# Init ElevenLabs client
client = ElevenLabs(api_key=ELEVENLABS_KEY)

# Function to split long text into chunks (ElevenLabs limit: ~5000 chars per request)
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
    audio_stream = client.text_to_speech.convert(
        text=chunk,
        voice_id="CwhRBWXzGAHq8TQ4Fs17",  # 'Rachel' voice ID; customize via dashboard
        model_id="eleven_turbo_v2",  # Or 'eleven_turbo_v2' for speed
        output_format="pcm_22050"  # Higher quality: 22kHz PCM
    )
    pcm_data = b''.join(audio_stream)
    
    # Create AudioSegment from PCM
    segment = AudioSegment.from_raw(
        io.BytesIO(pcm_data),
        sample_width=2,  # 16-bit
        frame_rate=22050,
        channels=1  # Mono
    )
    audio_segments.append(segment)

# Concatenate all segments
full_audio = sum(audio_segments) if audio_segments else AudioSegment.empty()

# Export to WAV
full_audio.export("extracted_audio.wav", format="wav")

print("Audio generated to extracted_audio.wav")