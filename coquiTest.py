import torch
from TTS.api import TTS  # Correct import

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Generate speech (requires a speaker reference for XTTS)
tts.tts_to_file(text="Hello, this is a test of XTTS-v2 running locally.",
                speaker_wav="roger.mp3",  # Required; see below for options
                language="en",
                file_path="output.wav")