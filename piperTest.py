import wave
from piper.voice import PiperVoice

MODEL_PATH = "/home/steve0gillet/piperVoices/en_US-lessac-medium.onnx"  # Your path

voice = PiperVoice.load(MODEL_PATH)

text = "This is a test sentence to check if Piper generates audio."

with wave.open("test.wav", "wb") as wav_file:
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(voice.config.sample_rate if hasattr(voice.config, 'sample_rate') else 22050)
    voice.synthesize_wav(text, wav_file)

print("Generated test.wav—check size and play it.")