import whisper  # Or from faster_whisper import WhisperModel for optimized

model = whisper.load_model("large-v3-turbo", device="cpu")  # Downloads ~3GB on first run
result = model.transcribe("output.wav", language="en")  # Or multilingual auto-detect
print(result["text"])