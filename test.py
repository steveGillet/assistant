from piper.voice import PiperVoice
MODEL_PATH = "/home/steve0gillet/piperVoices/en_US-lessac-medium.onnx"
voice = PiperVoice.load(MODEL_PATH)
synthesize = voice.synthesize("test text")
print(dir(next(synthesize)))