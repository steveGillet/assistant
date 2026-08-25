import torch
from diffusers import AudioLDMPipeline
pipe = AudioLDMPipeline.from_pretrained('cvssp/audioldm-s-full-v2', torch_dtype=torch.float32)
pipe = pipe.to('cpu')
prompt = 'dog barking loudly'
audio = pipe(prompt, num_inference_steps=20, audio_length_in_s=3.0).audios[0]
import torchaudio
torchaudio.save('dog_bark_audioldm.wav', torch.tensor([audio]), 16000)
print('AudioLDM dog bark generated!')
