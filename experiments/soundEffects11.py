import os
from elevenlabs.client import ElevenLabs
from elevenlabs import save

def generate_sfx(prompt: str, duration_seconds: int = 5, output_file: str = "sfx_output.mp3"):
    api_key = os.environ.get("ELEVENLABS_KEY")
    if not api_key:
        raise ValueError("ELEVENLABS_KEY environment variable is required")

    client = ElevenLabs(api_key=api_key)
    
    try:
        audio = client.text_to_sound_effects.convert(
            text=prompt,
            duration_seconds=duration_seconds,
            prompt_influence=0.5  # Adjust 0-1: higher follows prompt more strictly
        )
        save(audio, output_file)
        print(f"Generated SFX saved to {output_file}")
    except Exception as e:
        print(f"Error generating SFX: {e}")

# Test with your script's SFX example
if __name__ == "__main__":
    test_prompt = "Ominous pulsing orchestral theme with low drums like heartbeats, swelling synthetic winds, distorted guitar rifts, gusting winds through crumbled skyscrapers, rustling debris, and faint guttural zombie moans."
    generate_sfx(test_prompt, duration_seconds=10, output_file="zombie_opening_sfx.mp3")