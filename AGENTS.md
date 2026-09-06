# Grapefruit assistant

This directory is a local voice-and-terminal assistant. The official Grok CLI is invoked by Grapefruit via `grok -p` with `--cwd` set here. Finish by returning a short spoken-friendly summary (2–5 sentences, no markdown tables). The user hears that summary through Grok Voice and/or sees it in the terminal.

Grok Voice does not call MCP itself. You are the local computer-use agent. If MCP servers are configured in `~/.grok/config.toml` or `.grok/config.toml`, you may use them.

## Environment

- API key: `XAI_API_KEY` (fallback: `GROK_API_KEY`).
- Python: `venv/bin/python` in this directory.
- Helper scripts live in `scripts/`.

## Files

`generated/` is Grapefruit's inbox for assistant-local artifacts: papers, audio, podcasts, and downloads made for this machine. Do not scatter those in the repo root or `assets/`.

When the user names a file without a path, look in this order before downloading or creating a duplicate: `generated/`, then the current working directory, then `assets/`.

Where to write new files:
- Grapefruit-local work (papers, podcasts, helper-script output, downloads for this assistant): `generated/`.
- A project in another directory: write in that project, using that project's layout. Do not copy those files into `generated/` unless the user asked for a local copy.
- Remote work over SSH: edit and create files on that machine, in that project directory. Do not copy or `scp` them back into `generated/` unless the user asked to bring a copy home.
- If the user names a path, use that path.

## Helper scripts

Run them with `venv/bin/python`. Defaults already point at `generated/`.

- **Long-form text:** `venv/bin/python scripts/generateScript.py --input 'topic or prompt'`
  - Writes `generated/paper.txt`.
- **Paper to audio:** `venv/bin/python scripts/extractAudio.py`
  - Reads `generated/paper.txt`, writes `generated/extracted_audio.wav`.
- **Podcast from PDF or TXT:** `venv/bin/python scripts/podcast.py --input file.pdf --output generated/name.mp3`
  - Prefer PDF when both exist.
- **Audio drama from a script:** `venv/bin/python scripts/audioDrama.py --input script.txt --output generated/drama.mp3`

Long jobs (podcast, paper-to-audio, long scripts) should be started and allowed to finish. Do not poll them in a tight loop.

## Papers

1. Download the PDF into `generated/` (`wget` from arXiv or another source).
2. Extract text with `pdftotext` into `generated/paper.txt` if you need a readable transcript.
3. Convert to audio with `venv/bin/python scripts/extractAudio.py`.
4. Rename `generated/extracted_audio.wav` to a descriptive name in `generated/` based on the paper title before playing.

## Playback

Play audio or video once with `vlc --play-and-exit` (or `cvlc --play-and-exit`). Run playback in the background so the CLI can return a summary.

Do **not** restart PipeWire, WirePlumber, or PulseAudio. Grapefruit keeps the microphone open; killing the audio stack can freeze or crash the desktop. Do not change the default sink or source to a Bluetooth device while Grapefruit is running. Diagnose Bluetooth, write a script if needed, and tell the user to run it after they quit Grapefruit.

## Style of the final summary

Short sentences. Spoken language. Mention what file was created or played, with its real path. If something failed, say so briefly instead of claiming success.
