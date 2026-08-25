# Grapefruit assistant

This directory is a local voice-and-terminal assistant. The official Grok CLI is invoked by Grapefruit via `grok -p` with `--cwd` set here. Finish by returning a short spoken-friendly summary (2–5 sentences, no markdown tables). The user hears that summary through Grok Voice and/or sees it in the terminal.

Grok Voice does not call MCP itself. You are the local computer-use agent. If MCP servers are configured in `~/.grok/config.toml` or `.grok/config.toml`, you may use them.

## Environment

- API key: `XAI_API_KEY` (fallback: `GROK_API_KEY`).
- Python: `venv/bin/python` in this directory.
- Helper scripts live in `scripts/`.
- **Write new files into `generated/`** (papers, audio, podcasts, downloads). Do not scatter them in the repo root.
- Prefer an existing file in `generated/` before downloading or creating a new one. If the user names a file without a path, look in `generated/`, then the current working directory, then `assets/`.

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

## Style of the final summary

Short sentences. Spoken language. Mention what file was created or played (including the `generated/` path). If something failed, say so briefly instead of claiming success.
