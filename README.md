# Grapefruit — Grok voice + terminal assistant

Local assistant on the **official Grok CLI** and **Grok Voice API**. You can speak, type in the terminal, or both on the same session.

## How computer use actually works

Grok Voice is a **cloud WebSocket** (`wss://api.x.ai/v1/realtime`). It does **not** attach to a local MCP server, and it does not speak ACP.

What the Voice model can do:

| Tool | Where it runs |
|------|----------------|
| `web_search`, `x_search` | xAI's servers |
| Remote MCP (`type: mcp`, `server_url: https://...`) | An HTTP/SSE MCP server you host. Local stdio MCP is not valid here. |
| Custom functions (`run_grok`, `play_file`, `end_conversation`) | **This process.** The model emits a function call; Python executes it and sends the result back. |

Local commands go through `run_grok` → official `grok -p --yolo` (Grok Build). That CLI *is* the local agent: shell, files, web, and **any MCP servers in `~/.grok/config.toml` or `.grok/config.toml`**. Voice never talks to those MCP tools itself; Grok CLI does, then returns a short summary that Voice speaks.

```
you (mic or typed line)
        │
        ▼
 Grok Voice (cloud)  ── web_search / x_search ──► xAI
        │
        │  function: run_grok
        ▼
 grokVoice.py  ── subprocess ──►  grok -p   ──► shell / files / MCP
        │
        ▼
 speaker + terminal transcript
```

The long-lived equivalent of a “local MCP server” in official Grok is **`grok agent serve`** (ACP), not MCP. Today we spawn `grok -p` per task so there is nothing extra to start. MCP still applies *inside* that CLI process if you configure it.

## Dual input

There is no local LLM “router.” `grokVoice.py` is a **client**: it reads the mic and the keyboard, and it writes two different WebSocket frames to the same Grok Voice session.

- Mic bytes → `input_audio_buffer.append`
- Typed line → `conversation.item.create` with `input_text` (not TTS’d, not a separate Grok call)

Grok Voice is what interprets the request and chooses tools. Typed and spoken turns share that session’s context.

```bash
source venv/bin/activate
python grokVoice.py                 # wake word OR type to start; then both
python grokVoice.py --no-wake       # session now; speak and/or type
python grokVoice.py --text-only     # no mic; type, hear replies
python grokVoice.py --silent        # start in CLI; say grapefruit or /unsilent for Voice
python grokVoice.py --list-mics     # see capture devices
python grokVoice.py --mic-device N  # skip the webcam if speakers couple into it
```

While Grapefruit is talking, the mic is muted (plus a short hangover) so the speakers cannot barge in. Typed lines can still interrupt. `--barge-in` restores voice interruption.

Slash commands are handled **locally** (they never go to Voice as a user turn):

| Command | Effect |
|---------|--------|
| `/restore [query]` | Load a saved conversation into Voice context |
| `/conversations` | List saved chats |
| `/save [title]` | Name the current chat |
| `/silent` | Park Voice; typed lines go to Grok CLI |
| `/unsilent` | Back to Voice (`/loud` is the same) |
| `/quit` | End Grapefruit |

In speech: “restore the conversation we had about robot manipulators yesterday” makes Voice call `restore_conversation`.

Logs live in `conversations/` (JSONL). Older turns are compacted when the estimated size crosses 70% of a 20k-token budget (`GROK_VOICE_CONTEXT_TOKENS`).

## Generated files

Grapefruit-local papers, audio, and downloads go in **`generated/`**, not `assets/` and not the repo root. Bare filenames are looked up in `generated/`, then cwd, then `assets/`. Work on another project or over SSH stays there; do not copy those files into `generated/` unless asked.

## Layout

```
AGENTS.md                 # official Grok CLI project rules
grokVoice.py              # launcher  (also: python -m grapefruit)
grapefruit/               # package: voice session, CLI wrapper, TTS
scripts/                  # generateScript, extractAudio, podcast, audioDrama
tests/                    # unit tests (default)
tests/live/               # optional API/CLI tests
assets/                   # vosk model, leftover sample media (not outputs)
generated/                # papers, audio, podcasts, downloads
conversations/            # saved Voice/terminal transcripts
experiments/              # old local-TTS / CUDA / SFX experiments
```

## Setup

1. Official Grok CLI: `curl -fsSL https://x.ai/cli/install.sh | bash`
2. `export XAI_API_KEY="xai-..."` (or `grok login`; `GROK_API_KEY` still works)
3. System packages: `vlc`/`cvlc`/`ffplay`/`paplay`, `ffmpeg`, `pdftotext`, `wget`
4. `python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`
5. Vosk model under `assets/vosk-model-small-en-us-0.15/` (unzip the zip there if needed)

## Tests

Default suite is offline and fast:

```bash
source venv/bin/activate
pytest
```

Hits xAI / Grok CLI (small spend):

```bash
pytest -m live
```

Re-run `pytest` after changing protocol, tools, or the CLI wrapper.

## Helper scripts

```bash
venv/bin/python scripts/generateScript.py --input 'topic'
venv/bin/python scripts/extractAudio.py
venv/bin/python scripts/podcast.py --input file.pdf --output out.mp3
venv/bin/python scripts/audioDrama.py --input script.txt
```
