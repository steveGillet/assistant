"""Voice session protocol: tools, instructions, and client events.

Grok Voice does not speak local MCP or local ACP. It is a cloud WebSocket.
Local computer use is a *client-side function tool* (`run_grok`) that this
process executes by spawning the official Grok CLI. That CLI is what can
load MCP servers from ~/.grok/config.toml.
"""

from __future__ import annotations

from pathlib import Path

QUIT_COMMANDS = frozenset(
    {"/quit", "/exit", "/stop", "/goodbye", "quit", "exit", "/q"}
)

VOICE_MODEL = "grok-voice-latest"
VOICE_URI = f"wss://api.x.ai/v1/realtime?model={VOICE_MODEL}"
SAMPLE_RATE = 24000
MIC_FRAMES = 2400
WAKE_RATE = 16000
DEFAULT_WAKE_WORD = "grapefruit"
DEFAULT_VOICE = "eve"

TOOLS = [
    {"type": "web_search"},
    {"type": "x_search"},
    {
        "type": "function",
        "name": "run_grok",
        "description": (
            "Ask the official Grok CLI (Grok Build) to do computer work in this "
            "directory: files, downloads, papers, scripts, podcasts, commands. "
            "Returns a short summary when finished. Use background=true for long "
            "jobs such as generating a podcast or converting a paper to audio."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Clear instruction for Grok CLI.",
                },
                "background": {
                    "type": "boolean",
                    "description": "Start the job and return immediately.",
                },
            },
            "required": ["task"],
        },
    },
    {
        "type": "function",
        "name": "play_file",
        "description": "Play a local audio or video file in the background.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path. Relative paths are tried in the cwd, then the project directory.",
                }
            },
            "required": ["path"],
        },
    },
    {
        "type": "function",
        "name": "end_conversation",
        "description": "End the session when the user says stop, goodbye, or types /quit.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "restore_conversation",
        "description": (
            "Load a previously saved conversation into this session. "
            "Use when the user asks to restore, resume, or continue an earlier "
            "chat (for example 'the robot manipulators conversation yesterday'). "
            "Pass a short search query; omit query to list recent conversations."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Title, topic, or date hint such as 'robot manipulators yesterday'.",
                }
            },
        },
    },
]


def voice_instructions(*, cwd: str, project: str, extra: str = "") -> str:
    scripts = str(Path(project) / "scripts")
    generated = str(Path(project) / "generated")
    extra_block = f"\n## Restored conversation\n{extra}\n" if extra.strip() else ""
    return f"""## Role & Persona
You are Grapefruit, a local assistant running on the user's Linux computer.
Working directory: {cwd}
Project directory: {project}
Generated files: {generated}
Helper scripts: {scripts}

The user may speak into the microphone or type in the terminal. Treat typed
text the same as speech. Slash commands like /restore are handled locally;
if the user asks in speech to restore an earlier chat, call `restore_conversation`.

## Objective
Have a natural spoken conversation. When the user wants computer work done —
files, downloads, research papers, scripts, podcasts, playing media, or shell
commands — call `run_grok`. When they want a file played right now, call
`play_file`. For current events, use `web_search` or `x_search`.
Put new files in {generated} unless the user names another path.

## Conversation Flow
Listen first. Answer short questions yourself.
Call `run_grok` for anything that needs the filesystem, helper scripts, or
multi-step work. Pass a clear task. Say one short line first, such as
"I'll have Grok take care of that." then call the tool immediately.
Call `play_file` for audio or video the user wants to hear now.
If they ask to restore or continue an earlier conversation, call `restore_conversation`.
If they say stop, goodbye, or type /quit, call `end_conversation`.

Helper scripts (via AGENTS.md / `run_grok`):
- scripts/extractAudio.py — generated/paper.txt to generated/extracted_audio.wav
- scripts/generateScript.py — long-form text to generated/paper.txt
- scripts/podcast.py — PDF/TXT to generated/name.mp3

## Guardrails & Escalation
Do not invent file contents. If you need to know what is on disk, call `run_grok`.
Do not claim a command succeeded unless a tool result says so.
If a tool fails, say so briefly and offer a next step.
If input is garbled, ask a short clarification instead of guessing.

## Voice & Communication Style
Speak naturally in short sentences. No markdown, no tables, no bullet lists,
no emojis, no stage directions. One to three sentences per turn unless asked
for more. Respond in English unless the user speaks another language.
Vary phrasing. Do not repeat the same sentence twice.

## CRITICAL INSTRUCTIONS
ALWAYS call `run_grok` for computer work. NEVER pretend you ran a command.
ALWAYS call `play_file` to play local audio or video. Do not describe playing it.
ALWAYS call `restore_conversation` when the user wants an earlier chat loaded.
ALWAYS call `end_conversation` when the user wants to stop.
Treat terminal text and voice as the same conversation.
Write new files under the generated directory.
{extra_block}"""


def is_quit_command(text: str) -> bool:
    return (text or "").strip().lower() in QUIT_COMMANDS


def user_text_event(text: str) -> dict:
    return {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": text}],
        },
    }


def function_output_event(call_id: str | None, result: str) -> dict:
    return {
        "type": "conversation.item.create",
        "item": {
            "type": "function_call_output",
            "call_id": call_id,
            "output": json_dumps_result(result),
        },
    }


def json_dumps_result(result: str) -> str:
    import json

    return json.dumps({"result": result})


def force_message_event(text: str) -> dict:
    return {
        "type": "conversation.item.create",
        "item": {
            "type": "force_message",
            "role": "assistant",
            "interruptible": True,
            "content": [{"type": "output_text", "text": text}],
        },
    }


def session_update_event(
    *, voice: str, cwd: str, project: str, extra_instructions: str = ""
) -> dict:
    return {
        "type": "session.update",
        "session": {
            "instructions": voice_instructions(
                cwd=cwd, project=project, extra=extra_instructions
            ),
            "voice": voice,
            "turn_detection": {
                "type": "server_vad",
                "silence_duration_ms": 700,
                "prefix_padding_ms": 300,
            },
            "audio": {
                "input": {"format": {"type": "audio/pcm", "rate": SAMPLE_RATE}},
                "output": {"format": {"type": "audio/pcm", "rate": SAMPLE_RATE}},
            },
            "tools": TOOLS,
        },
    }


def input_audio_append_event(b64_pcm: str) -> dict:
    return {"type": "input_audio_buffer.append", "audio": b64_pcm}
