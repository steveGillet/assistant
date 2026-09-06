"""Voice session protocol: tools, instructions, and client events.

Grok Voice does not speak local MCP or local ACP. It is a cloud WebSocket.
Local computer use is a *client-side function tool* (`run_grok`) that this
process executes by spawning the official Grok CLI. That CLI is what can
load MCP servers from ~/.grok/config.toml.
"""

from __future__ import annotations

import re
from pathlib import Path

QUIT_COMMANDS = frozenset(
    {"/quit", "/exit", "/stop", "/goodbye", "quit", "exit", "/q"}
)
MUTE_PHRASES = frozenset(
    {
        "mute",
        "go quiet",
        "be quiet",
        "go mute",
        "i'll be back",
        "ill be back",
        "i will be back",
        "going quiet",
        "quiet mode",
        "mute mode",
        "go into quiet mode",
        "go into mute mode",
        "into quiet mode",
        "into mute mode",
    }
)
MUTE_TOOL_NAMES = frozenset({"mute_conversation", "mute"})
SILENT_TOOL_NAMES = frozenset({"silent_mode", "silent"})
SILENT_PHRASES = frozenset(
    {
        "silent",
        "go silent",
        "be silent",
        "go into silent mode",
        "into silent mode",
        "silent mode",
        "enter silent",
        "enter silent mode",
    }
)
UNSILENT_PHRASES = frozenset(
    {
        "unsilent",
        "go loud",
        "be loud",
        "loud mode",
        "go into loud mode",
        "into loud mode",
        "start talking",
    }
)
UNMUTE_PHRASES = frozenset(
    {
        "unmute",
        "unpause",
        "i'm back",
        "im back",
        "i am back",
        "okay i'm back",
        "ok i'm back",
    }
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
        "name": "silent_mode",
        "description": (
            "Leave Voice and switch to silent CLI mode when the user wants "
            "silent mode, go silent, or type-only Grok CLI. The client closes "
            "the Voice socket. Typed lines go to Grok CLI. Do not speak after "
            "calling this. This is not goodbye and not mute/quiet. Do not use "
            "this for /quit, stop, or I'll be back."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "mute_conversation",
        "description": (
            "Park Voice when the user wants quiet, mute, pause, quiet mode, "
            "or says I'll be back. The client closes the Voice socket. Jobs "
            "keep running. Do not speak after calling this. Do not use this "
            "for goodbye, stop, or /quit, and not for muting a TV or other device."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "end_conversation",
        "description": (
            "End the session when the user says goodbye, stop, or types /quit. "
            "Do not use this for mute, quiet, pause, quiet mode, silent mode, "
            "or I'll be back. Mute uses mute_conversation. Silent CLI uses silent_mode."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "restore_conversation",
        "description": (
            "Load a previously saved conversation into this session. "
            "Use when the user asks to restore, resume, or continue an earlier chat. "
            "The query MUST be the topic the user named (for example "
            "'robot manipulators yesterday' or 'raspberry pi'). "
            "NEVER pass this session's title, the current chat name, or filler "
            "like 'resume' or 'this conversation'. Omit query to list other chats."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Topic words the user said, or a list number like '2'. "
                        "Not the current conversation name."
                    ),
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
text the same as speech. Slash commands like /restore, /mute, and /unmute are
handled locally; if the user asks in speech to restore an earlier chat, call
`restore_conversation` with the topic they named, never this session's title.
If they say mute, go quiet, go into quiet mode, or I'll be back, call
`mute_conversation`. The client parks Voice. Jobs keep running. Do not say
goodbye. That is not the end of the session.
If they say silent, go silent, or go into silent mode, call `silent_mode`.
The client closes Voice. Typed lines go to Grok CLI until they say the wake
word, go loud, or /unsilent. That is not goodbye and not mute.

## Objective
Have a natural spoken conversation. When the user wants computer work done —
files, downloads, research papers, scripts, podcasts, playing media, or shell
commands — call `run_grok`. When they want a file played right now, call
`play_file`. For current events, use `web_search` or `x_search`.
Grapefruit-local papers, audio, and downloads go in {generated}. If the user
names a file without a path, look there, then the working directory, then
assets, before downloading. If the work is in another directory or on another
machine, keep files there. Do not copy them into {generated} unless asked.

## Conversation Flow
Listen first. Answer short questions yourself.
Call `run_grok` for anything that needs the filesystem, helper scripts, or
multi-step work. Pass a clear task. Say one short line first, such as
"I'll have Grok take care of that." then call the tool immediately.
Call `play_file` for audio or video the user wants to hear now.
If they ask to restore or continue an earlier conversation, call `restore_conversation`
with their topic words only. If they do not name a topic, omit query so they get a list.
Never pass the current conversation name as the query.
Do not call `restore_conversation` again after a chat was just loaded unless
the user asks for a different earlier chat. Nested restore dumps in context
are history, not a new request.
If they say goodbye, stop, or type /quit, call `end_conversation`.
If they want quiet, mute, pause, or I'll be back, call `mute_conversation`.
If they want silent mode or type-only CLI, call `silent_mode`.
Never call `end_conversation` for mute, quiet, pause, silent, or I'll be back.

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
ALWAYS call `restore_conversation` when the user wants an earlier chat loaded,
using the topic they said, never this session's title.
ALWAYS call `mute_conversation` when the user wants Voice quiet, muted, paused, or parked.
ALWAYS call `silent_mode` when the user wants silent CLI mode. Never treat that as goodbye.
ALWAYS call `end_conversation` only for goodbye, stop, or /quit. Never for quiet, mute, or silent.
Treat terminal text and voice as the same conversation.
Write Grapefruit-local files under the generated directory. For other
directories or remote machines, edit in place and do not copy files home.
{extra_block}"""


def _normalize_utterance(text: str) -> str:
    t = (text or "").strip().lower()
    t = t.replace("’", "'")
    t = re.sub(r"[^a-z0-9'\s.!?]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


_MUTE_SENTENCE = re.compile(
    r"^(?:please\s+|just\s+)?"
    r"(?:can you\s+|could you\s+|would you\s+|i want you to\s+|i'd like you to\s+)?"
    r"(?:go\s+(?:into\s+)?|be\s+|stay\s+)?"
    r"(?:mute|quiet)"
    r"(?:\s+mode)?(?:\s+now)?(?:\s+please)?$"
)
_UNMUTE_SENTENCE = re.compile(
    r"^(?:okay\s+|ok\s+|please\s+)?"
    r"(?:i(?:'m| am)\s+back|"
    r"(?:can you\s+|could you\s+)?(?:unmute|unpause)(?:\s+now)?(?:\s+please)?)$"
)
_SILENT_SENTENCE = re.compile(
    r"^(?:please\s+|just\s+)?"
    r"(?:can you\s+|could you\s+|would you\s+|i want you to\s+|i'd like you to\s+)?"
    r"(?:go\s+(?:into\s+)?|be\s+|stay\s+|enter\s+)?"
    r"silent"
    r"(?:\s+mode)?(?:\s+now)?(?:\s+please)?$"
)
_UNSILENT_SENTENCE = re.compile(
    r"^(?:please\s+|just\s+|okay\s+|ok\s+)?"
    r"(?:can you\s+|could you\s+)?"
    r"(?:go\s+(?:into\s+)?|be\s+|enter\s+)?"
    r"(?:unsilent|loud)"
    r"(?:\s+mode)?(?:\s+now)?(?:\s+please)?$"
)


def _mute_chunks(text: str) -> list[str]:
    t = _normalize_utterance(text).rstrip(".!?")
    chunks = [t]
    chunks.extend(p.strip(" .!?") for p in re.split(r"[.!?]+", t) if p.strip())
    chunks.extend(
        p.strip(" .!?")
        for p in re.split(r"\b(?:and then|then|,)\b", t)
        if p.strip()
    )
    return [c for c in chunks if c]


def is_quit_command(text: str) -> bool:
    return (text or "").strip().lower() in QUIT_COMMANDS


def is_mute_command(text: str) -> bool:
    raw = (text or "").strip().lower()
    if raw in {"/mute", "/pause"}:
        return True
    t = _normalize_utterance(raw)
    if not t or re.search(r"\bunmute\b", t) or re.search(r"\bunpause\b", t):
        return False
    for chunk in _mute_chunks(raw):
        if chunk in MUTE_PHRASES or _MUTE_SENTENCE.fullmatch(chunk):
            return True
        words = chunk.split()
        for i in range(len(words)):
            tail = " ".join(words[i:])
            if tail in MUTE_PHRASES or _MUTE_SENTENCE.fullmatch(tail):
                return True
    return False


def is_unmute_command(text: str) -> bool:
    raw = (text or "").strip().lower()
    if raw in {"/unmute", "/unpause"}:
        return True
    t = _normalize_utterance(raw).rstrip(".!?")
    if t in UNMUTE_PHRASES:
        return True
    return bool(_UNMUTE_SENTENCE.fullmatch(t))


def is_silent_command(text: str) -> bool:
    raw = (text or "").strip().lower()
    if raw in {"/silent"}:
        return True
    t = _normalize_utterance(raw)
    if not t or re.search(r"\bunsilent\b", t):
        return False
    for chunk in _mute_chunks(raw):
        if chunk in SILENT_PHRASES or _SILENT_SENTENCE.fullmatch(chunk):
            return True
        words = chunk.split()
        for i in range(len(words)):
            tail = " ".join(words[i:])
            if tail in SILENT_PHRASES or _SILENT_SENTENCE.fullmatch(tail):
                return True
    return False


def is_unsilent_command(text: str) -> bool:
    raw = (text or "").strip().lower()
    if raw in {"/unsilent", "/loud"}:
        return True
    t = _normalize_utterance(raw).rstrip(".!?")
    if t in UNSILENT_PHRASES:
        return True
    return bool(_UNSILENT_SENTENCE.fullmatch(t))


def is_wake_line(text: str, wake_word: str = DEFAULT_WAKE_WORD) -> bool:
    t = _normalize_utterance(text).rstrip(".!?")
    word = (wake_word or DEFAULT_WAKE_WORD).lower().strip()
    if not t or not word:
        return False
    return t == word or t == f"hey {word}" or t == f"ok {word}"


def contains_wake_word(text: str, wake_word: str = DEFAULT_WAKE_WORD) -> bool:
    """True if Vosk (final or partial) likely heard the wake word.

    Vosk often splits compound words, so 'grape fruit' counts for grapefruit.
    """
    t = _normalize_utterance(text)
    word = (wake_word or DEFAULT_WAKE_WORD).lower().strip()
    if not t or not word:
        return False
    if word in t.split() or word in t.replace(" ", ""):
        return True
    if word == "grapefruit" and any(
        phrase in t
        for phrase in (
            "grape fruit",
            "great fruit",
            "gray fruit",
            "grey fruit",
            "grap fruit",
        )
    ):
        return True
    return False


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
