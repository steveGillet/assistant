from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from grapefruit.grok_cli import run_grok
from grapefruit.memory import load_restore_text
from grapefruit.paths import GENERATED, ROOT, ack_wav
from grapefruit.search import web_search, x_search


def resolve_media_path(
    path: str,
    *,
    cwd: str | None = None,
    project: Path = ROOT,
) -> str | None:
    cwd = cwd or os.getcwd()
    expanded = os.path.expanduser(path)
    candidates = [
        expanded,
        os.path.join(cwd, expanded),
        str(project / expanded),
        os.path.join(cwd, os.path.basename(expanded)),
        str(project / os.path.basename(expanded)),
        str(Path(project) / "generated" / os.path.basename(expanded)),
        str(Path(project) / "assets" / os.path.basename(expanded)),
        str(GENERATED / os.path.basename(expanded)),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    return None


def play_file(path: str, *, cwd: str | None = None) -> str:
    cwd = cwd or os.getcwd()
    resolved = resolve_media_path(path, cwd=cwd)
    if not resolved:
        return f"File not found: {path}"

    player = None
    for cmd in (
        ["cvlc", "--play-and-exit", "--no-video-title-show", resolved],
        ["vlc", "--play-and-exit", resolved],
        ["ffplay", "-nodisp", "-autoexit", resolved],
        ["paplay", resolved],
        ["mpv", resolved],
    ):
        if shutil.which(cmd[0]):
            player = cmd
            break
    if not player:
        return "No media player found (install vlc, ffplay, paplay, or mpv)."

    subprocess.Popen(
        player, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=cwd
    )
    return f"Playing {resolved}"


def handle_tool(name: str, args: dict) -> tuple[str, bool]:
    if name == "run_grok":
        return run_grok(args.get("task", ""), bool(args.get("background", False))), False
    if name == "play_file":
        return play_file(args.get("path", "")), False
    if name in {"web_search", "websearch"}:
        return web_search(args.get("query") or args.get("q") or ""), False
    if name in {"x_search", "xsearch"}:
        return x_search(args.get("query") or args.get("q") or ""), False
    if name == "restore_conversation":
        body, meta = load_restore_text(args.get("query") or "")
        if meta is None:
            return body, False
        return (
            "Prior conversation loaded. Use it as context. "
            "Confirm the topic in one short sentence. Do not read the log aloud.\n\n"
            + body
        ), False
    if name == "end_conversation":
        return "Goodbye.", True
    return f"Unknown tool: {name}", False


def play_ack() -> None:
    ack = ack_wav()
    if ack is not None and shutil.which("paplay"):
        subprocess.run(["paplay", str(ack)], check=False)
        return
    try:
        import pyaudio
    except ImportError:
        return
    pa = pyaudio.PyAudio()
    stream = pa.open(rate=16000, channels=1, format=pyaudio.paInt16, output=True)
    samples = bytearray()
    for i in range(16000 // 5):
        value = int(8000 * (1 if (i // 18) % 2 == 0 else -1))
        samples += int.to_bytes(value & 0xFFFF, 2, "little", signed=False)
    stream.write(bytes(samples))
    stream.stop_stream()
    stream.close()
    pa.terminate()
