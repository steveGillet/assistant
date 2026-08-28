from __future__ import annotations

import argparse
import asyncio
import queue
import sys

from grapefruit import ui
from grapefruit.audio import print_input_devices
from grapefruit.env import get_xai_api_key
from grapefruit.grok_cli import find_grok_bin
from grapefruit.paths import ensure_dirs
from grapefruit.protocol import DEFAULT_VOICE, DEFAULT_WAKE_WORD
from grapefruit.session import run_session, start_stdin_thread
from grapefruit.wake import listen_for_wake_word


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Grapefruit assistant: Grok Voice plus official Grok CLI. "
            "Speak, type, or both."
        )
    )
    parser.add_argument("--wake-word", default=DEFAULT_WAKE_WORD)
    parser.add_argument(
        "--voice", default=DEFAULT_VOICE, help="Grok voice id, e.g. eve, ara, rex"
    )
    parser.add_argument(
        "--no-wake",
        action="store_true",
        help="Skip the wake word and open a session immediately",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="No microphone. Type in the terminal; replies still play as speech.",
    )
    parser.add_argument(
        "--no-speaker",
        action="store_true",
        help="Do not play assistant audio (transcripts still print)",
    )
    parser.add_argument(
        "--list-mics",
        action="store_true",
        help="List capture devices and exit",
    )
    parser.add_argument(
        "--mic-device",
        type=int,
        default=None,
        help="PyAudio input device index (see --list-mics). Prefer a headset over the webcam.",
    )
    parser.add_argument(
        "--barge-in",
        action="store_true",
        help="Allow voice to interrupt assistant playback (off: mic is muted while speaking)",
    )
    parser.add_argument(
        "--idle-sec",
        type=int,
        default=None,
        help=(
            "Close the Voice session after this many seconds with no user input "
            "(default 600, or GROK_VOICE_IDLE_SEC). 0 disables."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    ensure_dirs()
    if args.list_mics:
        print_input_devices()
        return
    api_key = get_xai_api_key()
    grok_bin = find_grok_bin()
    ui.banner(grok_bin)
    if not grok_bin:
        ui.error("official grok CLI not found · curl -fsSL https://x.ai/cli/install.sh | bash")

    typed: queue.Queue[str] = queue.Queue()
    if sys.stdin.isatty() or args.text_only:
        start_stdin_thread(typed)
        ui.status("type a line · /restore · /conversations · /help · /quit")

    voice = args.voice.lower()
    enable_mic = not args.text_only
    enable_speaker = not args.no_speaker
    if enable_mic:
        print_input_devices()
        if args.mic_device is not None:
            ui.status(f"mic device {args.mic_device}")

    session_kwargs = dict(
        enable_mic=enable_mic,
        enable_speaker=enable_speaker,
        typed=typed,
        mic_device=args.mic_device,
        mute_mic_while_speaking=not args.barge_in,
        voice_barge_in=args.barge_in,
        idle_sec=args.idle_sec,
    )

    if args.text_only or args.no_wake:
        asyncio.run(
            run_session(
                api_key,
                voice,
                park_in_process=True,
                **session_kwargs,
            )
        )
        return

    listen_for_wake_word(
        args.wake_word.lower(),
        api_key,
        voice,
        **session_kwargs,
    )
