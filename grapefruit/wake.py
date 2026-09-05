from __future__ import annotations

import asyncio
import json
import queue
import threading
import time

import pyaudio
from vosk import KaldiRecognizer, Model

from grapefruit import ui
from grapefruit.audio import hush_alsa
from grapefruit.commands import handle_line
from grapefruit.hold import HoldState
from grapefruit.paths import vosk_model_dir
from grapefruit.memory import ConversationLog
from grapefruit.protocol import WAKE_RATE, is_unmute_command, is_unsilent_command
from grapefruit.session import run_session
from grapefruit.silent import SilentState, handle_silent_line
from grapefruit.tools import play_ack


def listen_for_wake_word(
    wake_word: str,
    api_key: str,
    voice: str,
    *,
    typed: queue.Queue[str] | None = None,
    enable_mic: bool = True,
    enable_speaker: bool = True,
    mic_device: int | None = None,
    mute_mic_while_speaking: bool = True,
    voice_barge_in: bool = False,
    idle_sec: int | None = None,
    start_silent: bool = False,
) -> None:
    model_path = vosk_model_dir()
    if not model_path.is_dir():
        raise SystemExit(
            f"Vosk model missing at {model_path}. "
            "Unzip assets/vosk-model-small-en-us-0.15.zip into assets/."
        )

    ui.status("loading wake-word model")
    with hush_alsa():
        vosk_model = Model(str(model_path))
    recognizer = KaldiRecognizer(vosk_model, WAKE_RATE)
    typed = typed if typed is not None else queue.Queue()
    chunks: queue.Queue[bytes] = queue.Queue(maxsize=12)
    stop_mic = threading.Event()

    with hush_alsa():
        pa = pyaudio.PyAudio()
        open_kwargs = dict(
            rate=WAKE_RATE,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=4096,
        )
        if mic_device is not None:
            open_kwargs["input_device_index"] = mic_device
        stream = pa.open(**open_kwargs)
        stream.start_stream()

    def mic_reader() -> None:
        while not stop_mic.is_set():
            if not stream.is_active():
                time.sleep(0.05)
                continue
            try:
                data = stream.read(4096, exception_on_overflow=False)
            except Exception:
                time.sleep(0.05)
                continue
            try:
                chunks.put_nowait(data)
            except queue.Full:
                try:
                    chunks.get_nowait()
                except queue.Empty:
                    pass
                try:
                    chunks.put_nowait(data)
                except queue.Full:
                    pass

    threading.Thread(target=mic_reader, daemon=True, name="wake-mic").start()
    hold = HoldState()
    parked_log = None
    silent_state: SilentState | None = None

    def enter_silent(log=None) -> None:
        nonlocal parked_log, silent_state
        parked_log = log or parked_log or hold.log
        if parked_log is None:
            parked_log = ConversationLog()
            parked_log.start("")
        hold.log = parked_log
        hold.outcome = "silent"
        silent_state = SilentState(log=parked_log)
        ui.status(
            f"silent · grok cli · say “{wake_word}” or /unsilent for voice · /quit ends"
        )

    if start_silent:
        enter_silent()
    else:
        ui.status(f"listening for “{wake_word}” · type a line anytime · /quit")
    session_kwargs = dict(
        enable_mic=enable_mic,
        enable_speaker=enable_speaker,
        typed=typed,
        mic_device=mic_device,
        mute_mic_while_speaking=mute_mic_while_speaking,
        voice_barge_in=voice_barge_in,
        idle_sec=idle_sec,
        hold=hold,
        park_in_process=False,
    )

    def launch_session(**extra):
        nonlocal parked_log, silent_state
        stream.stop_stream()
        try:
            kwargs = dict(session_kwargs)
            kwargs.update(extra)
            if parked_log is not None:
                kwargs["log"] = parked_log
            outcome = asyncio.run(run_session(api_key, voice, **kwargs))
            if outcome == "silent":
                enter_silent(hold.log)
            elif outcome == "mute":
                parked_log = hold.log
                hold.outcome = "mute"
                if hold.busy():
                    ui.status("muted · job still running · chime when done")
                else:
                    ui.status(
                        f"muted · say “{wake_word}” or /unmute when you want the recap"
                    )
            else:
                parked_log = None
                silent_state = None
                hold.outcome = "run"
        except Exception as exc:
            ui.error(str(exc))
            parked_log = None
            silent_state = None
        time.sleep(0.5)
        recognizer_reset = KaldiRecognizer(vosk_model, WAKE_RATE)
        stream.start_stream()
        if hold.outcome == "silent":
            ui.status(
                f"silent · grok cli · say “{wake_word}” or /unsilent for voice · /quit ends"
            )
        else:
            ui.status(f"listening for “{wake_word}”")
        return recognizer_reset

    try:
        while True:
            if hold.outcome == "mute":
                finished = hold.consume_finished(parked_log, chime=True)
                if finished:
                    ui.status(
                        f"job done · say “{wake_word}” or /unmute for the recap"
                    )
            try:
                typed_line = typed.get(timeout=0.05)
            except queue.Empty:
                typed_line = None
            if typed_line is not None:
                text = typed_line.strip()
                if not text:
                    continue
                if hold.outcome == "silent":
                    if silent_state is None:
                        enter_silent()
                    parked = handle_silent_line(text, silent_state)
                    if parked == "quit":
                        break
                    if parked == "unsilent":
                        extra = ""
                        opening = ""
                        if parked_log is not None:
                            extra = parked_log.context_for_model()
                            opening = hold.take_spoken_recap(
                                parked_log.meta.title if parked_log.meta else ""
                            )
                        recognizer = launch_session(
                            extra_instructions=extra,
                            opening_line=opening,
                        )
                    continue
                muted = hold.outcome == "mute"
                action = handle_line(text, parked_log, muted=muted)
                if action.kind == "quit":
                    ui.session("bye")
                    break
                if action.kind == "silent":
                    enter_silent()
                    continue
                if action.kind == "mute":
                    ui.status("no live voice to mute")
                    continue
                if action.kind == "status":
                    ui.info(hold.status_text())
                    continue
                if action.kind in {"print", "help", "save"}:
                    ui.info(action.text)
                    continue
                if action.kind == "unmute" or (muted and action.kind == "passthrough"):
                    opening = ""
                    extra = ""
                    if parked_log is not None:
                        extra = parked_log.context_for_model()
                        opening = hold.take_spoken_recap(
                            parked_log.meta.title if parked_log.meta else ""
                        )
                    init = action.text if action.kind == "passthrough" else None
                    recognizer = launch_session(
                        extra_instructions=extra,
                        opening_line=opening,
                        initial_text=init,
                    )
                    continue
                if action.kind == "restore":
                    ui.status(action.text)
                    parked_log = None
                    recognizer = launch_session(
                        extra_instructions=action.restore_body,
                        restore_title=action.restore_title,
                    )
                    continue
                ui.user(action.text)
                parked_log = None
                recognizer = launch_session(initial_text=action.text)
                continue

            try:
                data = chunks.get_nowait()
            except queue.Empty:
                continue
            if not recognizer.AcceptWaveform(data):
                continue
            heard = json.loads(recognizer.Result()).get("text", "").lower()
            parked = hold.outcome in {"mute", "silent"}
            if hold.outcome == "mute" and hold.busy():
                continue
            come_back = parked and (
                wake_word in heard
                or is_unmute_command(heard)
                or is_unsilent_command(heard)
            )
            if not come_back and wake_word not in heard:
                continue

            ui.status(f"wake “{wake_word}”")
            play_ack()
            time.sleep(0.15)
            extra = ""
            opening = ""
            if parked_log is not None:
                extra = parked_log.context_for_model()
                opening = hold.take_spoken_recap(
                    parked_log.meta.title if parked_log.meta else ""
                )
            recognizer = launch_session(
                extra_instructions=extra,
                opening_line=opening,
            )
    except KeyboardInterrupt:
        ui.session("bye")
    finally:
        stop_mic.set()
        try:
            stream.stop_stream()
            stream.close()
        except Exception:
            pass
        pa.terminate()
