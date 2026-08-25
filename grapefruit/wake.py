from __future__ import annotations

import asyncio
import json
import queue
import time

import pyaudio
from vosk import KaldiRecognizer, Model

from grapefruit import ui
from grapefruit.commands import handle_line
from grapefruit.paths import vosk_model_dir
from grapefruit.protocol import WAKE_RATE
from grapefruit.session import run_session
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
) -> None:
    model_path = vosk_model_dir()
    if not model_path.is_dir():
        raise SystemExit(
            f"Vosk model missing at {model_path}. "
            "Unzip assets/vosk-model-small-en-us-0.15.zip into assets/."
        )

    ui.status("loading wake-word model")
    vosk_model = Model(str(model_path))
    recognizer = KaldiRecognizer(vosk_model, WAKE_RATE)
    typed = typed if typed is not None else queue.Queue()

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
    ui.status(f"listening for “{wake_word}” · type a line anytime · /quit")

    try:
        while True:
            try:
                typed_line = typed.get_nowait()
            except queue.Empty:
                typed_line = None
            if typed_line is not None:
                text = typed_line.strip()
                if not text:
                    continue
                action = handle_line(text)
                if action.kind == "quit":
                    ui.session("bye")
                    break
                if action.kind in {"print", "help", "save"}:
                    ui.info(action.text)
                    continue
                stream.stop_stream()
                try:
                    if action.kind == "restore":
                        ui.status(action.text)
                        asyncio.run(
                            run_session(
                                api_key,
                                voice,
                                enable_mic=enable_mic,
                                enable_speaker=enable_speaker,
                                typed=typed,
                                extra_instructions=action.restore_body,
                                restore_title=action.restore_title,
                                mic_device=mic_device,
                                mute_mic_while_speaking=mute_mic_while_speaking,
                                voice_barge_in=voice_barge_in,
                            )
                        )
                    else:
                        ui.user(action.text)
                        asyncio.run(
                            run_session(
                                api_key,
                                voice,
                                enable_mic=enable_mic,
                                enable_speaker=enable_speaker,
                                typed=typed,
                                initial_text=action.text,
                                mic_device=mic_device,
                                mute_mic_while_speaking=mute_mic_while_speaking,
                                voice_barge_in=voice_barge_in,
                            )
                        )
                except Exception as exc:
                    ui.error(str(exc))
                time.sleep(0.5)
                recognizer = KaldiRecognizer(vosk_model, WAKE_RATE)
                stream.start_stream()
                ui.status(f"listening for “{wake_word}”")
                continue

            data = stream.read(4096, exception_on_overflow=False)
            if not recognizer.AcceptWaveform(data):
                continue
            heard = json.loads(recognizer.Result()).get("text", "").lower()
            if wake_word not in heard:
                continue

            ui.status(f"wake “{wake_word}”")
            stream.stop_stream()
            play_ack()
            time.sleep(0.15)
            try:
                asyncio.run(
                    run_session(
                        api_key,
                        voice,
                        enable_mic=enable_mic,
                        enable_speaker=enable_speaker,
                        typed=typed,
                        mic_device=mic_device,
                        mute_mic_while_speaking=mute_mic_while_speaking,
                        voice_barge_in=voice_barge_in,
                    )
                )
            except Exception as exc:
                ui.error(str(exc))
            time.sleep(1.0)
            recognizer = KaldiRecognizer(vosk_model, WAKE_RATE)
            stream.start_stream()
            ui.status(f"listening for “{wake_word}”")
    except KeyboardInterrupt:
        ui.session("bye")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
