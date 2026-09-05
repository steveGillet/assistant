"""Local audio device helpers."""

from __future__ import annotations

import os
from contextlib import contextmanager
from collections.abc import Iterator


@contextmanager
def hush_alsa() -> Iterator[None]:
    """Hide ALSA plugin probe spam on stderr while opening PortAudio."""
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        saved = os.dup(2)
        os.dup2(devnull, 2)
    except OSError:
        yield
        return
    try:
        yield
    finally:
        try:
            os.dup2(saved, 2)
            os.close(saved)
            os.close(devnull)
        except OSError:
            pass


def list_input_devices() -> list[tuple[int, str, int]]:
    import pyaudio

    devices: list[tuple[int, str, int]] = []
    with hush_alsa():
        pa = pyaudio.PyAudio()
        try:
            for index in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(index)
                channels = int(info.get("maxInputChannels") or 0)
                if channels <= 0:
                    continue
                name = str(info.get("name") or f"device {index}")
                devices.append((index, name, channels))
        finally:
            pa.terminate()
    return devices


def print_input_devices() -> None:
    from grapefruit import ui

    devices = list_input_devices()
    if not devices:
        ui.status("no input devices found")
        return
    ui.status("input devices")
    for index, name, channels in devices:
        ui.info(f"{index}: {name} ({channels} ch)")
    ui.info("use --mic-device INDEX  (skip the webcam if speakers couple into it)")
