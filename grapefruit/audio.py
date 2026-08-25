"""Local audio device helpers."""

from __future__ import annotations


def list_input_devices() -> list[tuple[int, str, int]]:
    import pyaudio

    pa = pyaudio.PyAudio()
    devices: list[tuple[int, str, int]] = []
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
