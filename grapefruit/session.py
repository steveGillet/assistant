"""One Grok Voice WebSocket session: mic and/or terminal stdin."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import queue
import sys
import threading
import time

import websockets

from grapefruit.commands import handle_line
from grapefruit.memory import ConversationLog, _short_title
from grapefruit.paths import ROOT, ensure_dirs
from grapefruit.protocol import (
    MIC_FRAMES,
    SAMPLE_RATE,
    VOICE_URI,
    force_message_event,
    function_output_event,
    input_audio_append_event,
    session_update_event,
    user_text_event,
)
from grapefruit import ui
from grapefruit.tools import handle_tool


class Playback:
    """Stream PCM16 to the speakers as deltas arrive."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, hangover_ms: int = 450):
        self.queue: queue.Queue[bytes | None] = queue.Queue()
        self._done = threading.Event()
        self._thread: threading.Thread | None = None
        self._sample_rate = sample_rate
        self._active = False
        self._hangover_s = hangover_ms / 1000.0
        self._unmute_at = 0.0

    def start(self) -> None:
        self._active = True
        self._done.set()
        self._unmute_at = 0.0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def speaking(self) -> bool:
        """True while assistant audio is queued/playing, plus a short hangover."""
        if not self._active:
            return False
        if not self._done.is_set():
            return True
        return time.monotonic() < self._unmute_at

    def _run(self) -> None:
        import pyaudio

        pa = pyaudio.PyAudio()
        stream = pa.open(
            rate=self._sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            output=True,
            frames_per_buffer=self._sample_rate // 10,
        )
        try:
            while self._active:
                chunk = self.queue.get()
                if chunk is None:
                    self._done.set()
                    self._unmute_at = time.monotonic() + self._hangover_s
                    continue
                self._done.clear()
                stream.write(chunk)
                if self.queue.empty():
                    self._done.set()
                    self._unmute_at = time.monotonic() + self._hangover_s
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()

    def push(self, pcm: bytes) -> None:
        if pcm:
            self._done.clear()
            self._unmute_at = time.monotonic() + 3600.0
            self.queue.put(pcm)

    def interrupt(self) -> None:
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        self._done.set()
        self._unmute_at = time.monotonic() + self._hangover_s

    def wait(self, timeout: float = 30.0) -> None:
        self._done.wait(timeout)

    def stop(self) -> None:
        self._active = False
        self.queue.put(None)


def start_stdin_thread(typed: queue.Queue[str]) -> threading.Thread:
    def reader() -> None:
        try:
            for line in sys.stdin:
                typed.put(line.rstrip("\n"))
        except (EOFError, OSError):
            return

    thread = threading.Thread(target=reader, daemon=True, name="stdin-reader")
    thread.start()
    return thread


async def run_session(
    api_key: str,
    voice: str,
    *,
    enable_mic: bool = True,
    enable_speaker: bool = True,
    typed: queue.Queue[str] | None = None,
    initial_text: str | None = None,
    extra_instructions: str = "",
    restore_title: str = "",
    cwd: str | None = None,
    project: str | None = None,
    mic_device: int | None = None,
    mute_mic_while_speaking: bool = True,
    voice_barge_in: bool = False,
) -> None:
    cwd = cwd or os.getcwd()
    project = project or str(ROOT)
    ensure_dirs()
    headers = {"Authorization": f"Bearer {api_key}"}
    playback = Playback() if enable_speaker else None
    mic_queue: queue.Queue[str] = queue.Queue()
    typed = typed if typed is not None else queue.Queue()
    session_active = True
    end_requested = False
    pending_calls: list[dict] = []
    session_ready = asyncio.Event()
    extra_instructions = extra_instructions or ""
    log = ConversationLog()
    log.start(initial_text or restore_title or "")
    assistant_bits: list[str] = []
    send_lock = asyncio.Lock()
    tool_tasks: set[asyncio.Task] = set()

    async with websockets.connect(
        VOICE_URI,
        additional_headers=headers,
        max_size=None,
        ping_interval=20,
        ping_timeout=120,
    ) as ws:

        async def safe_send(payload: dict) -> None:
            async with send_lock:
                await ws.send(json.dumps(payload))

        async def push_session_update() -> None:
            await safe_send(
                session_update_event(
                    voice=voice,
                    cwd=cwd,
                    project=project,
                    extra_instructions=extra_instructions,
                )
            )

        await push_session_update()

        def mic_streamer() -> None:
            import pyaudio

            pa = pyaudio.PyAudio()
            kwargs = dict(
                rate=SAMPLE_RATE,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=MIC_FRAMES,
            )
            if mic_device is not None:
                kwargs["input_device_index"] = mic_device
            stream = pa.open(**kwargs)
            ui.status("mic live · muted while grok speaks")
            try:
                while session_active:
                    data = stream.read(MIC_FRAMES, exception_on_overflow=False)
                    mic_queue.put(base64.b64encode(data).decode("ascii"))
            except Exception as exc:
                ui.error(f"mic {exc}")
            finally:
                stream.stop_stream()
                stream.close()
                pa.terminate()

        if playback:
            playback.start()
        mic_thread = None
        if enable_mic:
            mic_thread = threading.Thread(target=mic_streamer, daemon=True)
            mic_thread.start()
        else:
            ui.status("type a message · /help for commands")

        async def run_tools(calls: list[dict]) -> None:
            nonlocal extra_instructions, end_requested, session_active
            ui.status("running tools")
            try:
                for call in calls:
                    name = call.get("name") or ""
                    call_id = call.get("call_id")
                    try:
                        args = json.loads(call.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    ui.tool(name, args)
                    result, should_end = await asyncio.to_thread(
                        handle_tool, name, args
                    )
                    ui.tool_result(result)
                    log.append("tool", result[:1500], source="tool", name=name)
                    if name == "restore_conversation" and result.startswith(
                        "Prior conversation loaded"
                    ):
                        extra_instructions = result
                        await push_session_update()
                    if should_end:
                        end_requested = True
                    await safe_send(function_output_event(call_id, result))
                if playback:
                    await asyncio.to_thread(playback.wait, 20)
                if end_requested:
                    session_active = False
                    try:
                        await ws.close()
                    except Exception:
                        pass
                    return
                await safe_send({"type": "response.create"})
            except Exception as exc:
                ui.error(f"tool {exc}")
                try:
                    await safe_send(
                        {
                            "type": "conversation.item.create",
                            "item": {
                                "type": "message",
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_text",
                                        "text": f"(tool failed: {exc})",
                                    }
                                ],
                            },
                        }
                    )
                    await safe_send({"type": "response.create"})
                except Exception:
                    pass

        async def send_mic() -> None:
            while session_active:
                if mute_mic_while_speaking and playback and playback.speaking():
                    while True:
                        try:
                            mic_queue.get_nowait()
                        except queue.Empty:
                            break
                    await asyncio.sleep(0.02)
                    continue
                sent = False
                while True:
                    try:
                        chunk = mic_queue.get_nowait()
                    except queue.Empty:
                        break
                    await safe_send(input_audio_append_event(chunk))
                    sent = True
                if not sent:
                    await asyncio.sleep(0.02)

        async def apply_restore(body: str, title: str, notice: str) -> None:
            nonlocal extra_instructions
            extra_instructions = body
            log.append("system", f"Restored: {title}", source="restore", kind="restore")
            await push_session_update()
            ui.status(notice)
            if playback:
                playback.interrupt()
            spoken = _short_title(title, width=80)
            await safe_send(
                force_message_event(f"Restored the conversation about {spoken}.")
            )

        async def send_typed() -> None:
            nonlocal session_active, end_requested
            await session_ready.wait()
            if extra_instructions and not initial_text:
                title = restore_title or (log.meta.title if log.meta else "earlier chat")
                await apply_restore(
                    extra_instructions,
                    title,
                    f"Loaded “{_short_title(title)}” into Voice context.",
                )
            if initial_text:
                ui.user(initial_text)
                log.append("user", initial_text, source="typed")
                await safe_send(user_text_event(initial_text))
                await safe_send({"type": "response.create"})
            while session_active:
                try:
                    text = typed.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.05)
                    continue
                text = (text or "").strip()
                if not text:
                    continue
                action = handle_line(text, log)
                if action.kind == "quit":
                    ui.user("/quit")
                    end_requested = True
                    session_active = False
                    try:
                        await ws.close()
                    except Exception:
                        pass
                    break
                if action.kind in {"print", "help", "save"}:
                    ui.info(action.text)
                    continue
                if action.kind == "restore":
                    await apply_restore(
                        action.restore_body, action.restore_title, action.text
                    )
                    continue
                ui.user(action.text)
                log.append("user", action.text, source="typed")
                if playback:
                    playback.interrupt()
                await safe_send(user_text_event(action.text))
                await safe_send({"type": "response.create"})

        async def recv_loop() -> None:
            nonlocal session_active, end_requested, extra_instructions
            async for message in ws:
                if not session_active:
                    break
                if isinstance(message, bytes):
                    if playback:
                        playback.push(message)
                    continue
                event = json.loads(message)
                etype = event.get("type")

                if etype in ("session.updated", "session.created"):
                    session_ready.set()

                elif etype in ("response.output_audio.delta", "response.audio.delta"):
                    delta = event.get("delta") or ""
                    if delta and playback:
                        playback.push(base64.b64decode(delta))

                elif etype in (
                    "input_audio_buffer.speech_started",
                    "input_audio_buffer.speech_started.delta",
                ):
                    if playback and (voice_barge_in or not playback.speaking()):
                        playback.interrupt()

                elif etype == "response.function_call_arguments.done":
                    pending_calls.append(event)

                elif etype == "response.done":
                    if pending_calls:
                        calls = list(pending_calls)
                        pending_calls.clear()
                        task = asyncio.create_task(run_tools(calls))
                        tool_tasks.add(task)
                        task.add_done_callback(tool_tasks.discard)
                    elif end_requested:
                        session_active = False
                        break

                elif etype == "error":
                    ui.error(str(event))
                    session_ready.set()

                elif etype == "response.output_audio_transcript.delta":
                    piece = event.get("delta") or ""
                    assistant_bits.append(piece)
                    ui.assistant_delta(piece)
                elif etype == "response.output_audio_transcript.done":
                    spoken = "".join(assistant_bits).strip()
                    assistant_bits.clear()
                    if spoken:
                        log.append("assistant", spoken, source="speech")
                    ui.assistant_end()
                elif etype == "conversation.item.input_audio_transcription.completed":
                    transcript = event.get("transcript") or ""
                    if transcript:
                        ui.heard(transcript)
                        log.append("user", transcript, source="speech")

        async def ready_fallback() -> None:
            await asyncio.sleep(1.5)
            session_ready.set()

        fallback = asyncio.create_task(ready_fallback())
        tasks = [
            asyncio.create_task(send_typed()),
            asyncio.create_task(recv_loop()),
        ]
        if enable_mic:
            tasks.append(asyncio.create_task(send_mic()))

        try:
            done, _pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                if task.cancelled():
                    continue
                exc = task.exception()
                if exc:
                    raise exc
        finally:
            session_active = False
            session_ready.set()
            fallback.cancel()
            if playback:
                playback.stop()
            for task in tasks:
                task.cancel()
            leftover = "".join(assistant_bits).strip()
            if leftover:
                log.append("assistant", leftover, source="speech")
            ui.session("ended")
