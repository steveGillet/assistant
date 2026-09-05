"""One Grok Voice WebSocket session: mic and/or terminal stdin."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import queue
import re
import sys
import threading
import time

import websockets
from websockets.exceptions import ConnectionClosed, ConnectionClosedError, InvalidStatus

from grapefruit.commands import handle_line
from grapefruit.hold import HoldState
from grapefruit.memory import (
    ConversationLog,
    _short_title,
    recap_from_restore_text,
    spoken_restore_confirm,
)
from grapefruit.paths import ROOT, ensure_dirs
from grapefruit.protocol import (
    MIC_FRAMES,
    SAMPLE_RATE,
    VOICE_URI,
    force_message_event,
    function_output_event,
    input_audio_append_event,
    is_mute_command,
    is_silent_command,
    MUTE_TOOL_NAMES,
    SILENT_TOOL_NAMES,
    session_update_event,
    user_text_event,
)
from grapefruit import ui
from grapefruit.silent import SilentState, handle_silent_line
from grapefruit.tools import handle_tool

DEFAULT_IDLE_SEC = int(os.getenv("GROK_VOICE_IDLE_SEC", "600"))
DEFAULT_AUTO_MUTE_SEC = int(os.getenv("GROK_VOICE_AUTO_MUTE_SEC", "60"))


def _handshake_error_text(exc: InvalidStatus) -> str:
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None) or "?"
    body = getattr(response, "body", None) or b""
    if isinstance(body, (bytes, bytearray)):
        text = body.decode("utf-8", "replace").strip()
    else:
        text = str(body).strip()
    detail = text or str(exc)
    return f"Voice handshake failed (HTTP {status}): {detail}"


def _restore_spoken_title(result: str, fallback: str = "earlier chat") -> str:
    for pattern in (r"Restored conversation '([^']+)'", r'Restored conversation "([^"]+)"'):
        match = re.search(pattern, result)
        if match:
            return _short_title(match.group(1), width=80)
    return fallback


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


async def _muted_wait(
    hold: HoldState,
    typed: queue.Queue[str],
    log: ConversationLog,
) -> str:
    ui.session("muted · jobs keep running · chime when done")
    while True:
        finished = hold.consume_finished(log, chime=True)
        if finished:
            ui.status("job done · /unmute when you want the recap")
        try:
            text = typed.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.2)
            continue
        text = (text or "").strip()
        if not text:
            continue
        action = handle_line(text, log, muted=True)
        if action.kind == "quit":
            return "quit"
        if action.kind == "silent":
            ui.status(action.text)
            return "silent"
        if action.kind == "unmute":
            ui.status(action.text)
            return "unmute"
        if action.kind == "mute":
            ui.status("already muted")
            continue
        if action.kind == "status":
            ui.info(hold.status_text())
            continue
        if action.kind in {"print", "help", "save"}:
            ui.info(action.text)
            continue
        if action.kind == "restore":
            ui.info("unmute first, then restore another chat")
            continue
        ui.status("queued until unmute")
        typed.put(text)
        return "unmute"


async def _silent_wait(
    hold: HoldState,
    typed: queue.Queue[str],
    log: ConversationLog,
) -> str:
    ui.session("silent · grok cli · /unsilent or grapefruit for voice · /quit ends")
    hold.outcome = "silent"
    state = SilentState(log=log)
    while True:
        hold.consume_finished(log, chime=False)
        try:
            text = typed.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.2)
            continue
        text = (text or "").strip()
        if not text:
            continue
        outcome = handle_silent_line(text, state)
        if outcome in {"quit", "unsilent"}:
            return outcome


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
    idle_sec: int | None = None,
    auto_mute_sec: int | None = None,
    log: ConversationLog | None = None,
    hold: HoldState | None = None,
    park_in_process: bool = False,
    opening_line: str = "",
) -> str:
    cwd = cwd or os.getcwd()
    project = project or str(ROOT)
    ensure_dirs()
    headers = {"Authorization": f"Bearer {api_key}"}
    typed = typed if typed is not None else queue.Queue()
    hold = hold or HoldState()
    if log is None:
        log = ConversationLog()
        log.start(initial_text or restore_title or "")
    hold.log = log
    idle_limit_outer = DEFAULT_IDLE_SEC if idle_sec is None else int(idle_sec)
    mute_after_outer = DEFAULT_AUTO_MUTE_SEC if auto_mute_sec is None else int(auto_mute_sec)
    extra = extra_instructions or ""
    init = initial_text
    spoken_open = opening_line
    title_hint = restore_title
    while True:
        hold.outcome = "run"
        outcome = await _voice_leg(
            api_key=api_key,
            voice=voice,
            headers=headers,
            cwd=cwd,
            project=project,
            enable_mic=enable_mic,
            enable_speaker=enable_speaker,
            typed=typed,
            initial_text=init,
            extra_instructions=extra,
            restore_title=title_hint,
            mic_device=mic_device,
            mute_mic_while_speaking=mute_mic_while_speaking,
            voice_barge_in=voice_barge_in,
            idle_limit=idle_limit_outer,
            mute_after=mute_after_outer,
            log=log,
            hold=hold,
            opening_line=spoken_open,
        )
        extra = ""
        init = None
        spoken_open = ""
        title_hint = ""
        if outcome == "mute" and park_in_process:
            wait = await _muted_wait(hold, typed, log)
            if wait == "unmute":
                extra = log.context_for_model()
                spoken_open = hold.take_spoken_recap(
                    log.meta.title if log.meta else ""
                )
                continue
            if wait != "silent":
                return wait
            outcome = "silent"
        if outcome == "silent" and park_in_process:
            wait = await _silent_wait(hold, typed, log)
            if wait == "unsilent":
                extra = log.context_for_model()
                spoken_open = hold.take_spoken_recap(
                    log.meta.title if log.meta else ""
                )
                continue
            return wait
        return outcome
    return "idle"


async def _voice_leg(
    *,
    api_key: str,
    voice: str,
    headers: dict,
    cwd: str,
    project: str,
    enable_mic: bool,
    enable_speaker: bool,
    typed: queue.Queue[str],
    initial_text: str | None,
    extra_instructions: str,
    restore_title: str,
    mic_device: int | None,
    mute_mic_while_speaking: bool,
    voice_barge_in: bool,
    idle_limit: int,
    mute_after: int,
    log: ConversationLog,
    hold: HoldState,
    opening_line: str,
) -> str:
    playback = Playback() if enable_speaker else None
    mic_queue: queue.Queue[str] = queue.Queue()
    session_active = True
    end_requested = False
    pending_calls: list[dict] = []
    session_ready = asyncio.Event()
    extra_instructions = extra_instructions or ""
    assistant_bits: list[str] = []
    send_lock = asyncio.Lock()
    tool_tasks: set[asyncio.Task] = set()
    last_activity = time.monotonic()
    outcome = "idle"

    def bump_activity() -> None:
        nonlocal last_activity
        last_activity = time.monotonic()

    async def request_mute(reason: str = "muted") -> None:
        await request_park(reason, "mute")

    async def request_park(reason: str, kind: str) -> None:
        nonlocal session_active, outcome
        if hold.outcome == kind:
            return
        hold.outcome = kind
        outcome = kind
        ui.status(reason)
        if playback:
            playback.interrupt()
        session_active = False
        try:
            await ws.close()
        except Exception:
            pass

    try:
        ws = await websockets.connect(
            VOICE_URI,
            additional_headers=headers,
            max_size=None,
            ping_interval=20,
            ping_timeout=120,
        )
    except InvalidStatus as exc:
        ui.error(_handshake_error_text(exc))
        return "error"

    async with ws:

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
                    if name in MUTE_TOOL_NAMES:
                        result, _should_end = handle_tool(name, args)
                        ui.tool_result(result)
                        log.append("tool", result, source="tool", name=name)
                        await request_park("muting voice · jobs keep running", "mute")
                        return
                    if name in SILENT_TOOL_NAMES:
                        result, _should_end = handle_tool(name, args)
                        ui.tool_result(result)
                        log.append("tool", result, source="tool", name=name)
                        await request_park(
                            "silent · grok cli · say grapefruit or /unsilent for voice",
                            "silent",
                        )
                        return
                    exclude_ids = log.exclude_ids()
                    label = str(args.get("task") or args.get("query") or name)
                    job = hold.submit(
                        name,
                        lambda n=name, a=args, e=exclude_ids: handle_tool(n, a, e),
                        label=label,
                    )
                    while not job.done.is_set():
                        if not session_active:
                            ui.status("voice parked · job still running")
                            return
                        await asyncio.sleep(0.2)
                    if not session_active:
                        return
                    hold.take_finished()
                    result, should_end = job.result, job.should_end
                    ui.tool_result(result)
                    log.append("tool", result[:1500], source="tool", name=name)
                    bump_activity()
                    voice_output = result
                    if name == "restore_conversation" and result.startswith(
                        "Prior conversation loaded"
                    ):
                        match = re.search(r"\(id ([^)]+)\)", result)
                        if match:
                            log.note_restored(match.group(1))
                        extra_instructions = result
                        await push_session_update()
                        topic = _restore_spoken_title(result)
                        recap = recap_from_restore_text(result)
                        voice_output = (
                            spoken_restore_confirm(topic, recap)
                            + " Do not read the rest of the log aloud. "
                            "Do not use this session's title. "
                            "Do not call restore_conversation again unless the user "
                            "asks for a different earlier chat."
                        )
                    if should_end:
                        end_requested = True
                        hold.outcome = "quit"
                        outcome = "quit"
                    await safe_send(function_output_event(call_id, voice_output))
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
            recap = recap_from_restore_text(body)
            if recap and recap not in notice:
                ui.info(recap)
            if playback:
                playback.interrupt()
            await safe_send(
                force_message_event(spoken_restore_confirm(title, recap or notice))
            )

        async def send_typed() -> None:
            nonlocal session_active, end_requested
            await session_ready.wait()
            if opening_line:
                if playback:
                    playback.interrupt()
                await safe_send(force_message_event(opening_line))
            elif extra_instructions and not initial_text:
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
                bump_activity()
                action = handle_line(text, log, muted=False)
                if action.kind == "silent":
                    ui.status(action.text)
                    hold.outcome = "silent"
                    outcome = "silent"
                    session_active = False
                    try:
                        await ws.close()
                    except Exception:
                        pass
                    break
                if action.kind == "mute":
                    ui.user(text)
                    log.append("user", text, source="typed")
                    await request_mute("muting voice · jobs keep running")
                    continue
                if action.kind == "unmute":
                    ui.status("voice is already live")
                    continue
                if action.kind == "status":
                    ui.info(hold.status_text())
                    continue
                if action.kind == "quit":
                    ui.user("/quit")
                    end_requested = True
                    hold.outcome = "quit"
                    outcome = "quit"
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

        async def idle_watch() -> None:
            nonlocal session_active, outcome
            if idle_limit <= 0 and mute_after <= 0:
                return
            if idle_limit > 0:
                ui.status(f"voice idle timeout {idle_limit}s")
            if mute_after > 0:
                ui.status(f"auto-mute after {mute_after}s while a job runs")
            while session_active:
                await asyncio.sleep(2)
                if not session_active:
                    return
                if playback and playback.speaking():
                    bump_activity()
                    continue
                idle_for = time.monotonic() - last_activity
                if hold.busy():
                    if mute_after > 0 and idle_for >= mute_after:
                        await request_mute(
                            f"auto-mute after {mute_after}s while a job runs · socket parked"
                        )
                    continue
                if idle_limit > 0 and idle_for >= idle_limit:
                    hold.outcome = "idle"
                    outcome = "idle"
                    ui.status(
                        f"idle timeout after {idle_limit}s · closing voice so it does not keep using credits"
                    )
                    session_active = False
                    try:
                        await ws.close()
                    except Exception:
                        pass
                    return

        async def recv_loop() -> None:
            nonlocal session_active, end_requested, extra_instructions
            try:
                await _recv_loop()
            except (ConnectionClosed, ConnectionClosedError, TimeoutError, asyncio.TimeoutError) as exc:
                ui.error(f"voice timed out or disconnected ({exc})")
            except Exception as exc:
                ui.error(f"voice {exc}")
            finally:
                session_active = False

        async def _recv_loop() -> None:
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
                    msg = str(event)
                    if "timeout" in msg.lower():
                        ui.error(
                            "voice timed out and was stopped so it would not keep using credits"
                        )
                    else:
                        ui.error(msg)
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
                        bump_activity()
                        ui.heard(transcript)
                        log.append("user", transcript, source="speech")
                        if is_silent_command(transcript):
                            await request_park(
                                "silent · grok cli · say grapefruit or /unsilent for voice",
                                "silent",
                            )
                        elif is_mute_command(transcript):
                            await request_mute("muting voice · jobs keep running")

        async def ready_fallback() -> None:
            await asyncio.sleep(1.5)
            session_ready.set()

        fallback = asyncio.create_task(ready_fallback())
        tasks = [
            asyncio.create_task(send_typed()),
            asyncio.create_task(recv_loop()),
        ]
        if idle_limit > 0 or mute_after > 0:
            tasks.append(asyncio.create_task(idle_watch()))
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
                if exc and not isinstance(
                    exc,
                    (ConnectionClosed, ConnectionClosedError, TimeoutError, asyncio.TimeoutError),
                ):
                    raise exc
                if exc:
                    ui.error(f"voice timed out or disconnected ({exc})")
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
            if hold.outcome == "mute" or outcome == "mute":
                ui.session("muted")
                return "mute"
            if end_requested or hold.outcome == "quit":
                ui.session("ended")
                return "quit"
            ui.session("ended")
            return outcome if outcome in {"idle", "mute", "quit"} else "idle"
