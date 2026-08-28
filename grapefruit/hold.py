"""Jobs that outlive a Grok Voice socket.

Mute parks Voice. CLI work keeps running here. When a job finishes
while muted we chime and stash a recap for unmute.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

from grapefruit.memory import spoken_job_recap
from grapefruit.tools import play_ack
from grapefruit import ui


@dataclass
class Job:
    name: str
    label: str
    fn: Callable[[], tuple[str, bool]]
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    result: str = ""
    should_end: bool = False
    announced: bool = False
    done: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None

    def start(self) -> None:
        def run() -> None:
            try:
                result, should_end = self.fn()
                self.result = result
                self.should_end = should_end
            except Exception as exc:
                self.result = f"Job failed: {exc}"
                self.should_end = False
            finally:
                self.done.set()

        self.thread = threading.Thread(
            target=run, daemon=True, name=f"grapefruit-job-{self.job_id}"
        )
        self.thread.start()


class HoldState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.jobs: list[Job] = []
        self.recaps: list[str] = []
        self.outcome: str = "run"
        self.log = None

    def busy(self) -> bool:
        with self.lock:
            return any(not job.done.is_set() for job in self.jobs)

    def running_labels(self) -> list[str]:
        with self.lock:
            return [job.label for job in self.jobs if not job.done.is_set()]

    def status_text(self) -> str:
        running = self.running_labels()
        recaps = list(self.recaps)
        lines = []
        if running:
            lines.append("Running: " + "; ".join(running))
        else:
            lines.append("No jobs running.")
        if recaps:
            lines.append("Finished while muted:")
            for item in recaps:
                lines.append(item)
        return "\n".join(lines)

    def submit(
        self,
        name: str,
        fn: Callable[[], tuple[str, bool]],
        *,
        label: str = "",
    ) -> Job:
        job = Job(name=name, label=label or name, fn=fn)
        with self.lock:
            self.jobs.append(job)
        job.start()
        return job

    def take_finished(self) -> list[Job]:
        with self.lock:
            ready = [job for job in self.jobs if job.done.is_set() and not job.announced]
            for job in ready:
                job.announced = True
            return ready

    def consume_finished(self, log=None, *, chime: bool) -> list[Job]:
        jobs = self.take_finished()
        for job in jobs:
            if chime:
                try:
                    play_ack()
                except Exception:
                    pass
            ui.tool_result(job.result or "")
            if log is not None and job.result:
                log.append("tool", job.result[:1500], source="tool", name=job.name)
            if chime and job.result:
                with self.lock:
                    self.recaps.append(job.result)
        return jobs

    def take_spoken_recap(self, title: str = "") -> str:
        with self.lock:
            recaps = list(self.recaps)
            self.recaps.clear()
            running = [job.label for job in self.jobs if not job.done.is_set()]
        bits = ["Still here."]
        if recaps:
            joined = "\n".join(recaps)
            bits.append("While you were away: " + spoken_job_recap(joined))
        elif running:
            bits.append("That job is still running: " + ", ".join(running) + ".")
        return " ".join(bits)

    def has_unmute_news(self) -> bool:
        with self.lock:
            if self.recaps:
                return True
            return any(not job.done.is_set() for job in self.jobs)


