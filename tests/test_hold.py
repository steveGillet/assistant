from __future__ import annotations

import time

from grapefruit.hold import HoldState


def test_submit_and_take_finished():
    hold = HoldState()
    job = hold.submit(
        "run_grok",
        lambda: (time.sleep(0.2) or "did the thing", False),
        label="ssh into pi",
    )
    assert hold.busy()
    job.done.wait(timeout=2)
    assert not hold.busy()
    finished = hold.take_finished()
    assert len(finished) == 1
    assert finished[0].result == "did the thing"
    assert hold.take_finished() == []


def test_spoken_recap_mentions_finished_job():
    hold = HoldState()
    hold.recaps.append("SSH worked. Host is steve-desktop.")
    text = hold.take_spoken_recap("Check. Oh yeah. No. Can you restore the Raspberry Pi conversation?")
    assert "Still here" in text
    assert "steve-desktop" in text
    assert "Picked up" not in text
    assert "Oh yeah" not in text
    assert hold.recaps == []


def test_spoken_recap_skips_thinking_and_uses_the_answer():
    from grapefruit.memory import spoken_job_recap

    blob = (
        "I'll locate robot-controller.service and read its full contents. "
        "The broad search is slow, so I'll check the usual systemd paths. "
        "The unit file is on the Pi at /etc/systemd/system/robot-controller.service. "
        "It runs controller.py from twoWheeledRedemption after boot. "
        "Use systemctl status without sudo. Restart, stop, and start need sudo."
    )
    recap = spoken_job_recap(blob)
    assert "I'll locate" not in recap
    assert "controller.py" in recap or "systemctl" in recap
    hold = HoldState()
    hold.recaps.append(blob)
    spoken = hold.take_spoken_recap("garbled title here")
    assert spoken.startswith("Still here.")
    assert "garbled" not in spoken
    assert "While you were away" in spoken


def test_spoken_recap_still_running():
    hold = HoldState()
    hold.submit("run_grok", lambda: (time.sleep(3) or "x", False), label="long job")
    text = hold.take_spoken_recap("session")
    assert "still running" in text.lower()
    assert "long job" in text
