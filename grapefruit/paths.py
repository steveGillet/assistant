from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = PACKAGE_DIR.parent
ASSETS = ROOT / "assets"
SCRIPTS = ROOT / "scripts"
GENERATED = ROOT / "generated"
CONVERSATIONS = ROOT / "conversations"
SESSION_FILE = ROOT / ".grok_cli_session"


def ensure_dirs() -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    CONVERSATIONS.mkdir(parents=True, exist_ok=True)


def vosk_model_dir() -> Path:
    for candidate in (
        ASSETS / "vosk-model-small-en-us-0.15",
        ROOT / "vosk-model-small-en-us-0.15",
    ):
        if candidate.is_dir():
            return candidate
    return ASSETS / "vosk-model-small-en-us-0.15"


def ack_wav() -> Path | None:
    for candidate in (ASSETS / "ack.wav", ROOT / "ack.wav"):
        if candidate.is_file():
            return candidate
    return None
